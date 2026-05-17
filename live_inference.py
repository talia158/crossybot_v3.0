import heapq
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import math

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PRUNE_AFTER_FRAMES = 30   # match BoT-SORT's track_buffer; once it drops a track, we can too
EDGE_EPS           = 2    # px tolerance for "bbox edge touches frame border"


@dataclass
class _TrackState:
    first_ts:     float
    last_ts:      float
    first_cam_x:  float   # hxscroll.cum_scroll_x at track-init; subtract to get world disp
    cum_disp:     float   # cumulative signed horizontal displacement, image frame (px)
    cum_disp_y:   float   # cumulative signed vertical displacement (px)
    prev_x1:      float
    prev_x2:      float
    prev_cx:      float
    prev_cy:      float
    last_seen:    int     # frame index of most recent update


class VelocityTracker:
    def __init__(self):
        self._tracks: dict = {}

    def update(self, tid: int, x1: float, y1: float, x2: float, y2: float,
               ts: float, frame_idx: int, cam_x: float = 0.0):
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        clipped = x1 <= EDGE_EPS or x2 >= W - EDGE_EPS
        s = self._tracks.get(tid)
        if s is None:
            # Entry phase: defer initialization until the bbox is fully in view.
            if clipped:
                return
            self._tracks[tid] = _TrackState(
                first_ts=ts, last_ts=ts, first_cam_x=cam_x,
                cum_disp=0.0, cum_disp_y=0.0,
                prev_x1=x1, prev_x2=x2, prev_cx=cx, prev_cy=cy,
                last_seen=frame_idx,
            )
            return
        # Exit phase: bbox now touches a screen edge — freeze velocity at its
        # last known value. Just refresh last_seen so we don't prune prematurely.
        if clipped:
            s.last_seen = frame_idx
            return
        # Fully in view: accumulate horizontal and vertical centroid deltas.
        s.cum_disp   += cx - s.prev_cx
        s.cum_disp_y += cy - s.prev_cy
        s.last_ts   = ts
        s.prev_x1   = x1
        s.prev_x2   = x2
        s.prev_cx   = cx
        s.prev_cy   = cy
        s.last_seen = frame_idx

    def velocity(self, tid: int, cam_x_now: float = 0.0) -> Optional[float]:
        """World-frame horizontal velocity in px/s.

        `cum_disp` is image-frame displacement; the camera's cumulative
        right-pan over the same window contributes `-cam_disp` to it.
        Inverting: world_disp = cum_disp + cam_disp."""
        s = self._tracks.get(tid)
        if s is None:
            return None
        dt = s.last_ts - s.first_ts
        if dt <= 0:
            return None
        cam_disp = cam_x_now - s.first_cam_x
        return (s.cum_disp + cam_disp) / dt

    def nearest_cum_disp_y(self, char_cx: float, char_cy: float, k: int = 3) -> float:
        """Mean cum_disp_y of the k tracks whose centroid is closest to (char_cx, char_cy)."""
        if not self._tracks:
            return 0.0
        by_dist = sorted(
            self._tracks.values(),
            key=lambda s: (s.prev_cx - char_cx) ** 2 + (s.prev_cy - char_cy) ** 2,
        )
        chosen = by_dist[:k]
        return sum(s.cum_disp_y for s in chosen) / len(chosen)

    def prune(self, current_frame: int):
        """Drop tracks that haven't appeared in PRUNE_AFTER_FRAMES frames."""
        stale = [tid for tid, s in self._tracks.items()
                 if current_frame - s.last_seen > PRUNE_AFTER_FRAMES]
        for tid in stale:
            del self._tracks[tid]


class _OneEuro1D:
    """One Euro filter (Casiez et al. 2012). Adaptive low-pass where the
    cutoff frequency scales with the magnitude of the estimated derivative:
    `cutoff = mincutoff + beta * |dx/dt|`. At rest the cutoff collapses to
    `mincutoff` (heavy smoothing → kills jitter); during fast motion the
    cutoff opens (low lag → tracks the hop)."""

    def __init__(self, mincutoff: float = 1.0, beta: float = 0.05,
                 dcutoff: float = 1.0):
        self.mincutoff = mincutoff
        self.beta      = beta
        self.dcutoff   = dcutoff
        self._x_prev:  Optional[float] = None
        self._dx_prev: float           = 0.0
        self._t_prev:  Optional[float] = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self._t_prev is None or self._x_prev is None:
            self._x_prev, self._t_prev = x, t
            return x
        dt = max(t - self._t_prev, 1e-3)
        dx = (x - self._x_prev) / dt
        ad = self._alpha(self.dcutoff, dt)
        dx_hat = ad * dx + (1.0 - ad) * self._dx_prev
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev, self._dx_prev, self._t_prev = x_hat, dx_hat, t
        return x_hat


GRID_PERIOD     = 21       # px per lane (one hop = one period)
_GRID_LANE_MARGIN = 1      # px gap rendered between adjacent lanes
_GRID_X_RANGE   = (20, 180)
_GRID_Y_RANGE   = (30, 420)
_GRID_JUMP_CLAMP = 15      # px; |delta| above this is treated as a scene cut


class HorizontalScrollEstimator:
    """Infer cumulative horizontal world-scroll via 1-D phase correlation on
    the column-collapsed Sobel-x signal of the ROI.

    Per frame: gray ROI → Sobel-x (vertical-edge field) → zero gradient inside
    moving-object bboxes (per-bbox, not row-wide) → collapse |sx| over the
    vertical axis to a 1-D column-edge-density signal c[x] → subtract a
    boxcar moving average (kills DC + lighting tilt) → Hann-window → 1-D
    phase correlation against the previous frame's c → argmax + parabolic
    sub-pixel → signed shift. Camera-right convention: shift = -dx.
    """

    def __init__(self, lane_h: int = GRID_PERIOD, sobel_thresh: int = 21,
                 hp_win: int = GRID_PERIOD,
                 jump_clamp_x: int = _GRID_JUMP_CLAMP):
        # lane_h kept only for trackbar compatibility — no longer used in math.
        self.lane_h:        int                  = max(2, int(lane_h))
        self.sobel_thresh:  int                  = max(1, int(sobel_thresh))
        self.hp_win:        int                  = max(3, int(hp_win) | 1)
        self.jump_clamp_x:  float                = float(jump_clamp_x)
        self._prev_cw:      Optional[np.ndarray] = None
        self._hann1d:       Optional[np.ndarray] = None
        self.cum_scroll_x:  float                = 0.0
        self.last_shift:    float                = 0.0
        self.last_dy:       float                = 0.0
        self.last_response: float                = 0.0
        self.last_binary:   Optional[np.ndarray] = None  # binarised |sx|, for `x` viewer

    def set_lane_h(self, h: int) -> None:
        self.lane_h = max(2, int(h))

    def set_sobel_thresh(self, t: int) -> None:
        self.sobel_thresh = max(1, int(t))

    def flush(self):
        self._prev_cw = None
        self.last_shift = 0.0

    def update(self, frame: np.ndarray,
               bboxes: Optional[list] = None) -> float:
        x0, x1 = _GRID_X_RANGE
        y0, y1 = _GRID_Y_RANGE
        roi  = frame[y0:y1, x0:x1, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)

        if bboxes:
            roi_h, roi_w = sx.shape
            for x_a, y_a, x_b, y_b in bboxes:
                rx1 = max(0, int(x_a) - x0)
                ry1 = max(0, int(y_a) - y0)
                rx2 = min(roi_w, int(x_b) - x0)
                ry2 = min(roi_h, int(y_b) - y0)
                if rx1 < rx2 and ry1 < ry2:
                    sx[ry1:ry2, rx1:rx2] = 0.0

        abs_sx = np.abs(sx)
        self.last_binary = (abs_sx >= self.sobel_thresh).astype(np.uint8) * 255

        c = abs_sx.sum(axis=0).astype(np.float32)
        N = c.shape[0]

        # High-pass: subtract a length-hp_win moving average (kills DC / lighting tilt).
        w = self.hp_win if (self.hp_win % 2 == 1) else self.hp_win + 1
        if 3 <= w < N:
            kernel = np.ones(w, dtype=np.float32) / w
            c = c - np.convolve(c, kernel, mode="same")

        if self._hann1d is None or self._hann1d.shape[0] != N:
            self._hann1d = np.hanning(N).astype(np.float32)
        cw = c * self._hann1d

        if self._prev_cw is None or self._prev_cw.shape != cw.shape:
            self._prev_cw = cw
            self.last_shift = 0.0
            return self.cum_scroll_x

        F  = np.fft.rfft(cw)
        Fp = np.fft.rfft(self._prev_cw)
        R  = F * np.conj(Fp)
        R  = R / (np.abs(R) + 1e-8)
        r  = np.fft.irfft(R, n=N)

        k_raw    = int(np.argmax(r))
        k_signed = k_raw - N if k_raw > N // 2 else k_raw

        sm = float(r[(k_raw - 1) % N])
        sc = float(r[k_raw])
        sp = float(r[(k_raw + 1) % N])
        denom = sm - 2.0 * sc + sp
        if abs(denom) > 1e-9:
            sub = float(np.clip((sm - sp) / (2.0 * denom), -1.0, 1.0))
        else:
            sub = 0.0
        dx = float(k_signed) + sub

        if abs(dx) > self.jump_clamp_x:
            dx = 0.0

        shift = -dx
        self.last_shift     = shift
        self.last_response  = sc
        self.last_dy        = 0.0
        self.cum_scroll_x  += shift
        self._prev_cw       = cw
        return self.cum_scroll_x


class GridScrollEstimator:
    """Estimate cumulative vertical world-scroll by fitting a periodic lane grid.

    Per frame: gray ROI → vertical Sobel → zero the gradient inside any
    moving-object bboxes (so cars/logs don't pollute the edge density) →
    sum |∂I/∂y| per row → reshape and sum across periods to score every
    offset o ∈ [0, P) → argmax + 3-point parabolic interpolation gives a
    sub-pixel offset → phase-unwrap across frames into a monotonic
    cumulative scroll.
    """

    def __init__(self, lane_h: int = GRID_PERIOD, sobel_thresh: int = 21):
        self.lane_h:        int               = max(2, int(lane_h))
        self.sobel_thresh:  int               = max(1, int(sobel_thresh))
        self.prev_offset:   Optional[float]   = None
        self.cum_scroll_y:  float             = 0.0
        self.last_score:    Optional[np.ndarray] = None
        self.last_sob:      Optional[np.ndarray] = None  # masked Sobel, for debug viz
        self.last_binary:   Optional[np.ndarray] = None  # binarised Sobel, for viz/score

    def set_lane_h(self, h: int) -> None:
        h = max(2, int(h))
        if h == self.lane_h:
            return
        self.lane_h = h
        self.prev_offset = None      # previous phase reference is in old units

    def set_sobel_thresh(self, t: int) -> None:
        self.sobel_thresh = max(1, int(t))

    def update(self, frame: np.ndarray,
               bboxes: Optional[list] = None) -> float:
        x0, x1 = _GRID_X_RANGE
        y0, y1 = _GRID_Y_RANGE
        roi  = frame[y0:y1, x0:x1, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sob  = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        # Zero the gradient response inside moving-object bboxes (translated
        # from absolute frame coords into ROI coords) so cars/logs don't
        # contribute to the edge-density signal.
        if bboxes:
            roi_h, roi_w = sob.shape
            for x_a, y_a, x_b, y_b in bboxes:
                rx1 = max(0, int(x_a) - x0)
                ry1 = max(0, int(y_a) - y0)
                rx2 = min(roi_w, int(x_b) - x0)
                ry2 = min(roi_h, int(y_b) - y0)
                if rx1 < rx2 and ry1 < ry2:
                    sob[ry1:ry2, rx1:rx2] = 0.0
        self.last_sob = sob

        binary = (np.abs(sob) >= self.sobel_thresh).astype(np.uint8)
        self.last_binary = binary
        e = binary.sum(axis=1).astype(np.float32)                      # (N,)
        N = e.shape[0]

        P = self.lane_h
        n = N // P
        if n < 1:
            self.prev_offset = None
            return self.cum_scroll_y
        score = e[:n * P].reshape(n, P).sum(axis=0)                    # (P,)
        self.last_score = score
        o_star = int(np.argmax(score))

        # Sub-pixel parabolic interpolation around the peak (mod-P wrap).
        sm = float(score[(o_star - 1) % P])
        sc = float(score[o_star])
        sp = float(score[(o_star + 1) % P])
        denom = sm - 2.0 * sc + sp
        if abs(denom) > 1e-9:
            sub = float(np.clip((sm - sp) / (2.0 * denom), -1.0, 1.0))
        else:
            sub = 0.0
        o_star_f = float((o_star + sub) % P)

        # Phase-unwrap into cumulative scroll.
        if self.prev_offset is None:
            delta = 0.0
        else:
            raw   = o_star_f - self.prev_offset
            delta = (raw + P / 2.0) % P - P / 2.0
            if abs(delta) > _GRID_JUMP_CLAMP:
                delta = 0.0
        self.cum_scroll_y += delta
        self.prev_offset   = o_star_f
        return self.cum_scroll_y


EMULATOR   = "/Users/talia/Library/Android/sdk/emulator/emulator"
AVD_NAME   = "Small_Phone"
MODEL_PATH = "model_weight/model.pt"
W, H       = 240, 480
FIFO_PATH  = "/tmp/crossybot_screen.h264"
MAX_FPS    = 30


def start_avd():
    # Force a clean cold boot: kill any running emulator, then start without
    # loading a saved snapshot so the AVD boots from scratch every run.
    print("Killing any running emulator…")
    subprocess.run(["adb", "emu", "kill"], capture_output=True, timeout=5)
    time.sleep(2)

    print("Starting AVD (cold boot)…")
    proc = subprocess.Popen(
        [EMULATOR, "-avd", AVD_NAME,
         "-no-audio", "-no-boot-anim", "-no-snapshot-load"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print("Waiting for boot…")
    for _ in range(60):
        r = subprocess.run(
            ["adb", "shell", "getprop", "sys.boot_completed"],
            capture_output=True, text=True,
        )
        if r.stdout.strip() == "1":
            print("AVD booted.")
            return proc
        time.sleep(2)
    proc.terminate()
    raise RuntimeError("AVD did not boot within 2 minutes")


def launch_app():
    r = subprocess.run(
        ["adb", "shell", "pm", "list", "packages"],
        capture_output=True, text=True,
    )
    pkg = next(
        (line.replace("package:", "").strip()
         for line in r.stdout.splitlines()
         if "cross" in line.lower()),
        None,
    )
    if pkg is None:
        raise RuntimeError("Could not find a Crossy Road package on the device")
    print(f"Launching {pkg}…")
    subprocess.run(
        ["adb", "shell", "monkey", "-p", pkg,
         "-c", "android.intent.category.LAUNCHER", "1"],
        capture_output=True,
    )
    time.sleep(5)


class FrameStream:
    """
    Streams H.264 from the device via adb screenrecord → FIFO → cv2.VideoCapture.
    A background thread reads continuously; latest_frame always has the newest image.
    Auto-restarts when screenrecord hits its 3-minute limit.
    """

    def __init__(self):
        self._latest: np.ndarray | None = None
        self._lock   = threading.Lock()
        self._ready  = threading.Event()
        self._running = True
        self._start_pipeline()
        threading.Thread(target=self._reader_loop, daemon=True).start()

    def _start_pipeline(self):
        fifo = FIFO_PATH
        if os.path.exists(fifo):
            os.unlink(fifo)
        os.mkfifo(fifo)

        def _writer():
            with open(fifo, "wb") as f:
                self._adb_proc = subprocess.Popen(
                    ["adb", "exec-out", "screenrecord",
                     "--output-format=h264",
                     f"--size={W}x{H}",
                     "/dev/stdout"],
                    stdout=f, stderr=subprocess.DEVNULL,
                )
                self._adb_proc.wait()

        threading.Thread(target=_writer, daemon=True).start()
        self._cap = cv2.VideoCapture(fifo)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _reader_loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                self._cap.release()
                self._start_pipeline()
                continue
            with self._lock:
                self._latest = frame
            self._ready.set()

    def next_frame(self) -> np.ndarray:
        self._ready.wait()
        with self._lock:
            return self._latest

    def stop(self):
        self._running = False
        self._cap.release()
        if hasattr(self, "_adb_proc"):
            self._adb_proc.terminate()
        if os.path.exists(FIFO_PATH):
            os.unlink(FIFO_PATH)


ROTATE_DEG = 14.5   # clockwise
SHEAR_DEG  = 14.5   # clockwise x-shear

# Pre-compute combined affine matrix (rotation then shear) at module load time.
# Frame size is fixed (W × H), so we only do this once.
def _build_transform(w: int, h: int) -> np.ndarray:
    M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), ROTATE_DEG, 1.0)
    R = np.vstack([M_rot, [0, 0, 1]])               # 2×3 → 3×3
    s = math.tan(math.radians(SHEAR_DEG))
    S = np.array([[1, s, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return (S @ R)[:2].astype(np.float32)           # back to 2×3

_TRANSFORM = _build_transform(W, H)


def transform_frame(frame: np.ndarray, shift_x: int = 0) -> np.ndarray:
    """Rotate+shear then translate left by `shift_x` px. Equivalent to
    composing T @ S @ R: rotation first, shear next, translation last."""
    M = _TRANSFORM.copy()
    M[0, 2] -= float(shift_x)
    return cv2.warpAffine(frame, M, (W, H))


YOLO_CONF       = 0.8
CHAR_INIT_XY    = (223, 315)   # initial centroid of the chicken on screen

# Morphology-based character detector (ported from crossybot_v2.0
# CharacterTools.detect_character). BGR-threshold the chicken's beak/body
# colour, branch on raw mask population for the right open/close/dilate
# recipe, pick the largest valid CC. Centroid is shifted down to land on
# the character's feet rather than the colored beak.
_CHAR_BGR_TARGET   = (92, 172, 255)
_CHAR_TOL          = 4
_GREEN_JUMP_THRESH = 30.0  # px/frame; reject morphology jumps above this as glitches
_GREEN_GLITCH_LIMIT = 30   # frames; force-accept after this many consecutive rejects
_CHAR_MIN_AREA_FRAC = 0.0005
_CHAR_MAX_AREA_FRAC = 0.05
_CHAR_OPEN_KSZ     = 3
_CHAR_CLOSE_KSZ    = 5
_CHAR_Y_SHIFT      = 11


def detect_character(bgr: np.ndarray) -> Optional[tuple[int, int]]:
    H_, W_ = bgr.shape[:2]
    B0, G0, R0 = _CHAR_BGR_TARGET
    tol = _CHAR_TOL
    lo = np.array([max(0, B0 - tol), max(0, G0 - tol), max(0, R0 - tol)],
                  dtype=np.uint8)
    hi = np.array([min(255, B0 + tol), min(255, G0 + tol), min(255, R0 + tol)],
                  dtype=np.uint8)
    mask_raw = cv2.inRange(bgr, lo, hi)
    n_on = int(cv2.countNonZero(mask_raw))
    K3  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    K35 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    if n_on <= 12:
        m = cv2.dilate(mask_raw, K3, iterations=2)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, K35, iterations=1)
        m = cv2.medianBlur(m, 3)
        min_area_abs = 2
    elif n_on <= 80:
        m = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, K3, iterations=1)
        m = cv2.dilate(m, K3, iterations=1)
        min_area_abs = 8
    else:
        ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (_CHAR_OPEN_KSZ, _CHAR_OPEN_KSZ))
        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (_CHAR_CLOSE_KSZ, _CHAR_CLOSE_KSZ))
        m = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, ko, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc, iterations=1)
        min_area_abs = int(max(1, _CHAR_MIN_AREA_FRAC * H_ * W_))

    num, _, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    max_area_abs = int(_CHAR_MAX_AREA_FRAC * H_ * W_)
    best_i, best_area = -1, -1
    for i in range(1, num):
        x, y, w, h, a = stats[i]
        if a < min_area_abs or a > max_area_abs:
            continue
        if w <= 1 and h <= 1:
            continue
        if a > best_area:
            best_i, best_area = i, a

    if best_i > 0:
        cx, cy = map(int, centroids[best_i])
        return (cx, int(np.clip(cy + _CHAR_Y_SHIFT, 0, H_ - 1)))
    if n_on > 0:
        ys, xs = np.where(mask_raw > 0)
        cx = int(np.clip(round(xs.mean()), 0, W_ - 1))
        cy = int(np.clip(round(ys.mean()), 0, H_ - 1))
        return (cx, int(np.clip(cy + _CHAR_Y_SHIFT, 0, H_ - 1)))
    return None


class GameOverDetector:
    """Detect Crossy Road's orange/yellow game-over banner via HSV
    thresholding (ported from crossybot_v2.0). Requires the masked
    fraction to exceed `min_frac` for `consec_needed` frames in a row
    to trigger, suppressing transient banner-coloured pixels."""

    def __init__(self,
                 hsv_lo: tuple        = (17, 160, 200),
                 hsv_hi: tuple        = (30, 255, 255),
                 min_frac: float      = 0.015,
                 consec_needed: int   = 3):
        self.lo            = np.array(hsv_lo, dtype=np.uint8)
        self.hi            = np.array(hsv_hi, dtype=np.uint8)
        self.min_frac      = float(min_frac)
        self.consec_needed = int(consec_needed)
        self._streak       = 0
        self.triggered     = False
        self._k_open       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._k_close      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def update(self, frame_bgr: np.ndarray) -> tuple[bool, float]:
        hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lo, self.hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._k_open,  iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._k_close, iterations=1)
        frac = cv2.countNonZero(mask) / mask.size
        seen = frac >= self.min_frac
        self._streak  = (self._streak + 1) if seen else 0
        self.triggered = self._streak >= self.consec_needed
        return self.triggered, float(frac)

    def reset(self) -> None:
        self._streak   = 0
        self.triggered = False


_KEYCODES     = {"up": 19, "down": 20, "left": 21, "right": 22}  # Android KEYCODE_DPAD_*
ANIM_DURATION = 0.3   # seconds per button-press animation
ANIM_DIST     = 22    # px of character displacement per press (= one lane)
_ANIM_K       = 10.0  # decay rate; 1/k ≈ 0.10 s = time of peak velocity

# D-pad button regions (x1, y1, x2, y2) in frame coordinates.
# Placed in the bottom-right corner of the 240×480 frame.
_BTNS: dict[str, tuple[int, int, int, int]] = {
    "up":    (181, 421, 207, 447),
    "left":  (152, 450, 178, 476),
    "down":  (181, 450, 207, 476),
    "right": (210, 450, 236, 476),
}
_BTN_LABELS = {"up": "^", "down": "v", "left": "<", "right": ">"}

# Each entry: (start_time, dx, dy) for an in-flight button animation.
_active_anims: list[tuple[float, float, float]] = []


def _anim_progress(elapsed: float) -> float:
    """Normalized position in [0, 1] for the given elapsed time.

    Uses p(t) ∝ 1 - (1 + kt)·exp(−kt), whose derivative is v(t) ∝ kt·exp(−kt):
    velocity rises to a sharp peak at t = 1/k then decays exponentially to zero.
    """
    def f(x: float) -> float:
        kx = _ANIM_K * x
        return 1.0 - (1.0 + kx) * math.exp(-kx)
    denom = f(ANIM_DURATION)
    return f(min(elapsed, ANIM_DURATION)) / denom if denom > 1e-9 else 1.0


class RunLogger:
    """JSONL run logger. One file per launch, line-buffered. Writes:
        {"type":"header", "wall_start":..., "perf_start":...}     once
        {"type":"frame",  "t":..., "green":..., "red":..., "truth":..., "objects":[...]}  per frame
        {"type":"key",    "t":..., "direction":"up|down|left|right"}                       per keystroke
    `t` is seconds since logger init via time.perf_counter() (monotonic, ~µs)."""

    def __init__(self, dir_path: str = "logs"):
        os.makedirs(dir_path, exist_ok=True)
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(dir_path, f"run_{ts}.jsonl")
        self._f   = open(self.path, "w", buffering=1)
        self._t0  = time.perf_counter()
        self._lock = threading.Lock()
        self._write({"type": "header",
                     "wall_start": time.time(),
                     "perf_start": self._t0})

    def _t(self) -> float:
        return time.perf_counter() - self._t0

    def _write(self, rec: dict) -> None:
        line = json.dumps(rec, default=float) + "\n"
        with self._lock:
            self._f.write(line)

    def log_frame(self, green_xy, red_xy, truth_xy, objects) -> None:
        self._write({
            "type":    "frame",
            "t":       self._t(),
            "green":   list(green_xy) if green_xy is not None else None,
            "red":     list(red_xy),
            "truth":   list(truth_xy),
            "objects": objects,
        })

    def log_key(self, direction: str) -> None:
        self._write({"type": "key", "t": self._t(), "direction": direction})

    def close(self) -> None:
        try:
            with self._lock:
                self._f.close()
        except Exception:
            pass


_run_logger: Optional[RunLogger] = None


def send_direction(direction: str):
    if _run_logger is not None:
        _run_logger.log_key(direction)
    cmd = ["adb", "shell", "input", "keyevent", str(_KEYCODES[direction])]
    threading.Thread(target=lambda: subprocess.run(cmd, capture_output=True),
                     daemon=True).start()
    adx, ady = {"up": (0, -ANIM_DIST), "down": (0, ANIM_DIST),
                "left": (-ANIM_DIST, 0), "right": (ANIM_DIST, 0)}[direction]
    _active_anims.append((time.time(), adx, ady))


def send_game_over_recovery():
    """Game-over recovery: press DPAD_DOWN 4× then ENTER. Sent as a
    single adb invocation so the keys arrive in order."""
    cmd = ["adb", "shell", "input", "keyevent",
           "20", "20", "20", "20", "66"]
    threading.Thread(target=lambda: subprocess.run(cmd, capture_output=True),
                     daemon=True).start()


def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for direction, (x1, y1, x2, y2) in _BTNS.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                send_direction(direction)
                break


def _draw_buttons(frame: np.ndarray) -> np.ndarray:
    for direction, (x1, y1, x2, y2) in _BTNS.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 50), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
        lbl = _BTN_LABELS[direction]
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, lbl,
                    (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return frame


class Planner:
    """A* search over (n_x, n_y, depth) states with moves up / wait /
    left / right. Cost structure encodes the user's priority:
        up   = 1   (always tried when safe)
        wait = 2   (tried only when current position is *not* in danger)
        side = 3   (tried only when staying put would get us hit)
    Heuristic: lanes-still-needed-to-reach-goal — admissible because
    the cheapest move advances one lane at unit cost.
    Returns the first move of the chosen path; "wait" is mapped to None."""

    _DELTAS = {
        "up":    (0,           -ANIM_DIST),
        "left":  (-ANIM_DIST,  0),
        "right": (ANIM_DIST,   0),
    }

    def __init__(self,
                 char_size: int          = 22,
                 t_horizon: float        = ANIM_DURATION,
                 obstacle_classes: tuple = (0, 1),  # cars, trains
                 log_class: int          = 2,
                 goal_lanes_ahead: int   = 2,
                 max_depth: int          = 5,
                 max_lateral: int        = 3,
                 n_time_samples: int     = 5):
        self.char_size        = char_size
        self.t_horizon        = t_horizon
        self.obstacle_classes = set(obstacle_classes)
        self.log_class        = log_class
        self.goal_lanes_ahead = goal_lanes_ahead
        self.max_depth        = max_depth
        self.max_lateral      = max_lateral
        self.n_time_samples   = max(2, int(n_time_samples))

    def plan(self, char_xy: tuple[float, float],
             objects: list) -> Optional[str]:
        cx0, cy0 = char_xy
        half = self.char_size / 2.0
        ad   = ANIM_DIST

        # Search-trace state for visualisation.
        self.last_origin: tuple        = (cx0, cy0)
        self.last_explored: list       = []   # [(nx, ny, depth), ...]
        self.last_path: list           = []   # [(nx, ny), ...]
        self.last_first_move: Optional[str] = None
        self.last_goal_reached: bool   = False

        def box_at(nx: int, ny: int) -> tuple:
            cx = cx0 + nx * ad
            cy = cy0 + ny * ad
            return (cx - half, cy - half, cx + half, cy + half)

        def heur(ny: int) -> int:
            return max(0, self.goal_lanes_ahead + ny)

        counter = 0
        init_state = (0, 0, 0)
        init_path  = ((0, 0),)
        heap: list = [(heur(0), counter, 0, init_state, None, init_path)]
        counter += 1
        visited: set = set()

        best_first = None
        best_ny    = 1
        best_g     = 0
        best_path  = init_path

        while heap:
            f, _, g, state, first_move, path = heapq.heappop(heap)
            if state in visited:
                continue
            visited.add(state)
            nx, ny, depth = state
            self.last_explored.append((nx, ny, depth))

            if ny <= -self.goal_lanes_ahead:
                self.last_path = list(path)
                self.last_first_move = first_move
                self.last_goal_reached = True
                return None if first_move == "wait" else first_move

            if ny < best_ny or (ny == best_ny and g < best_g):
                best_ny    = ny
                best_g     = g
                best_first = first_move
                best_path  = path

            if depth >= self.max_depth:
                continue

            t_offset    = depth * self.t_horizon
            wait_box    = box_at(nx, ny)
            wait_unsafe = self._is_unsafe(wait_box, objects, t_offset)

            candidates = []
            up_box = box_at(nx, ny - 1)
            if not self._is_unsafe(up_box, objects, t_offset, prev_box=wait_box):
                candidates.append(("up", (nx, ny - 1, depth + 1), 1, (nx, ny - 1)))
            if not wait_unsafe:
                candidates.append(("wait", (nx, ny, depth + 1), 2, (nx, ny)))
            if wait_unsafe:
                for name, dx in (("left", -1), ("right", 1)):
                    new_nx = nx + dx
                    if abs(new_nx) > self.max_lateral:
                        continue
                    side_box = box_at(new_nx, ny)
                    if not self._is_unsafe(side_box, objects, t_offset,
                                           prev_box=wait_box):
                        candidates.append((name, (new_nx, ny, depth + 1), 3, (new_nx, ny)))

            for move_name, new_state, step_cost, new_pos in candidates:
                new_g = g + step_cost
                new_f = new_g + heur(new_state[1])
                new_first = move_name if first_move is None else first_move
                new_path  = path + (new_pos,)
                heapq.heappush(heap,
                               (new_f, counter, new_g, new_state, new_first, new_path))
                counter += 1

        self.last_path = list(best_path)
        self.last_first_move = best_first
        self.last_goal_reached = False
        if best_first is None or best_first == "wait":
            return None
        return best_first

    def _is_unsafe(self, char_box, objects, t_offset: float = 0.0,
                   prev_box: Optional[tuple] = None) -> bool:
        """`char_box` is unsafe over [t_offset, t_offset+t_horizon] if (a)
        any car/train overlaps the chicken at any of N sampled time
        points along the window, or (b) the row is a water row at arrival
        but no log overlaps the destination at arrival.

        If `prev_box` is given, the chicken's box at sample dt is
        interpolated between `prev_box` and `char_box` using the same
        burst-decay curve `_anim_progress` that drives the on-screen
        animation. With `prev_box=None` (or `prev_box==char_box`), the
        chicken stays at `char_box` for the whole window — used for
        the in-danger / wait-safety check."""
        n     = self.n_time_samples
        t0    = t_offset
        t_end = t_offset + self.t_horizon
        sample_dt = [self.t_horizon * i / (n - 1) for i in range(n)]

        if prev_box is None or prev_box == char_box:
            char_at = lambda _dt: char_box
        else:
            px1, py1, px2, py2 = prev_box
            cx1, cy1, cx2, cy2 = char_box
            def char_at(dt: float) -> tuple:
                a = _anim_progress(dt)
                return (px1 + a * (cx1 - px1), py1 + a * (cy1 - py1),
                        px2 + a * (cx2 - px2), py2 + a * (cy2 - py2))

        cy_lo, cy_hi = char_box[1], char_box[3]
        log_in_row = False
        on_log     = False
        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            vx = obj.get("vx", 0.0)
            vy = obj.get("vy", 0.0)
            cls = obj.get("cls")
            if cls in self.obstacle_classes:
                for dt in sample_dt:
                    t = t0 + dt
                    cb = char_at(dt)
                    ob = (x1 + vx * t, y1 + vy * t,
                          x2 + vx * t, y2 + vy * t)
                    if self._intersects(cb, ob):
                        return True
            elif cls == self.log_class:
                # Log riding is decided at arrival time only.
                box = (x1 + vx * t_end, y1 + vy * t_end,
                       x2 + vx * t_end, y2 + vy * t_end)
                if box[1] < cy_hi and box[3] > cy_lo:
                    log_in_row = True
                    if self._intersects(char_box, box):
                        on_log = True
        return log_in_row and not on_log

    @staticmethod
    def _intersects(a, b) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


class CharStateMachine:
    """Two-state machine driven directly off `_active_anims`. STATIONARY
    when no hop is in flight (planner is allowed to issue a move);
    MOVING while any animation is active (planner is muted)."""

    STATIONARY = "stationary"
    MOVING     = "moving"

    def __init__(self, anims: list):
        self._anims = anims
        self.state  = self.STATIONARY

    def is_stationary(self) -> bool:
        self.state = self.STATIONARY if not self._anims else self.MOVING
        return self.state == self.STATIONARY

    def reset(self) -> None:
        self.state = self.STATIONARY


def main():
    global _run_logger
    _run_logger = RunLogger()
    print(f"Logging to {_run_logger.path}")

    emu_proc = start_avd()
    launch_app()

    model      = YOLO(MODEL_PATH)
    stream     = FrameStream()
    vtrack     = VelocityTracker()
    gridscroll = GridScrollEstimator()
    hxscroll   = HorizontalScrollEstimator()
    char_fx    = _OneEuro1D(mincutoff=1.0, beta=0.05)
    char_fy    = _OneEuro1D(mincutoff=1.0, beta=0.05)
    planner    = Planner()
    state_machine = CharStateMachine(_active_anims)
    gameover   = GameOverDetector()
    cv2.namedWindow("Detections")
    cv2.setMouseCallback("Detections", _on_mouse)
    def _on_lane_h(v):
        gridscroll.set_lane_h(v)
        hxscroll.set_lane_h(v)

    def _on_sobel_th(v):
        gridscroll.set_sobel_thresh(v)
        hxscroll.set_sobel_thresh(v)

    cv2.createTrackbar("lane_h",   "Detections", gridscroll.lane_h, 40, _on_lane_h)
    cv2.setTrackbarMin("lane_h",   "Detections", 5)
    cv2.createTrackbar("sobel_th", "Detections", gridscroll.sobel_thresh, 100, _on_sobel_th)
    cv2.setTrackbarMin("sobel_th", "Detections", 1)
    cv2.createTrackbar("grid_shift", "Detections", 0, 20, lambda v: None)
    cv2.createTrackbar("shift_x",    "Detections", 91, 200, lambda v: None)
    input("Press Enter to start detection… ")
    print("Streaming — press q to quit.")

    try:
        frame_count  = 0
        settled_x    = 0.0   # cumulative px from completed animations
        settled_y    = 0.0
        show_grid    = False
        show_sobel   = False
        show_sobx    = False
        show_hx      = False
        show_plan    = True
        minimal_mode = True   # h: hide all overlays except A* plan
        prev_cum_y   = 0.0
        red_offset_x: float                          = 0.0
        red_offset_y: float                          = 0.0
        prev_green_xy: Optional[tuple[float, float]] = None
        prev_raw_green_xy: Optional[tuple[int, int]] = None
        green_glitch_count: int                      = 0
        pause_until: Optional[float]                 = None

        def reset_run():
            nonlocal settled_x, settled_y, prev_cum_y
            nonlocal red_offset_x, red_offset_y
            nonlocal prev_green_xy, prev_raw_green_xy, green_glitch_count
            settled_x = 0.0
            settled_y = 0.0
            _active_anims.clear()
            hxscroll.cum_scroll_x = 0.0
            hxscroll.flush()
            gridscroll.cum_scroll_y = 0.0
            gridscroll.prev_offset = None
            prev_cum_y = 0.0
            char_fx.reset()
            char_fy.reset()
            red_offset_x = 0.0
            red_offset_y = 0.0
            prev_green_xy = None
            prev_raw_green_xy = None
            green_glitch_count = 0
            state_machine.reset()
            gameover.reset()

        while True:
            shift_x = cv2.getTrackbarPos("shift_x", "Detections")
            frame   = transform_frame(stream.next_frame(), shift_x=shift_x)
            results = model.track(
                frame,
                imgsz=480, device="mps", conf=YOLO_CONF,
                persist=True,
                tracker="botsort.yaml",
                verbose=False,
            )

            annotated = frame.copy() if minimal_mode else results[0].plot()

            # Mirror ultralytics Annotator font metrics so text aligns with labels
            lw  = max(round(sum(annotated.shape) / 2 * 0.003), 2)
            sf  = lw / 3        # YOLO label font scale
            tf  = max(lw - 1, 1)
            vsf = sf * 0.7      # velocity text at 0.7x
            vtf = max(tf - 1, 1)

            boxes = results[0].boxes
            xyxys: list = []
            cls_list: list = []
            id_list: list = []
            ts = time.time()
            if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy) > 0:
                xyxys = boxes.xyxy.cpu().tolist()
                cls_list = (boxes.cls.int().cpu().tolist()
                            if boxes.cls is not None else [-1] * len(xyxys))
                id_list  = (boxes.id.int().cpu().tolist()
                            if boxes.id is not None else [None] * len(xyxys))

            # Retire finished animations into the settled offset.
            now = time.time()
            done = [(t0, dx, dy) for t0, dx, dy in _active_anims
                    if now - t0 >= ANIM_DURATION]
            for t0, dx, dy in done:
                settled_x += dx
                settled_y += dy
            _active_anims[:] = [(t0, dx, dy) for t0, dx, dy in _active_anims
                                 if now - t0 < ANIM_DURATION]

            # Character position: world-scroll (y) + button-press displacement.
            btn_x = settled_x + sum(_anim_progress(now - t0) * dx
                                    for t0, dx, dy in _active_anims)
            btn_y = settled_y + sum(_anim_progress(now - t0) * dy
                                    for t0, dx, dy in _active_anims)
            cx0, cy0 = CHAR_INIT_XY
            # Use button-driven position as reference so nearest-object selection
            # doesn't depend on the world-scroll value we're trying to estimate.
            ref_x = cx0 + btn_x
            ref_y = cy0 + btn_y
            world_scroll_y = gridscroll.update(frame, xyxys)

            # Horizontal scroll from inter-frame vertical-edge cross-correlation.
            if abs(gridscroll.cum_scroll_y - prev_cum_y) >= _GRID_JUMP_CLAMP:
                hxscroll.flush()
            prev_cum_y = gridscroll.cum_scroll_y
            hxscroll.update(frame, xyxys)
            hx_now = hxscroll.cum_scroll_x

            # Per-track velocity (world-frame, camera-corrected) + planner list.
            objs_for_planner: list = []
            for tid, cls, (x1, y1, x2, y2) in zip(id_list, cls_list, xyxys):
                v: float = 0.0
                if tid is not None:
                    vtrack.update(tid, x1, y1, x2, y2, ts, frame_count, cam_x=hx_now)
                    vv = vtrack.velocity(tid, cam_x_now=hx_now)
                    if vv is not None:
                        v = vv
                        if not minimal_mode:
                            txt = f"v={v:+.0f}"
                            (tw, th), _ = cv2.getTextSize(txt, 0, vsf, vtf)
                            tx = int((x1 + x2) / 2) - tw // 2
                            ty = int((y1 + y2) / 2) + th // 2
                            cv2.putText(annotated, txt,
                                        (tx, ty), 0, vsf,
                                        (0, 255, 255), vtf, cv2.LINE_AA)
                objs_for_planner.append({
                    "bbox": (x1, y1, x2, y2),
                    "vx":   v,
                    "vy":   0.0,
                    "cls":  cls,
                })

            # Morphology detection + One Euro smoothing — green dot is the
            # source of truth when visible. Raw observations are gated on
            # frame-to-frame jump so colour-match glitches don't poison the
            # filter; force-accept after _GREEN_GLITCH_LIMIT consecutive
            # rejects so a legitimate respawn unlocks the gate.
            char_xy = detect_character(frame)
            if char_xy is not None and prev_raw_green_xy is not None:
                jump = math.hypot(char_xy[0] - prev_raw_green_xy[0],
                                  char_xy[1] - prev_raw_green_xy[1])
                if jump > _GREEN_JUMP_THRESH and green_glitch_count < _GREEN_GLITCH_LIMIT:
                    green_glitch_count += 1
                    char_xy = None
            green_xy: Optional[tuple[float, float]] = None
            if char_xy is not None:
                prev_raw_green_xy = char_xy
                green_glitch_count = 0
                t_now = time.time()
                cx_s = char_fx(float(char_xy[0]), t_now)
                cy_s = char_fy(float(char_xy[1]), t_now)
                green_xy = (cx_s, cy_s)

            # Free-running red prediction (animation + hxscroll), pre-rebase.
            red_pred_x = ref_x - hxscroll.cum_scroll_x
            red_pred_y = ref_y + world_scroll_y

            # On green→absent transition, snap the red dot's anchor to the
            # last known green so the red continues from there.
            if green_xy is None and prev_green_xy is not None:
                red_offset_x = prev_green_xy[0] - red_pred_x
                red_offset_y = prev_green_xy[1] - red_pred_y
            prev_green_xy = green_xy

            char_draw_x = int(round(red_pred_x + red_offset_x))
            char_draw_y = int(round(red_pred_y + red_offset_y))
            if not minimal_mode:
                cv2.line(annotated, (char_draw_x, 0), (char_draw_x, H),
                         (0, 0, 255), 1)
                cv2.circle(annotated, (char_draw_x, char_draw_y),
                           4, (0, 0, 255), -1)

            if green_xy is not None:
                if not minimal_mode:
                    cv2.circle(annotated,
                               (int(round(green_xy[0])), int(round(green_xy[1]))),
                               4, (0, 255, 0), -1)
                truth_xy = green_xy
            else:
                truth_xy = (red_pred_x + red_offset_x,
                            red_pred_y + red_offset_y)
            tx, ty = int(round(truth_xy[0])), int(round(truth_xy[1]))
            if not minimal_mode:
                cv2.rectangle(annotated, (tx - 11, ty - 11),
                              (tx + 10, ty + 10), (0, 255, 255), 1)

            _run_logger.log_frame(
                green_xy=green_xy,
                red_xy=(red_pred_x + red_offset_x,
                        red_pred_y + red_offset_y),
                truth_xy=truth_xy,
                objects=objs_for_planner,
            )

            # TEMP: game-over flow disabled
            # go_triggered, _ = gameover.update(frame)
            # if go_triggered and pause_until is None:
            #     print("GAME OVER")
            #     pause_until = time.time() + 3.0

            if pause_until is not None:
                remaining = pause_until - time.time()
                if not minimal_mode:
                    cv2.putText(annotated,
                                f"GAME OVER — restarting in {max(0.0, remaining):.1f}s",
                                (5, 30), 0, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                if remaining <= 0.0:
                    reset_run()
                    send_game_over_recovery()
                    pause_until = None
            elif state_machine.is_stationary():
                move = planner.plan(truth_xy, objs_for_planner)
                if move is not None:
                    send_direction(move)

            if show_plan and getattr(planner, "last_origin", None) is not None:
                ox, oy = planner.last_origin
                ad     = ANIM_DIST
                ph     = planner.char_size // 2
                # Explored states (dim blue dots)
                for nx, ny, _depth in planner.last_explored:
                    cv2.circle(annotated,
                               (int(round(ox + nx * ad)),
                                int(round(oy + ny * ad))),
                               2, (200, 100, 0), -1)
                # Chosen path (cyan if goal reached, dim cyan if partial)
                path = planner.last_path
                path_color = (0, 255, 255) if planner.last_goal_reached else (0, 180, 180)
                prev_pt = None
                for i, (nx, ny) in enumerate(path):
                    cx = int(round(ox + nx * ad))
                    cy = int(round(oy + ny * ad))
                    thick = 2 if i == 1 else 1   # bold the immediate next step
                    cv2.rectangle(annotated,
                                  (cx - ph, cy - ph),
                                  (cx + ph, cy + ph),
                                  path_color, thick)
                    if prev_pt is not None:
                        cv2.line(annotated, prev_pt, (cx, cy), path_color, 1)
                    prev_pt = (cx, cy)
                # First-move tag
                fm = planner.last_first_move or "wait"
                tag = f"plan: {fm}{' ✓' if planner.last_goal_reached else ' …'}"
                cv2.putText(annotated, tag, (5, 50),
                            0, 0.4, path_color, 1, cv2.LINE_AA)

            if not minimal_mode and show_grid and gridscroll.prev_offset is not None:
                lh = gridscroll.lane_h
                m  = _GRID_LANE_MARGIN
                shift = cv2.getTrackbarPos("grid_shift", "Detections")
                y0_grid = (int(round(gridscroll.prev_offset)) + shift) % lh
                for y in range(y0_grid, H, lh):
                    cv2.line(annotated, (0, y), (W, y), (0, 255, 0), 1)
                    yb = y + (lh - m) - 1
                    if yb < H and yb != y:
                        cv2.line(annotated, (0, yb), (W, yb), (0, 200, 0), 1)
                cv2.putText(annotated,
                            f"scroll={gridscroll.cum_scroll_y:+.2f} o={gridscroll.prev_offset:.2f} h={lh} s={shift}",
                            (5, 15), 0, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

            if not minimal_mode and show_hx:
                hud = (f"hx: shift={hxscroll.last_shift:+5.2f} "
                       f"cum={hxscroll.cum_scroll_x:+6.1f} "
                       f"btn={btn_x:+6.1f} "
                       f"d={hxscroll.cum_scroll_x - btn_x:+5.1f} "
                       f"r={hxscroll.last_response:.2f} dy={hxscroll.last_dy:+4.2f}")
                cv2.putText(annotated, hud,
                            (5, H - 8), 0, 0.35, (0, 220, 255), 1, cv2.LINE_AA)

            del results  # release MPS tensors immediately

            frame_count += 1
            if frame_count % 100 == 0:
                torch.mps.empty_cache()
            if frame_count % 30 == 0:
                vtrack.prune(frame_count)

            if not minimal_mode:
                _draw_buttons(annotated)
            cv2.imshow("Detections", annotated)

            if not minimal_mode and show_sobel and gridscroll.last_binary is not None:
                vis = (gridscroll.last_binary * 255).astype(np.uint8)
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                if gridscroll.prev_offset is not None:
                    lh = gridscroll.lane_h
                    m  = _GRID_LANE_MARGIN
                    shift = cv2.getTrackbarPos("grid_shift", "Detections")
                    y0_grid = (int(round(gridscroll.prev_offset)) + shift) % lh
                    vh, vw = vis.shape[:2]
                    for y in range(y0_grid, vh, lh):
                        cv2.line(vis, (0, y), (vw, y), (0, 255, 0), 1)
                        yb = y + (lh - m) - 1
                        if yb < vh and yb != y:
                            cv2.line(vis, (0, yb), (vw, yb), (0, 200, 0), 1)
                cv2.imshow("Sobel", vis)

            if not minimal_mode and show_sobx and hxscroll.last_binary is not None:
                vx = cv2.cvtColor(hxscroll.last_binary, cv2.COLOR_GRAY2BGR)
                cv2.imshow("SobelX", vx)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("g"):
                show_grid = not show_grid
            elif key == ord("s"):
                show_sobel = not show_sobel
                if not show_sobel:
                    cv2.destroyWindow("Sobel")
            elif key == ord("x"):
                show_sobx = not show_sobx
                if not show_sobx:
                    cv2.destroyWindow("SobelX")
            elif key == ord("h"):
                minimal_mode = not minimal_mode
            elif key == ord("p"):
                show_plan = not show_plan
            elif key == ord("r"):
                reset_run()
                pause_until = None
    finally:
        stream.stop()
        cv2.destroyAllWindows()
        emu_proc.terminate()
        if _run_logger is not None:
            _run_logger.close()


if __name__ == "__main__":
    main()

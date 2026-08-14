import json
import os
import subprocess
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Union

import math

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PRUNE_AFTER_FRAMES = 30   # match BoT-SORT's track_buffer; once it drops a track, we can too
EDGE_EPS           = 4    # px tolerance for "bbox edge touches frame border"
KALMAN_MISSING_FRAMES = 30
KALMAN_PROCESS_VAR = 140.0
KALMAN_MEAS_VAR = 64.0


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

    def reset(self) -> None:
        self._tracks.clear()

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


@dataclass
class _KalmanTrack:
    state: np.ndarray
    cov: np.ndarray
    cls: int
    w: float
    h: float
    last_ts: float
    last_seen_frame: int
    last_update_frame: int


@dataclass
class TrackedObject:
    tid: Optional[int]
    cls: int
    bbox: tuple[float, float, float, float]
    vx: float
    vy: float
    predicted: bool


class MovingObjectKalmanTracker:
    """Constant-velocity Kalman tracker over bbox center.

    State is [cx, cy, vx, vy]. Bbox width/height are side data used only
    for drawing/masking. Tracks keep predicting for a short window after
    detections disappear, then update when the same track id reappears.
    """

    def __init__(self, max_missing_frames: int = KALMAN_MISSING_FRAMES,
                 process_var: float = KALMAN_PROCESS_VAR,
                 meas_var: float = KALMAN_MEAS_VAR):
        self.max_missing_frames = max(1, int(max_missing_frames))
        self.process_var = float(process_var)
        self.meas_var = float(meas_var)
        self._tracks: dict[int, _KalmanTrack] = {}

    @staticmethod
    def _measurement(bbox: Union[list, tuple]) -> np.ndarray:
        x1, y1, x2, y2 = map(float, bbox)
        return np.array([
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        ], dtype=np.float64)

    @staticmethod
    def _bbox_size(bbox: Union[list, tuple]) -> tuple[float, float]:
        x1, y1, x2, y2 = map(float, bbox)
        return max(1.0, x2 - x1), max(1.0, y2 - y1)

    @staticmethod
    def _bbox_from_track(track: _KalmanTrack) -> tuple[float, float, float, float]:
        cx, cy = track.state[:2]
        w = max(1.0, float(track.w))
        h = max(1.0, float(track.h))
        x1 = float(np.clip(cx - w / 2.0, 0, W))
        y1 = float(np.clip(cy - h / 2.0, 0, H))
        x2 = float(np.clip(cx + w / 2.0, 0, W))
        y2 = float(np.clip(cy + h / 2.0, 0, H))
        return x1, y1, x2, y2

    def _measurement_for_update(self, track: _KalmanTrack,
                                bbox: Union[list, tuple]) -> np.ndarray:
        x1, y1, x2, y2 = map(float, bbox)
        clipped_left = x1 <= EDGE_EPS
        clipped_right = x2 >= W - EDGE_EPS
        if not clipped_left and not clipped_right:
            return self._measurement((x1, y1, x2, y2))

        track_w = max(1.0, float(track.w))
        cy = (y1 + y2) / 2.0
        if clipped_left and not clipped_right:
            cx = x2 - track_w / 2.0
        elif clipped_right and not clipped_left:
            cx = x1 + track_w / 2.0
        else:
            cx = float(track.state[0])
        return np.array([cx, cy], dtype=np.float64)

    def reset(self) -> None:
        self._tracks.clear()

    def _new_track(self, bbox: Union[list, tuple], cls: int,
                   ts: float, frame_idx: int) -> _KalmanTrack:
        z = self._measurement(bbox)
        w, h = self._bbox_size(bbox)
        state = np.zeros(4, dtype=np.float64)
        state[:2] = z
        cov = np.diag([25.0, 25.0, 400.0, 400.0]).astype(np.float64)
        return _KalmanTrack(state, cov, int(cls), w, h, ts, frame_idx, frame_idx)

    def _predict_one(self, track: _KalmanTrack, ts: float) -> None:
        dt = max(min(ts - track.last_ts, 0.25), 1.0 / MAX_FPS)
        F = np.eye(4, dtype=np.float64)
        F[0, 2] = dt
        F[1, 3] = dt
        q = self.process_var
        Q = np.diag([
            q * dt ** 4, q * dt ** 4,
            q * dt ** 2, q * dt ** 2,
        ]).astype(np.float64)
        track.state = F @ track.state
        track.cov = F @ track.cov @ F.T + Q
        track.last_ts = ts

    def _update_one(self, track: _KalmanTrack, bbox: Union[list, tuple],
                    cls: int, frame_idx: int) -> None:
        x1, _y1, x2, _y2 = map(float, bbox)
        if x1 > EDGE_EPS and x2 < W - EDGE_EPS:
            track.w, track.h = self._bbox_size(bbox)
        else:
            _w, h = self._bbox_size(bbox)
            track.h = h

        z = self._measurement_for_update(track, bbox)
        Hm = np.zeros((2, 4), dtype=np.float64)
        Hm[0, 0] = Hm[1, 1] = 1.0
        R = np.eye(2, dtype=np.float64) * self.meas_var
        y = z - Hm @ track.state
        S = Hm @ track.cov @ Hm.T + R
        K = track.cov @ Hm.T @ np.linalg.inv(S)
        track.state = track.state + K @ y
        track.cov = (np.eye(4, dtype=np.float64) - K @ Hm) @ track.cov
        track.cls = int(cls)
        track.last_seen_frame = frame_idx
        track.last_update_frame = frame_idx

    def update(self, bboxes: list, class_ids: list, track_ids: list,
               ts: float, frame_idx: int) -> list[TrackedObject]:
        for track in self._tracks.values():
            self._predict_one(track, ts)

        updated: set[int] = set()
        for det_idx, (bbox, cls, tid) in enumerate(zip(bboxes, class_ids, track_ids)):
            if tid is None:
                continue
            tid = int(tid)
            track = self._tracks.get(tid)
            if track is None:
                track = self._new_track(bbox, cls, ts, frame_idx)
                self._tracks[tid] = track
            else:
                self._update_one(track, bbox, cls, frame_idx)
            updated.add(tid)

        stale = [tid for tid, track in self._tracks.items()
                 if frame_idx - track.last_seen_frame > self.max_missing_frames]
        for tid in stale:
            del self._tracks[tid]

        out: list[TrackedObject] = []
        for tid, track in self._tracks.items():
            if frame_idx - track.last_seen_frame > self.max_missing_frames:
                continue
            x1, y1, x2, y2 = self._bbox_from_track(track)
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(TrackedObject(
                tid=tid,
                cls=track.cls,
                bbox=(x1, y1, x2, y2),
                vx=float(track.state[2]),
                vy=float(track.state[3]),
                predicted=tid not in updated,
            ))

        # Keep untracked detections usable, but they cannot be predicted
        # through misses without a stable id.
        for det_idx, (bbox, cls, tid) in enumerate(zip(bboxes, class_ids, track_ids)):
            if tid is not None:
                continue
            x1, y1, x2, y2 = map(float, bbox)
            out.append(TrackedObject(
                tid=None,
                cls=int(cls),
                bbox=(x1, y1, x2, y2),
                vx=0.0,
                vy=0.0,
                predicted=False,
            ))
        return out


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


GRID_PERIOD     = 44       # px per lane (one hop = one period)
_GRID_LANE_MARGIN_DEFAULT = 2      # px gap rendered between adjacent lanes
_GRID_START_OFFSET = 18
_GRID_X_RANGE   = (40, 360)
_GRID_Y_RANGE   = (60, 840)
_GRID_JUMP_CLAMP = 30      # px; |delta| above this is treated as a scene cut
_LANE_OFFSET_CYCLE_DEFAULT = 44.0
_LANE_OFFSET_DROP_WINDOW = 8
_LANE_OFFSET_DROP_FRAC = 0.5
_LANE_OFFSET_REARM_FRAC = 0.5


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
        self.start_offset:  int               = _GRID_START_OFFSET % self.lane_h
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
        self.start_offset %= self.lane_h
        self.prev_offset = None      # previous phase reference is in old units

    def set_start_offset(self, offset: int) -> None:
        offset = int(offset) % self.lane_h
        if offset == self.start_offset:
            return
        self.start_offset = offset
        self.prev_offset = None      # avoid an unwrap jump when tuning manually

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
        o_star_f = float((o_star_f + self.start_offset) % P)

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
W, H       = 480, 960
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
CHAR_INIT_XY    = (446, 630)   # initial centroid of the chicken on screen

# Morphology-based character detector (ported from crossybot_v2.0
# CharacterTools.detect_character). BGR-threshold the chicken's beak/body
# colour, branch on raw mask population for the right open/close/dilate
# recipe, pick the largest valid CC. Centroid is shifted down to land on
# the character's feet rather than the colored beak.
_CHAR_BGR_TARGET   = (92, 172, 255)
_CHAR_TOL          = 4
_GREEN_JUMP_THRESH = 60.0  # px/frame; reject morphology jumps above this as glitches
_GREEN_GLITCH_LIMIT = 30   # frames; force-accept after this many consecutive rejects
_CHAR_MIN_AREA_FRAC = 0.0005
_CHAR_MAX_AREA_FRAC = 0.05
_CHAR_OPEN_KSZ     = 6
_CHAR_CLOSE_KSZ    = 10
_CHAR_Y_SHIFT      = 22


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
SEND_VM_KEYSTROKES = False
ANIM_DURATION = 0.3   # seconds per button-press animation
ANIM_DIST     = 44    # px of character displacement per press (= one lane)
_ANIM_K       = 10.0  # decay rate; 1/k ≈ 0.10 s = time of peak velocity
_LANE_COLOR_MIN_PIXELS = 1
_LANE_ASSIGNMENT_MODE_FRAMES = 7
LOG_CLASS_ID = 2
LOG_LANE_Y_OFFSET_DEFAULT = 66

# D-pad button regions (x1, y1, x2, y2) in frame coordinates.
# Placed in the bottom-right corner of the 480×960 frame.
_BTNS: dict[str, tuple[int, int, int, int]] = {
    "up":    (362, 842, 414, 894),
    "left":  (304, 900, 356, 952),
    "down":  (362, 900, 414, 952),
    "right": (420, 900, 472, 952),
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
    if not SEND_VM_KEYSTROKES:
        return
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
    if not SEND_VM_KEYSTROKES:
        return
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


class LaneType(Enum):
    ROAD = "road"
    WATER = "water"
    GRASS = "grass"


@dataclass
class LaneClassification:
    idx: int
    y0: int
    y1: int
    lane_type: LaneType
    gray_px: int
    blue_px: int
    ignored_spans: list[tuple[int, int]]
    assigned_objects: list[dict]


def _lane_bands(frame_h: int, lane_h: int, offset: float,
                lane_margin: int = _GRID_LANE_MARGIN_DEFAULT) -> list[tuple[int, int]]:
    lane_h = max(2, int(lane_h))
    lane_margin = int(np.clip(lane_margin, 0, lane_h - 1))
    y = int(round(offset)) % lane_h
    while y > 0:
        y -= lane_h
    bands: list[tuple[int, int]] = []
    while y < frame_h:
        y0 = max(0, y)
        y1 = min(frame_h, y + lane_h - lane_margin)
        if y1 > y0:
            bands.append((y0, y1))
        y += lane_h
    return bands


class LaneAssignmentSmoother:
    def __init__(self, window_frames: int = _LANE_ASSIGNMENT_MODE_FRAMES):
        self.window_frames = max(1, int(window_frames))
        self._history: dict = {}
        self._last_seen: dict = {}

    def reset(self) -> None:
        self._history.clear()
        self._last_seen.clear()

    @staticmethod
    def _raw_lane_index(bands: list[tuple[int, int]], bbox: list,
                        y_offset: float = 0.0) -> Optional[int]:
        _x1, y1, _x2, y2 = bbox
        cy = (float(y1) + float(y2)) / 2.0 + float(y_offset)
        for i, (lane_y0, lane_y1) in enumerate(bands):
            if lane_y0 <= cy < lane_y1:
                return i
        return None

    def lane_index(self, key, bands: list[tuple[int, int]], bbox: list,
                   frame_idx: int, y_offset: float = 0.0) -> Optional[int]:
        raw = self._raw_lane_index(bands, bbox, y_offset)
        if raw is None:
            return None

        hist = self._history.setdefault(key, deque(maxlen=self.window_frames))
        hist.append(raw)
        self._last_seen[key] = frame_idx

        counts = Counter(hist)
        best_count = max(counts.values())
        candidates = {lane for lane, count in counts.items() if count == best_count}
        for lane in reversed(hist):
            if lane in candidates:
                return lane
        return raw

    def prune(self, frame_idx: int) -> None:
        stale_after = self.window_frames * 3
        stale = [key for key, seen in self._last_seen.items()
                 if frame_idx - seen > stale_after]
        for key in stale:
            self._history.pop(key, None)
            self._last_seen.pop(key, None)


def _lane_color_masks(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2Lab)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lab_a = lab[..., 1].astype(np.float32)
    lab_b = lab[..., 2].astype(np.float32)

    gray_hsv = ((h >= 95) & (h <= 135) & (s <= 80) &
                (v >= 80) & (v <= 130))
    da = np.abs(lab_a - 128.0)
    db = np.abs(lab_b - 128.0)
    chroma = np.sqrt((lab_a - 128.0) ** 2 + (lab_b - 128.0) ** 2)
    gray_lab = (da <= 10.0) & (db <= 12.0) & (chroma <= 16.0)
    gray = (gray_hsv & gray_lab).astype(np.uint8) * 255

    blue_hsv = ((h >= 95) & (h <= 110) & (s >= 100) & (v >= 200))
    blue_lab = lab[..., 2] <= 125
    blue = (blue_hsv & blue_lab).astype(np.uint8) * 255

    k = np.ones((3, 3), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, k, iterations=1)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k, iterations=1)
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, k, iterations=1)
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, k, iterations=1)
    return gray, blue


def classify_lanes(frame_bgr: np.ndarray, lane_h: int,
                   offset: Optional[float],
                   bboxes: Optional[list] = None,
                   track_ids: Optional[list] = None,
                   class_ids: Optional[list] = None,
                   lane_smoother: Optional[LaneAssignmentSmoother] = None,
                   frame_idx: int = 0,
                   log_y_offset: int = 0,
                   lane_idx_base: int = 0,
                   lane_margin: int = _GRID_LANE_MARGIN_DEFAULT) -> list[LaneClassification]:
    if offset is None:
        return []
    gray, blue = _lane_color_masks(frame_bgr)
    x0, x1 = _GRID_X_RANGE
    bands = _lane_bands(frame_bgr.shape[0], lane_h, offset, lane_margin)
    ignored_by_band: list[list[tuple[int, int]]] = [[] for _ in bands]
    assigned_by_band: list[list[dict]] = [[] for _ in bands]
    if bboxes:
        track_ids = track_ids or [None] * len(bboxes)
        class_ids = class_ids or [None] * len(bboxes)
        for det_idx, (bbox, tid, cls) in enumerate(zip(bboxes, track_ids, class_ids)):
            lane_idx = None
            y_offset = log_y_offset if cls == LOG_CLASS_ID else 0
            if lane_smoother is not None:
                key = ("track", int(tid)) if tid is not None else ("det", det_idx)
                lane_idx = lane_smoother.lane_index(
                    key, bands, bbox, frame_idx, y_offset
                )
            else:
                lane_idx = LaneAssignmentSmoother._raw_lane_index(
                    bands, bbox, y_offset
                )
            if lane_idx is None or lane_idx >= len(ignored_by_band):
                continue

            bx1, _by1, bx2, _by2 = bbox
            ix0 = max(0, min(frame_bgr.shape[1], int(math.floor(bx1))))
            ix1 = max(0, min(frame_bgr.shape[1], int(math.ceil(bx2))))
            if ix1 > ix0:
                ignored_by_band[lane_idx].append((ix0, ix1))
                assigned_by_band[lane_idx].append({
                    "x0": float(ix0),
                    "x1": float(ix1),
                    "band_idx": int(lane_idx),
                    "cls": int(cls) if cls is not None else -1,
                    "id": int(tid) if tid is not None else None,
                    "det_idx": det_idx,
                })

    out: list[LaneClassification] = []
    visible_band_count = sum(
        1 for y0, y1 in bands
        if min(y1, _GRID_Y_RANGE[1]) > max(y0, _GRID_Y_RANGE[0])
    )
    visible_idx = 0
    for i, (y0, y1) in enumerate(bands):
        ys = max(y0, _GRID_Y_RANGE[0])
        ye = min(y1, _GRID_Y_RANGE[1])
        if ye <= ys:
            continue
        display_idx = lane_idx_base + visible_band_count - visible_idx - 1
        visible_idx += 1
        gray_roi = gray[ys:ye, x0:x1].copy()
        blue_roi = blue[ys:ye, x0:x1].copy()
        for ix0, ix1 in ignored_by_band[i]:
            rx0 = max(0, ix0 - x0)
            rx1 = min(x1 - x0, ix1 - x0)
            if rx1 > rx0:
                gray_roi[:, rx0:rx1] = 0
                blue_roi[:, rx0:rx1] = 0
        gray_px = int(cv2.countNonZero(gray_roi))
        blue_px = int(cv2.countNonZero(blue_roi))
        if gray_px >= _LANE_COLOR_MIN_PIXELS:
            lane_type = LaneType.ROAD
        elif blue_px >= _LANE_COLOR_MIN_PIXELS:
            lane_type = LaneType.WATER
        else:
            lane_type = LaneType.GRASS
        out.append(LaneClassification(
            display_idx, y0, y1, lane_type, gray_px, blue_px,
            ignored_by_band[i], assigned_by_band[i]
        ))
    return out


def draw_lane_classifications(frame: np.ndarray,
                              lanes: list[LaneClassification]) -> None:
    colors = {
        LaneType.ROAD:  (120, 120, 120),
        LaneType.WATER: (255, 120, 0),
        LaneType.GRASS: (60, 180, 60),
    }
    overlay = frame.copy()
    for lane in lanes:
        color = colors[lane.lane_type]
        cv2.rectangle(overlay, (0, lane.y0), (W, lane.y1), color, -1)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0.0, dst=frame)

    ignored_overlay = frame.copy()
    for lane in lanes:
        for x0, x1 in lane.ignored_spans:
            cv2.rectangle(ignored_overlay, (x0, lane.y0), (x1, lane.y1),
                          (95, 95, 95), -1)
    cv2.addWeighted(ignored_overlay, 0.55, frame, 0.45, 0.0, dst=frame)

    for lane in lanes:
        color = colors[lane.lane_type]
        for x0, x1 in lane.ignored_spans:
            cv2.rectangle(frame, (x0, lane.y0), (x1, lane.y1),
                          (190, 190, 190), 1)
        cy = (lane.y0 + lane.y1) // 2
        text = f"{lane.idx}: {lane.lane_type.value} g={lane.gray_px} b={lane.blue_px}"
        cv2.putText(frame, text, (5, max(12, cy + 4)),
                    0, 0.4, color, 1, cv2.LINE_AA)


def draw_lane_assigned_boxes(frame: np.ndarray,
                             lanes: list[LaneClassification]) -> None:
    for lane in lanes:
        for x0, x1 in lane.ignored_spans:
            cv2.rectangle(frame, (x0, lane.y0), (x1, lane.y1),
                          (190, 190, 190), 1)


PLANNER_DEPTH = 4
PLANNER_ACTION_DT = ANIM_DURATION
PLANNER_PLAYER_HALF_W_FRAC = 0.36
PLANNER_EDGE_MARGIN = 20.0
PLANNER_MIN_ACTION_INTERVAL = ANIM_DURATION * 0.85
PLANNER_SEND_KEYS = False
_PLANNER_ACTIONS = ("up", "left", "right", "down", "stay")
_PLANNER_DELTAS = {
    "up": (0.0, -1.0),
    "down": (0.0, 1.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "stay": (0.0, 0.0),
}


@dataclass
class PlannerObject:
    lane_idx: int
    x0: float
    x1: float
    vx: float
    cls: int


@dataclass
class PlannerDecision:
    action: str
    score: float
    state: Optional[list[int]]
    safe_actions: dict[str, bool]


class GameTreePlanner:
    """Small CV-backed game-tree planner inspired by Crossy-Road-AI's minimax.

    The Unity version simulates exact colliders. Here we mirror the idea with
    screen-space lane bands, assigned object spans, and measured velocities.
    """

    def __init__(self, depth: int = PLANNER_DEPTH,
                 action_dt: float = PLANNER_ACTION_DT):
        self.depth = max(1, int(depth))
        self.action_dt = float(action_dt)

    @staticmethod
    def _lane_for_y(lanes: list[LaneClassification],
                    y: float) -> Optional[LaneClassification]:
        if not lanes:
            return None
        for lane in lanes:
            if lane.y0 <= y < lane.y1:
                return lane
        return min(lanes, key=lambda lane: abs(((lane.y0 + lane.y1) / 2.0) - y))

    @staticmethod
    def _lane_by_idx(lanes: list[LaneClassification],
                     idx: int) -> Optional[LaneClassification]:
        for lane in lanes:
            if lane.idx == idx:
                return lane
        return None

    def _objects_for_lane(self, lane: LaneClassification,
                          objects: list[PlannerObject]) -> list[PlannerObject]:
        return [obj for obj in objects if obj.lane_idx == lane.idx]

    @staticmethod
    def _span_overlaps_tile(obj: PlannerObject, x: float, lane_h: float,
                            t: float) -> bool:
        half_w = lane_h * PLANNER_PLAYER_HALF_W_FRAC
        ox0 = obj.x0 + obj.vx * t
        ox1 = obj.x1 + obj.vx * t
        return x - half_w < ox1 and x + half_w > ox0

    def _tile_status(self, x: float, y: float, lane_h: float,
                     lanes: list[LaneClassification],
                     objects: list[PlannerObject], t: float
                     ) -> tuple[bool, LaneType, float]:
        if x < PLANNER_EDGE_MARGIN or x > W - PLANNER_EDGE_MARGIN:
            return False, LaneType.GRASS, 0.0

        lane = self._lane_for_y(lanes, y)
        if lane is None:
            return False, LaneType.GRASS, 0.0

        lane_objects = self._objects_for_lane(lane, objects)

        if lane.lane_type == LaneType.WATER:
            best_log_vx = 0.0
            for obj in lane_objects:
                if obj.cls == LOG_CLASS_ID and self._span_overlaps_tile(
                        obj, x, lane_h, t):
                    best_log_vx = obj.vx
                    return True, lane.lane_type, best_log_vx
            return False, lane.lane_type, 0.0

        if lane.lane_type == LaneType.ROAD:
            for obj in lane_objects:
                if obj.cls != LOG_CLASS_ID and self._span_overlaps_tile(
                        obj, x, lane_h, t):
                    return False, lane.lane_type, 0.0

        return True, lane.lane_type, 0.0

    def _state_vector(self, x: float, y: float, lane_h: float,
                      lanes: list[LaneClassification],
                      objects: list[PlannerObject]) -> Optional[list[int]]:
        curr_lane = self._lane_for_y(lanes, y)
        if curr_lane is None:
            return None

        state: list[int] = []
        checks = [
            (-1, -1), (0, -1), (1, -1),
            (-1, 0), (1, 0),
            (-1, 1), (0, 1), (1, 1),
        ]
        for dx, dy in checks:
            safe, _lane_type, _log_vx = self._tile_status(
                x + dx * lane_h, y + dy * lane_h, lane_h, lanes, objects,
                self.action_dt
            )
            state.append(0 if safe else 1)

        for rel in (-1, 0, 1):
            lane = self._lane_by_idx(lanes, curr_lane.idx - rel)
            lane_type = lane.lane_type if lane is not None else LaneType.GRASS
            state.append({
                LaneType.GRASS: 0,
                LaneType.ROAD: 1,
                LaneType.WATER: 2,
            }[lane_type])

        for rel in (-1, 0, 1):
            lane = self._lane_by_idx(lanes, curr_lane.idx - rel)
            if lane is None:
                state.append(0)
                continue
            lane_objs = self._objects_for_lane(lane, objects)
            if not lane_objs:
                state.append(0)
            else:
                mean_vx = sum(obj.vx for obj in lane_objs) / len(lane_objs)
                state.append(1 if mean_vx > 1.0 else -1 if mean_vx < -1.0 else 0)

        return state

    def _score_terminal(self, x: float, y: float, start_y: float,
                        lane_h: float, lane_type: LaneType) -> float:
        progress = (start_y - y) / lane_h
        center_penalty = abs(x - W / 2.0) / max(W / 2.0, 1.0)
        lane_bonus = {
            LaneType.GRASS: 5.0,
            LaneType.ROAD: 0.0,
            LaneType.WATER: -1.0,
        }[lane_type]
        return progress * 100.0 - center_penalty * 8.0 + lane_bonus

    def _recurse(self, x: float, y: float, depth: int, elapsed: float,
                 start_y: float, lane_h: float,
                 lanes: list[LaneClassification],
                 objects: list[PlannerObject]) -> float:
        safe, lane_type, log_vx = self._tile_status(
            x, y, lane_h, lanes, objects, elapsed
        )
        if not safe:
            return -100000.0 - depth
        if depth <= 0:
            return self._score_terminal(x, y, start_y, lane_h, lane_type)

        if lane_type == LaneType.WATER:
            x += log_vx * self.action_dt

        best = -1000000.0
        for action in _PLANNER_ACTIONS:
            dx, dy = _PLANNER_DELTAS[action]
            nx = x + dx * lane_h
            ny = y + dy * lane_h
            score = self._recurse(
                nx, ny, depth - 1, elapsed + self.action_dt,
                start_y, lane_h, lanes, objects
            )
            if action == "up":
                score += 6.0
            elif action == "down":
                score -= 12.0
            elif action == "stay":
                score -= 5.0
            if score > best:
                best = score
        return best

    def choose(self, truth_xy: tuple[float, float],
               lanes: list[LaneClassification],
               objects_for_frame: list[dict],
               lane_h: int) -> PlannerDecision:
        lane_h_f = float(max(2, lane_h))
        x, y = map(float, truth_xy)
        by_id = {
            int(obj["id"]): obj for obj in objects_for_frame
            if obj.get("id") is not None
        }
        by_det_idx = {
            i: obj for i, obj in enumerate(objects_for_frame)
            if obj.get("id") is None
        }
        objects: list[PlannerObject] = []
        for lane in lanes:
            for assigned in lane.assigned_objects:
                src = None
                if assigned.get("id") is not None:
                    src = by_id.get(int(assigned["id"]))
                if src is None:
                    src = by_det_idx.get(int(assigned.get("det_idx", -1)))
                objects.append(PlannerObject(
                    lane_idx=lane.idx,
                    x0=float(assigned["x0"]),
                    x1=float(assigned["x1"]),
                    vx=float(src.get("vx", 0.0)) if src is not None else 0.0,
                    cls=int(assigned.get("cls", -1)),
                ))

        state = self._state_vector(x, y, lane_h_f, lanes, objects)
        safe_actions: dict[str, bool] = {}
        best_action = "stay"
        best_score = -1000000.0
        for action in _PLANNER_ACTIONS:
            dx, dy = _PLANNER_DELTAS[action]
            nx = x + dx * lane_h_f
            ny = y + dy * lane_h_f
            safe, _lane_type, _log_vx = self._tile_status(
                nx, ny, lane_h_f, lanes, objects, self.action_dt
            )
            safe_actions[action] = safe
            score = self._recurse(
                nx, ny, self.depth - 1, self.action_dt,
                y, lane_h_f, lanes, objects
            )
            if action == "up":
                score += 6.0
            elif action == "down":
                score -= 12.0
            elif action == "stay":
                score -= 5.0
            if score > best_score:
                best_score = score
                best_action = action

        return PlannerDecision(best_action, best_score, state, safe_actions)


def draw_planner_decision(frame: np.ndarray,
                          decision: PlannerDecision,
                          auto_enabled: bool) -> None:
    status = "auto" if auto_enabled else "advice"
    safe = "".join(
        name[0].upper() if ok else name[0].lower()
        for name, ok in decision.safe_actions.items()
    )
    text = f"planner {status}: {decision.action} score={decision.score:.0f} safe={safe}"
    cv2.putText(frame, text, (5, 31), 0, 0.4, (0, 220, 255), 1, cv2.LINE_AA)


def main():
    global _run_logger
    _run_logger = RunLogger()
    print(f"Logging to {_run_logger.path}")

    emu_proc = start_avd()
    launch_app()

    model      = YOLO(MODEL_PATH)
    stream     = FrameStream()
    vtrack     = VelocityTracker()
    ktrack     = MovingObjectKalmanTracker()
    gridscroll = GridScrollEstimator()
    hxscroll   = HorizontalScrollEstimator()
    lane_smoother = LaneAssignmentSmoother()
    char_fx    = _OneEuro1D(mincutoff=1.0, beta=0.05)
    char_fy    = _OneEuro1D(mincutoff=1.0, beta=0.05)
    gameover   = GameOverDetector()
    planner    = GameTreePlanner()
    cv2.namedWindow("Detections")
    cv2.setMouseCallback("Detections", _on_mouse)
    def _on_lane_h(v):
        gridscroll.set_lane_h(v)
        hxscroll.set_lane_h(v)

    def _on_sobel_th(v):
        gridscroll.set_sobel_thresh(v)
        hxscroll.set_sobel_thresh(v)

    def _on_grid_start(v):
        gridscroll.set_start_offset(v)

    cv2.createTrackbar("lane_h",   "Detections", gridscroll.lane_h, 80, _on_lane_h)
    cv2.setTrackbarMin("lane_h",   "Detections", 10)
    cv2.createTrackbar("sobel_th", "Detections", gridscroll.sobel_thresh, 100, _on_sobel_th)
    cv2.setTrackbarMin("sobel_th", "Detections", 1)
    cv2.createTrackbar("grid_start", "Detections", gridscroll.start_offset, 80, _on_grid_start)
    cv2.createTrackbar("grid_shift", "Detections", 0, 40, lambda v: None)
    cv2.createTrackbar("grid_margin", "Detections",
                       _GRID_LANE_MARGIN_DEFAULT, 20, lambda v: None)
    cv2.createTrackbar("log_y_offset", "Detections",
                       LOG_LANE_Y_OFFSET_DEFAULT + 80, 160, lambda v: None)
    cv2.createTrackbar("cycle_x", "Detections",
                       int(round(_LANE_OFFSET_CYCLE_DEFAULT * 10)), 440, lambda v: None)
    cv2.setTrackbarMin("cycle_x", "Detections", 400)
    cv2.createTrackbar("shift_x",    "Detections", 182, 400, lambda v: None)
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
        show_boxes   = False
        show_detections = False
        minimal_mode = True   # h: hide all overlays
        planner_auto = PLANNER_SEND_KEYS
        last_planner_action_ts = 0.0
        prev_cum_y   = 0.0
        lane_idx_base = 0
        offset_cycle_history = deque(maxlen=_LANE_OFFSET_DROP_WINDOW)
        offset_cycle_armed = False
        prev_cycle_x = _LANE_OFFSET_CYCLE_DEFAULT
        red_offset_x: float                          = 0.0
        red_offset_y: float                          = 0.0
        prev_green_xy: Optional[tuple[float, float]] = None
        prev_raw_green_xy: Optional[tuple[int, int]] = None
        green_glitch_count: int                      = 0
        pause_until: Optional[float]                 = None

        def reset_run():
            nonlocal settled_x, settled_y, prev_cum_y
            nonlocal lane_idx_base, offset_cycle_armed, prev_cycle_x
            nonlocal red_offset_x, red_offset_y
            nonlocal prev_green_xy, prev_raw_green_xy, green_glitch_count
            nonlocal last_planner_action_ts
            settled_x = 0.0
            settled_y = 0.0
            _active_anims.clear()
            vtrack.reset()
            ktrack.reset()
            hxscroll.cum_scroll_x = 0.0
            hxscroll.flush()
            gridscroll.cum_scroll_y = 0.0
            gridscroll.prev_offset = None
            gridscroll.set_start_offset(cv2.getTrackbarPos("grid_start", "Detections"))
            lane_smoother.reset()
            prev_cum_y = 0.0
            lane_idx_base = 0
            prev_cycle_x = cv2.getTrackbarPos("cycle_x", "Detections") / 10.0
            offset_cycle_history.clear()
            offset_cycle_armed = False
            char_fx.reset()
            char_fy.reset()
            red_offset_x = 0.0
            red_offset_y = 0.0
            prev_green_xy = None
            prev_raw_green_xy = None
            green_glitch_count = 0
            last_planner_action_ts = 0.0
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

            annotated = frame.copy()

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
            kalman_objects = ktrack.update(xyxys, cls_list, id_list, ts, frame_count)
            missing_kalman_objects = [obj for obj in kalman_objects if obj.predicted]
            active_xyxys = list(xyxys) + [obj.bbox for obj in missing_kalman_objects]
            active_cls_list = list(cls_list) + [obj.cls for obj in missing_kalman_objects]
            active_id_list = list(id_list) + [obj.tid for obj in missing_kalman_objects]

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
            world_scroll_y = gridscroll.update(frame, active_xyxys)
            log_y_offset = cv2.getTrackbarPos("log_y_offset", "Detections") - 80
            grid_margin = min(cv2.getTrackbarPos("grid_margin", "Detections"),
                              gridscroll.lane_h - 1)
            cycle_x = cv2.getTrackbarPos("cycle_x", "Detections") / 10.0
            if abs(cycle_x - prev_cycle_x) > 1e-6:
                prev_cycle_x = cycle_x
                offset_cycle_history.clear()
                offset_cycle_armed = False
            if gridscroll.prev_offset is not None and cycle_x > 0:
                current_offset = gridscroll.prev_offset
                if current_offset > cycle_x * _LANE_OFFSET_REARM_FRAC:
                    offset_cycle_armed = True
                if (offset_cycle_armed and offset_cycle_history and
                        max(offset_cycle_history) - current_offset >
                        cycle_x * _LANE_OFFSET_DROP_FRAC):
                    lane_idx_base += 1
                    offset_cycle_armed = False
                    offset_cycle_history.clear()
                offset_cycle_history.append(current_offset)
            lane_classes = classify_lanes(frame, gridscroll.lane_h,
                                          gridscroll.prev_offset, active_xyxys,
                                          active_id_list, active_cls_list, lane_smoother,
                                          frame_count, log_y_offset, lane_idx_base,
                                          grid_margin)

            # Horizontal scroll from inter-frame vertical-edge cross-correlation.
            if abs(gridscroll.cum_scroll_y - prev_cum_y) >= _GRID_JUMP_CLAMP:
                hxscroll.flush()
            prev_cum_y = gridscroll.cum_scroll_y
            hxscroll.update(frame, active_xyxys)

            # Raw detections use the old velocity tracker. Kalman is kept in
            # parallel and only contributes objects while detections are missing.
            objs_for_frame: list = []
            for tid, cls, (x1, y1, x2, y2) in zip(id_list, cls_list, xyxys):
                v: float = 0.0
                if tid is not None:
                    vtrack.update(tid, x1, y1, x2, y2, ts, frame_count,
                                  cam_x=hxscroll.cum_scroll_x)
                    vv = vtrack.velocity(tid, cam_x_now=hxscroll.cum_scroll_x)
                    if vv is not None:
                        v = vv
                if not minimal_mode and show_boxes:
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 255, 255), lw)
                if not minimal_mode:
                    box_color = (0, 255, 255)
                    txt = f"v={v:+.0f}"
                    (tw, th), _ = cv2.getTextSize(txt, 0, vsf, vtf)
                    tx = int((x1 + x2) / 2) - tw // 2
                    ty = int((y1 + y2) / 2) + th // 2
                    cv2.putText(annotated, txt,
                                (tx, ty), 0, vsf,
                                box_color, vtf, cv2.LINE_AA)
                objs_for_frame.append({
                    "bbox": (x1, y1, x2, y2),
                    "vx":   v,
                    "vy":   0.0,
                    "cls":  cls,
                    "id":   tid,
                    "predicted": False,
                })

            for obj in missing_kalman_objects:
                x1, y1, x2, y2 = obj.bbox
                v = obj.vx + hxscroll.last_shift * MAX_FPS
                if not minimal_mode and show_boxes:
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (160, 160, 160), lw)
                if not minimal_mode:
                    box_color = (160, 160, 160)
                    txt = f"kf {obj.tid if obj.tid is not None else '-'} v={v:+.0f} pred"
                    (tw, th), _ = cv2.getTextSize(txt, 0, vsf, vtf)
                    tx = int((x1 + x2) / 2) - tw // 2
                    ty = int((y1 + y2) / 2) + th // 2
                    cv2.putText(annotated, txt,
                                (tx, ty), 0, vsf,
                                box_color, vtf, cv2.LINE_AA)
                objs_for_frame.append({
                    "bbox": (x1, y1, x2, y2),
                    "vx":   v,
                    "vy":   obj.vy,
                    "cls":  obj.cls,
                    "id":   obj.tid,
                    "predicted": True,
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

            planner_decision: Optional[PlannerDecision] = None
            if lane_classes:
                planner_decision = planner.choose(
                    truth_xy, lane_classes, objs_for_frame, gridscroll.lane_h
                )
                if not minimal_mode:
                    draw_planner_decision(annotated, planner_decision, planner_auto)

                ready_for_action = (
                    not _active_anims and
                    now - last_planner_action_ts >= PLANNER_MIN_ACTION_INTERVAL
                )
                if (planner_auto and ready_for_action and
                        planner_decision.action != "stay"):
                    send_direction(planner_decision.action)
                    last_planner_action_ts = now

            _run_logger.log_frame(
                green_xy=green_xy,
                red_xy=(red_pred_x + red_offset_x,
                        red_pred_y + red_offset_y),
                truth_xy=truth_xy,
                objects=objs_for_frame,
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

            if (not minimal_mode and show_detections and
                    not show_grid and gridscroll.prev_offset is not None):
                draw_lane_assigned_boxes(annotated, lane_classes)

            if not minimal_mode and show_grid and gridscroll.prev_offset is not None:
                lh = gridscroll.lane_h
                m  = grid_margin
                shift = cv2.getTrackbarPos("grid_shift", "Detections")
                draw_lane_classifications(annotated, lane_classes)
                y0_grid = (int(round(gridscroll.prev_offset)) + shift) % lh
                for y in range(y0_grid, H, lh):
                    cv2.line(annotated, (0, y), (W, y), (0, 255, 0), 1)
                    yb = y + (lh - m) - 1
                    if yb < H and yb != y:
                        cv2.line(annotated, (0, yb), (W, yb), (0, 200, 0), 1)
                cv2.putText(annotated,
                            f"scroll={gridscroll.cum_scroll_y:+.2f} o={gridscroll.prev_offset:.2f} h={lh} m={m} base={lane_idx_base} X={cycle_x:.1f} start={gridscroll.start_offset} log_y={log_y_offset:+d} s={shift}",
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
                lane_smoother.prune(frame_count)

            if not minimal_mode:
                _draw_buttons(annotated)
            cv2.imshow("Detections", annotated)

            if not minimal_mode and show_sobel and gridscroll.last_binary is not None:
                vis = (gridscroll.last_binary * 255).astype(np.uint8)
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                if gridscroll.prev_offset is not None:
                    lh = gridscroll.lane_h
                    m  = grid_margin
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
                if show_grid:
                    minimal_mode = False
            elif key == ord("s"):
                show_sobel = not show_sobel
                if not show_sobel:
                    cv2.destroyWindow("Sobel")
            elif key == ord("x"):
                show_sobx = not show_sobx
                if not show_sobx:
                    cv2.destroyWindow("SobelX")
            elif key == ord("b"):
                show_boxes = not show_boxes
                if show_boxes:
                    minimal_mode = False
            elif key == ord("d"):
                show_detections = not show_detections
            elif key == ord("h"):
                minimal_mode = not minimal_mode
            elif key == ord("a"):
                planner_auto = not planner_auto
                print(f"planner auto={'on' if planner_auto else 'off'}")
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

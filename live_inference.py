import os
import subprocess
import threading
import time
from dataclasses import dataclass
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
    first_ts:  float
    last_ts:   float
    cum_disp:  float   # cumulative signed displacement (px) from first sample
    prev_x1:   float
    prev_x2:   float
    prev_cx:   float
    last_seen: int     # frame index of most recent update


class VelocityTracker:
    def __init__(self):
        self._tracks: dict = {}

    def update(self, tid: int, x1: float, x2: float, ts: float, frame_idx: int):
        cx = (x1 + x2) / 2
        clipped = x1 <= EDGE_EPS or x2 >= W - EDGE_EPS
        s = self._tracks.get(tid)
        if s is None:
            # Entry phase: defer initialization until the bbox is fully in view.
            if clipped:
                return
            self._tracks[tid] = _TrackState(
                first_ts=ts, last_ts=ts, cum_disp=0.0,
                prev_x1=x1, prev_x2=x2, prev_cx=cx,
                last_seen=frame_idx,
            )
            return
        # Exit phase: bbox now touches a screen edge — freeze velocity at its
        # last known value. Just refresh last_seen so we don't prune prematurely.
        if clipped:
            s.last_seen = frame_idx
            return
        # Fully in view: standard centroid delta.
        s.cum_disp += cx - s.prev_cx
        s.last_ts   = ts
        s.prev_x1   = x1
        s.prev_x2   = x2
        s.prev_cx   = cx
        s.last_seen = frame_idx

    def velocity(self, tid: int) -> Optional[float]:
        """Cumulative displacement / cumulative time, in pixels/second."""
        s = self._tracks.get(tid)
        if s is None:
            return None
        dt = s.last_ts - s.first_ts
        if dt <= 0:
            return None
        return s.cum_disp / dt

    def prune(self, current_frame: int):
        """Drop tracks that haven't appeared in PRUNE_AFTER_FRAMES frames."""
        stale = [tid for tid, s in self._tracks.items()
                 if current_frame - s.last_seen > PRUNE_AFTER_FRAMES]
        for tid in stale:
            del self._tracks[tid]

EMULATOR   = "/Users/agent/Library/Android/sdk/emulator/emulator"
AVD_NAME   = "Small_Phone"
MODEL_PATH = "model_weight/model.pt"
W, H       = 240, 480
FIFO_PATH  = "/tmp/crossybot_screen.h264"
MAX_FPS    = 30


def start_avd():
    print("Starting AVD…")
    proc = subprocess.Popen(
        [EMULATOR, "-avd", AVD_NAME, "-no-audio", "-no-boot-anim"],
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


def transform_frame(frame: np.ndarray) -> np.ndarray:
    return cv2.warpAffine(frame, _TRANSFORM, (W, H))


YOLO_CONF         = 0.1    # very low so even faint chicken misfires are caught
VEHICLE_CONF      = 0.8    # vehicles only visualized + tracked above this threshold
CHAR_START_XY     = (223.5, 315.0)   # observed centroid of chicken at game start
CHAR_START_TOL    = 6.0    # px tolerance around the start position when locking on
CHAR_MAX_JUMP_PX  = 22     # max plausible character displacement between frames (one hop)
CHAR_FORGET_FRAMES = 30    # drop last-known position after this many missed frames


def main():
    emu_proc = start_avd()
    launch_app()

    model  = YOLO(MODEL_PATH)
    stream = FrameStream()
    vtrack = VelocityTracker()
    input("Press Enter to start detection… ")
    print("Streaming — press q to quit.")

    try:
        frame_count    = 0
        character_id   = None    # current BoT-SORT track id of the chicken
        char_last_xy   = None    # last seen (cx, cy) — used to re-acquire when id changes
        char_last_seen = -1      # frame index of last successful localisation
        while True:
            frame   = transform_frame(stream.next_frame())
            results = model.track(
                frame,
                imgsz=480, device="mps", conf=YOLO_CONF,
                persist=True,
                tracker="botsort.yaml",
                verbose=False,
            )

            raw_boxes = results[0].boxes
            if raw_boxes is not None and raw_boxes.id is not None:
                ids_t   = raw_boxes.id.int()
                confs_t = raw_boxes.conf
                xyxys_t = raw_boxes.xyxy
                cxs = (xyxys_t[:, 0] + xyxys_t[:, 2]) / 2
                cys = (xyxys_t[:, 1] + xyxys_t[:, 3]) / 2

                # Step 1: initial lock-on at the known start position.
                if character_id is None and char_last_xy is None:
                    sx, sy = CHAR_START_XY
                    ds = ((cxs - sx) ** 2 + (cys - sy) ** 2).sqrt()
                    within = ds <= CHAR_START_TOL
                    if within.any():
                        widx = within.nonzero(as_tuple=True)[0]
                        best = widx[ds[widx].argmin()]
                        character_id  = int(ids_t[best].item())
                        char_last_xy  = (float(cxs[best]), float(cys[best]))
                        char_last_seen = frame_count
                        print(f"Character locked on track id={character_id}")

                # Step 2: if the locked id is still present, refresh state.
                elif character_id is not None and (ids_t == character_id).any():
                    bidx = (ids_t == character_id).nonzero(as_tuple=True)[0][0]
                    char_last_xy   = (float(cxs[bidx]), float(cys[bidx]))
                    char_last_seen = frame_count

                # Step 3: re-acquire — locked id is gone but we know roughly where
                # the chicken was. Adopt the nearest detection within max jump.
                elif char_last_xy is not None:
                    if frame_count - char_last_seen > CHAR_FORGET_FRAMES:
                        # Position is too stale to trust; give up and wait.
                        char_last_xy = None
                        character_id = None
                    else:
                        lx, ly = char_last_xy
                        ds = ((cxs - lx) ** 2 + (cys - ly) ** 2).sqrt()
                        within = ds <= CHAR_MAX_JUMP_PX
                        if within.any():
                            widx = within.nonzero(as_tuple=True)[0]
                            best = widx[ds[widx].argmin()]
                            character_id  = int(ids_t[best].item())
                            char_last_xy  = (float(cxs[best]), float(cys[best]))
                            char_last_seen = frame_count

                # Filter what gets visualised: high-conf vehicles + character.
                keep = confs_t >= VEHICLE_CONF
                if character_id is not None:
                    keep = keep | (ids_t == character_id)
                results[0].boxes = raw_boxes[keep]

            annotated = results[0].plot()

            # Mirror ultralytics Annotator font metrics so text aligns with labels
            lw  = max(round(sum(annotated.shape) / 2 * 0.003), 2)
            sf  = lw / 3        # YOLO label font scale
            tf  = max(lw - 1, 1)
            vsf = sf * 0.7      # velocity text at 0.7x
            vtf = max(tf - 1, 1)

            boxes = results[0].boxes
            if boxes is not None and boxes.id is not None:
                ids   = boxes.id.int().cpu().tolist()
                xyxys = boxes.xyxy.cpu().tolist()
                ts    = time.time()
                for tid, (x1, y1, x2, y2) in zip(ids, xyxys):
                    if tid == character_id:
                        cv2.rectangle(annotated,
                                      (int(x1), int(y1)), (int(x2), int(y2)),
                                      (0, 0, 255), 2)
                        continue
                    # Everything left here is a vehicle (conf >= VEHICLE_CONF)
                    vtrack.update(tid, x1, x2, ts, frame_count)
                    v = vtrack.velocity(tid)
                    if v is None:
                        continue
                    txt = f"v={v:+.0f}"
                    (tw, th), _ = cv2.getTextSize(txt, 0, vsf, vtf)
                    tx = int((x1 + x2) / 2) - tw // 2
                    ty = int((y1 + y2) / 2) + th // 2
                    cv2.putText(annotated, txt,
                                (tx, ty), 0, vsf,
                                (0, 255, 255), vtf, cv2.LINE_AA)

            del results  # release MPS tensors immediately

            frame_count += 1
            if frame_count % 100 == 0:
                torch.mps.empty_cache()
            if frame_count % 30 == 0:
                vtrack.prune(frame_count)

            cv2.imshow("Detections", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()
        emu_proc.terminate()


if __name__ == "__main__":
    main()

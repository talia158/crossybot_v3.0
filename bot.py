import math
import os
import subprocess
import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO

EMULATOR   = "/Users/talia/Library/Android/sdk/emulator/emulator"
AVD_NAME   = "Small_Phone"
MODEL_PATH = "model_weight/model.pt"
W, H       = 480, 960
FIFO_PATH  = "/tmp/crossybot_screen.h264"
MAX_FPS    = 30

YOLO_CONF  = 0.8

ROTATE_DEG = 14.5   # clockwise
SHEAR_DEG  = 12.5   # clockwise x-shear


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


def _build_transform(w: int, h: int, shear_deg: float) -> np.ndarray:
    M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), ROTATE_DEG, 1.0)
    R = np.vstack([M_rot, [0, 0, 1]])               # 2×3 → 3×3
    s = math.tan(math.radians(shear_deg))
    S = np.array([[1, s, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return (S @ R)[:2].astype(np.float32)           # back to 2×3


def transform_frame(frame: np.ndarray, shift_x: int = 0,
                    shear_deg: float = SHEAR_DEG) -> np.ndarray:
    """Rotate+shear then translate left by `shift_x` px. Equivalent to
    composing T @ S @ R: rotation first, shear next, translation last."""
    M = _build_transform(W, H, shear_deg)
    M[0, 2] -= float(shift_x)
    return cv2.warpAffine(frame, M, (W, H))


def main():
    emu_proc = start_avd()
    launch_app()

    model  = YOLO(MODEL_PATH)
    stream = FrameStream()
    cv2.namedWindow("Detections")
    cv2.createTrackbar("shift_x", "Detections", 182, 400, lambda v: None)
    input("Press Enter to start detection… ")
    print("Streaming — press q to quit.")

    try:
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

            annotated = results[0].plot()
            del results  # release MPS tensors immediately

            cv2.imshow("Detections", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()
        emu_proc.terminate()


if __name__ == "__main__":
    main()

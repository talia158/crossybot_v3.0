import os
import subprocess
import threading
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

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


ROTATE_DEG = 14.5  # clockwise → positive angle for cv2


def rotate_frame(frame: np.ndarray, deg: float) -> np.ndarray:
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(frame, M, (w, h))


def main():
    emu_proc = start_avd()
    launch_app()

    model  = YOLO(MODEL_PATH)
    stream = FrameStream()
    print("Streaming — press q to quit.")

    try:
        frame_count = 0
        while True:
            frame     = rotate_frame(stream.next_frame(), ROTATE_DEG)
            results   = model(frame, imgsz=480, device="mps", verbose=False)
            annotated = results[0].plot()
            del results  # release MPS tensors immediately

            frame_count += 1
            if frame_count % 100 == 0:
                torch.mps.empty_cache()  # flush MPS allocator every 100 frames

            cv2.imshow("Detections", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()
        emu_proc.terminate()


if __name__ == "__main__":
    main()

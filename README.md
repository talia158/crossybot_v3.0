
# crossybot_v3.0

 

Crossy Road bot that uses a fine-tuned YOLO model for object detection, Android emulator screen streaming, motion/scroll estimation, and A* path planning to choose safe moves in real time.

 
## What it does

  

`crossybot_v3.0` runs Crossy Road inside an Android Emulator, captures the game screen through `adb screenrecord`, detects moving objects with a YOLO model, estimates object velocity and world scrolling, tracks the player character, and sends D-pad key events back to the emulator.

  

Core capabilities:

  

- Live H.264 screen streaming from Android Emulator via ADB.

- Fine-tuned YOLO inference and object tracking with Ultralytics + BoT-SORT.

- Character detection using color thresholding and morphology.

- Horizontal and vertical world-scroll estimation using Sobel edge signals.

- Per-object velocity estimation for moving hazards.

- A* planning over candidate moves: up, wait, left, and right.

- JSONL run logging for later debugging and analysis.

- OpenCV debug windows for detections, path planning, Sobel views, and grid overlays.

  

## Repository structure

  

```text

crossybot_v3.0/

├── live_inference.py

└── model_weight/

└── model.pt

```

  

## Requirements

  

### System requirements

  

- Apple Silicon Mac (ideally M4 or above).

- Android Studio / Android Emulator installed.

-  `adb` available on your shell `PATH`.

- An Android Virtual Device named `Small_Phone`, or an updated `AVD_NAME` in the script.

- Crossy Road installed on the emulator.

- Python 3.10+ recommended.

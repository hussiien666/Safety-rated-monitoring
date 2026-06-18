# Cobot Vision Safety Monitor — Control Console

A professional desktop GUI for a collaborative-robot (cobot) vision safety
system. It unifies the work from `PHASE1`–`PHASE5` and
`COBOT_VISION_SAFETY_MONITOR.py` into one polished application.

![status](https://img.shields.io/badge/build-passing-27d796) ![py](https://img.shields.io/badge/python-3.10-4d9bff)

## What it does

| Subsystem | Technology |
|-----------|-----------|
| Object detection | **YOLOv8** (Ultralytics, CUDA) |
| Hand tracking & proximity | **MediaPipe Hands** |
| Safety logic | Hand-area → distance proxy → GREEN *safe* / RED *emergency stop* |
| Robotic arm | Live **3D URDF** render (pyrender) — freezes red on a stop event |
| UI | **PySide6 / Qt 6** premium dark console |

When a hand crosses the configured proximity threshold the system trips to an
**EMERGENCY STOP** state and the 3D arm freezes — exactly the behaviour a real
cobot light-curtain / safety controller would enforce.

## Run

```bash
py COBOT_GUI.py
```

First launch loads the YOLOv8 weights (`yolov8s.pt`) and MediaPipe models, then
opens the camera — give it a few seconds.

## Controls

| Control | Action |
|---------|--------|
| **START / STOP MONITOR** | begin / end live monitoring |
| **YOLO confidence** slider | detection sensitivity |
| **Safety distance** slider | hand-area threshold for the stop zone |
| **Show hand skeleton** | toggle MediaPipe landmark overlay |
| **Mirror camera** | flip the feed horizontally |
| **Compute** | CUDA (GPU) / CPU (applies on next START) |
| **Capture Snapshot** | save the annotated frame to `snapshots/` |
| `Space` start/stop · `+/-` threshold · `S` snapshot · `Q`/`Esc` quit |

## Live metrics

Throughput (FPS), inference latency (ms), hand count, object count, proximity %,
a real-time FPS / latency chart, and a rolling **safety timeline** strip.

## Architecture

```
COBOT_GUI.py
├── VisionPipeline      YOLO + MediaPipe, pure CV → annotated frame + FrameResult
├── VisionWorker (QThread)   camera capture + pipeline, off the UI thread
├── RobotArm3D / render_arm_2d   URDF 3D render with graceful 2D fallback
├── ArmWorker (QThread)      continuous arm rendering, ~30 fps
└── MainWindow          Qt UI: header, control sidebar, dual live views,
                        KPI cards, charts, safety timeline, system log
```

Frames cross thread boundaries as deep-copied `QImage`s. Heavy ML imports are
lazy (inside the worker threads) so the window paints instantly.

## Self-test (no camera / no display)

```bash
QT_QPA_PLATFORM=offscreen py COBOT_GUI.py --selftest
```

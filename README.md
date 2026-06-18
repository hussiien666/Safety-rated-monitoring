# GRAD VISION — Cobot Vision Safety System

A computer-vision safety system for a collaborative robot (cobot) arm, built as a
graduation project. It fuses **object detection**, **hand-proximity safety
logic**, **face recognition**, and a **live 3D robotic-arm visualisation** into a
single PySide6 "mission control" desktop console — plus a set of standalone
phase scripts and a PyBullet arm simulation used during development.

![status](https://img.shields.io/badge/build-passing-27d796)
![py](https://img.shields.io/badge/python-3.10-4d9bff)
![ui](https://img.shields.io/badge/UI-PySide6%20%2F%20Qt6-a855f7)

---

## Table of contents

- [What it does](#what-it-does)
- [The main app — `COBOT_GUI.py`](#the-main-app--cobot_guipy)
  - [Run](#run)
  - [Layout & features](#layout--features)
  - [Controls](#controls)
  - [Configuration knobs](#configuration-knobs)
  - [Robotic-arm joint configuration](#robotic-arm-joint-configuration)
  - [Architecture](#architecture)
  - [Self-test](#self-test)
- [Other scripts](#other-scripts)
- [Face recognition](#face-recognition)
- [Project layout](#project-layout)
- [Installation](#installation)
- [Troubleshooting](#troubleshooting)

---

## What it does

| Subsystem | Technology |
|-----------|-----------|
| Object detection | **YOLOv8** (Ultralytics, CUDA/CPU) |
| Hand tracking & proximity | **MediaPipe Hands** |
| Face recognition | **InsightFace** (`buffalo_l`) + a local face database |
| Safety logic | Hand-box area → distance proxy → **GREEN** *safe* / **RED** *emergency stop* |
| Robotic arm | Live **3D URDF** render (pyrender) with a 2D fallback |
| Voice alerts | Spoken safety alerts + per-person greeting (pyttsx3 → Windows SAPI → beep) |
| UI | **PySide6 / Qt 6** vibrant dark console |

When a hand crosses the configured proximity threshold the system trips to an
**EMERGENCY STOP** state and the 3D arm freezes — mirroring what a real cobot
light-curtain / safety controller would enforce.

---

## The main app — `COBOT_GUI.py`

The production console. Everything from the development phases is integrated here.

### Run

```bash
py COBOT_GUI.py
```

> Use `py` (the Windows Python launcher), **not** `python`. First launch loads the
> YOLOv8 weights (`yolov8s.pt`), MediaPipe models, the InsightFace database, and
> the URDF arm — give it a few seconds.

### Layout & features

- **Gradient header** with an animated SAFE / STOP status pill.
- **Left rail** — colour-coded ring gauges: throughput (FPS), proximity (% of
  limit), inference latency (ms).
- **Centre** — the hero live camera feed with YOLO boxes, hand skeletons,
  proximity boxes, and face-ID labels.
- **Right rail** — the live **3D robotic arm** (URDF via pyrender), with status
  chips for arm state, object count, hand count, and identity.
- **Robot link colour key** — a full-width strip that lists every robot link
  next to its render colour, so each coloured link is identifiable.
- **Telemetry strip** — live FPS / latency sparklines and a rolling safety
  timeline.
- **Control dock** — start/stop, sliders, feature toggles, device selector,
  snapshot, and a system log.

The header carries the title and an animated SAFE / STOP pill plus an **uptime**
indicator (the wall-clock time display was removed).

The 3D arm animates each joint within its own configured angle range, colours
every link distinctly, keeps the whole range of motion framed in view, and
**freezes (turns red) on an emergency stop**. If the URDF or OpenGL is
unavailable it degrades gracefully to a hand-drawn 2D arm so the panel is never
empty.

#### Voice alerts & greeting

The console speaks through the laptop speaker (backend auto-selected:
**pyttsx3 → Windows SAPI → beep**, so it works on Windows with nothing to
install). Toggle it with the **"Voice alerts (hello + red zone)"** checkbox.

- **Red-zone alert** — on *entering* the red zone it says *"Red zone"* once,
  sounds a repeating buzzer, and **flashes the robotic-arm window** red — all
  for `RED_ALERT_SECS`, then they stop on their own. (The flash is on the
  **robot/arm view**, not the camera feed.)
- **Greeting** — when a known person appears the system says *"Hello &lt;name&gt;"*
  **once** for each (e.g. *"Hello Hussien. Hello Mazen"*). The same person is not
  greeted again until they've been out of frame for `GREET_REENTRY_SECS` (8 s),
  so brief recognition drop-outs don't repeat the greeting.
- **Emergency-stop priority** — emergency-stop phrases are spoken at **high
  priority**: they replace anything queued and block greetings until they
  finish. A greeting is **dropped** if a safety alert is in flight, and
  greetings are suppressed entirely while the emergency stop is engaged — so the
  safety alert always wins if the two clash.

> The voice layer is developed standalone in **`PHASE8.py`** (see *Other
> scripts*) and integrated here as the `AudioAlerter` class.

### Controls

| Control | Action |
|---------|--------|
| **START / STOP** | begin / end live monitoring |
| **Safety distance** slider | hand-box-area threshold for the stop zone |
| **Hand skeleton** checkbox | toggle the MediaPipe landmark overlay |
| **Mirror camera** checkbox | flip the feed horizontally |
| **Face ID** checkbox | toggle face recognition |
| **Object detection** checkbox | enable / disable YOLO object detection live |
| **Compute** selector | CUDA (GPU) / CPU — applies on next START |
| **Snapshot** | save the annotated frame to `snapshots/` |
| Keyboard | `Space` start/stop · `+/-` threshold · `S` snapshot · `Q`/`Esc` quit |

> **Note:** Detection **confidence values are hidden** on every overlay — object
> and face labels show only the name, not a score. The YOLO confidence threshold
> is kept internally at its default and is no longer exposed in the UI.

### Configuration knobs

All near the top of `COBOT_GUI.py`:

| Constant | Meaning |
|----------|---------|
| `CAM_INDEX` | camera device index |
| `FRAME_WIDTH` / `FRAME_HEIGHT` | capture resolution |
| `FLIP_VIEW` | mirror the camera by default |
| `YOLO_MODEL_PATH` | YOLOv8 weights file (`yolov8s.pt`) |
| `YOLO_CONF` | internal detection confidence (hidden from UI) |
| `MAX_HANDS`, `HAND_DET_CONF`, `HAND_TRACK_CONF` | MediaPipe hand settings |
| `AREA_THRESHOLD`, `THRESHOLD_MIN/MAX/STEP` | hand-area safety thresholds |
| `URDF_PATH` | path to the robot URDF |
| `ARM_W`, `ARM_H` | arm render resolution |
| `ARM_SWEEP_DEG` | max swing per joint (cap, default 100°) |
| `ARM_SPEED` | animation speed |
| `FACE_DB_PATH`, `FACE_THRESHOLD`, `FACE_EVERY_N` | face-recognition settings |
| `VOICE_ALERTS_DEFAULT` | voice alerts on/off by default |
| `RED_ZONE_PHRASE`, `BUZZER_FREQ/DUR_MS/GAP_S` | red-zone spoken phrase + buzzer tone |
| `ESTOP_ON_PHRASE`, `ESTOP_OFF_PHRASE` | emergency-stop spoken phrases |
| `RED_ALERT_SECS` | red-zone alert length (voice + buzzer + arm-window flash) |
| `GREET_REENTRY_SECS` | re-greet cooldown after a known person leaves frame |
| `SNAPSHOT_DIR` | where snapshots are saved |

### Robotic-arm joint configuration

The arm in the URDF has four actuated joints — `m2, m3, m4, m5` (indices `0–3`);
`base` and `end effector` are fixed. Motion is controlled by two dicts near the
top of the file:

```python
# Per-joint sweep range, in degrees (index 0=m2, 1=m3, 2=m4, 3=m5)
RANGE_JOINTS = {
    0: (60.0, 90.0),    # m2
    1: (60.0, 90.0),    # m3  (link 2)
    2: (60.0, 90.0),    # m4
    3: (60.0, 120.0),   # m5  (link 4)
}

# Hold a joint at a fixed angle (overrides RANGE_JOINTS for that joint)
FIXED_JOINTS = {3: 100.0}   # link 4 locked at 100°
```

- **`RANGE_JOINTS[i] = (min, max)`** — joint `i` oscillates between `min` and
  `max` degrees.
- **`FIXED_JOINTS[i] = angle`** — joint `i` is held still at `angle` (remove the
  entry to let it sweep its range again).
- Joint angles are capped at `±ARM_SWEEP_DEG` so links never coil through one
  another, and the IK/limits keep continuous joints from spinning a full 360°.

### Architecture

```
COBOT_GUI.py
├── VisionPipeline          YOLO + MediaPipe + InsightFace → annotated frame + FrameResult
├── VisionWorker (QThread)  camera capture + pipeline, off the UI thread
├── RobotArm3D              URDF 3D render: per-link colour, per-joint motion,
│                           full-motion framing, legend export
├── render_arm_2d           dependency-free 2-link fallback
├── ArmWorker (QThread)     continuous arm rendering (~30 fps) + legend signal
├── LinkLegend              painted full-width robot link colour key
├── AudioAlerter            spoken alerts + greeting + buzzer, priority queue
└── MainWindow              Qt UI: header, rails, dual live views, gauges,
                            charts, safety timeline, control dock, log
```

Frames cross thread boundaries as deep-copied `QImage`s. Heavy ML imports are
lazy (inside the worker threads) so the window paints instantly.

### Self-test

Runs the full UI build + one update cycle with **no camera and no display**:

```bash
QT_QPA_PLATFORM=offscreen py COBOT_GUI.py --selftest
```

Prints `[selftest] PASS` on success.

---

## Other scripts

These are the development phases the GUI was built from. They run standalone
(each opens its own OpenCV / PyBullet window).

| Script | Purpose |
|--------|---------|
| `PHASE1.py` | Minimal **YOLOv8** object detection from the webcam. |
| `PHASE2.py` | **MediaPipe Hands** proximity safety monitor (GREEN/RED zones via hand-box area). |
| `PHASE3.py` | YOLO + MediaPipe with an **Arduino serial** link (`COM4`) to drive real hardware. |
| `PHASE4.py` | Combined vision pipeline (object + hand) with safety logic. |
| `PHASE5.py` | **PyBullet** robotic-arm path-tracing sim — the arm follows square / rectangle / triangle / circle paths and shows live tracking error. |
| `PHASE6.py` | Full **integrated dashboard** — YOLO + hand safety + live **3D URDF arm** composed into one OpenCV "mission control" window (the pre-GUI integrated monitor). |
| `PHASE7.py` | Standalone **InsightFace** webcam face recogniser (loads `faces_db.npz`). |
| `PHASE8.py` | Standalone **voice safety alert & greeting** system — the GUI's `AudioAlerter` developed on its own: red-zone spoken alert + buzzer, per-person *"Hello &lt;name&gt;"* greeting, and emergency-stop-over-greeting priority. |
| `COBOT_VISION_SAFETY_MONITOR.py` | The pre-GUI integrated OpenCV safety monitor. |
| `recognize_webcam.py` | Standalone **InsightFace** webcam face recogniser. |
| `build_database.py` | Builds the face database (`faces_db.npz`) from `known_faces/`. |

**`PHASE8.py` run modes** (no camera or GUI needed):

```bash
py PHASE8.py              # interactive REPL: type events, hear the audio
py PHASE8.py --demo       # scripted audible demo: greet → red zone → e-stop clash
py PHASE8.py --selftest   # silent logic self-test (greeting / priority rules)
```

---

## Face recognition

Face recognition uses **InsightFace** (`buffalo_l`) with a local database.

1. Drop reference images into `known_faces/` (one or more images per person;
   the filename / subfolder is used as the identity name).
2. Build the database:
   ```bash
   py build_database.py
   ```
   This writes `faces_db.npz`.
3. The GUI (and `recognize_webcam.py`) load `faces_db.npz` automatically.
   Matches above `FACE_THRESHOLD` (default `0.60`) are named; others show as
   *Unknown*.

---

## Project layout

```
GRAD VISION/
├── COBOT_GUI.py                     ← main application (this is the one to run)
├── COBOT_VISION_SAFETY_MONITOR.py   pre-GUI integrated monitor
├── PHASE1.py … PHASE8.py            development phases (detection → sim →
│                                    integrated dashboard → face → voice alerts)
├── recognize_webcam.py              standalone face recogniser
├── build_database.py                builds faces_db.npz from known_faces/
├── faces_db.npz                     prebuilt face database
├── known_faces/                     reference face images
├── yolov8s.pt                       YOLOv8 weights
├── urdf1/ , URDF/                   robot URDF + meshes
├── snapshots/                       saved annotated frames
├── README.md                        this file
└── README_GUI.md                    GUI-focused quick reference
```

---

## Installation

Requires **Python 3.10** (the shared env must stay on `numpy < 2`).

```bash
py -m pip install PySide6 opencv-python ultralytics mediapipe ^
                  insightface onnxruntime-gpu ^
                  pyrender yourdfpy trimesh ^
                  pybullet pyserial ^
                  "numpy<2"
```

Notes:
- Keep **numpy `1.26.4`** and **opencv-python `4.10.x`** — numpy 2 breaks the
  tensorflow/mediapipe import chain in this environment.
- Use `onnxruntime-gpu` for CUDA face recognition, or `onnxruntime` for CPU.
- `pyserial` is only needed for the hardware phase (`PHASE3.py`).
- **Voice alerts** need nothing extra on Windows (they fall back to the built-in
  SAPI voice). For nicer in-process speech, `py -m pip install pyttsx3`.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Window opens but camera is black | check `CAM_INDEX`; another app may hold the camera |
| 3D arm shows the flat 2D fallback | URDF/OpenGL unavailable — check `URDF_PATH` and `pyrender`/`yourdfpy` install |
| Arm links overlap / spin | tune `RANGE_JOINTS` / `ARM_SWEEP_DEG`; check joint limits in the URDF |
| Robot too small / off-centre | adjust the camera distance factor in `RobotArm3D._camera_pose` |
| Face ID always "Unknown" | rebuild `faces_db.npz` with `build_database.py`; lower `FACE_THRESHOLD` |
| numpy / mediapipe import error | pin `numpy<2` (`1.26.4`) and `opencv-python 4.10.x` |
| Slow / no GPU | set the Compute selector to **CPU**, or install CUDA-enabled torch / onnxruntime |
```

"""
COBOT VISION SAFETY MONITOR  -  Mission Control Console
=======================================================
A unified, colour-rich PySide6 desktop application for a collaborative-robot
(cobot) vision safety system. It fuses three subsystems that previously lived in
separate OpenCV windows (PHASE1-5 + COBOT_VISION_SAFETY_MONITOR.py) into one
"mission control" console:

    * Live camera  ->  YOLOv8 object detection  +  MediaPipe hand tracking
    * Hand-proximity safety logic (GREEN "safe" / RED "emergency stop")
    * Real-time 3D robotic arm (URDF via pyrender) that freezes on a stop event

Layout: gradient header + animated status pill, a left rail of colour-coded
circular gauges (throughput / proximity / latency), a hero camera feed, the 3D
arm with status chips, a telemetry strip (live charts + safety timeline) and a
control dock.

Engineering notes
-----------------
* Heavy CV runs in a QThread (VisionWorker); the 3D arm renders in its own
  QThread (ArmWorker). Frames cross threads as deep-copied QImages.
* The 3D arm degrades gracefully to a hand-drawn 2D arm if URDF/OpenGL is
  unavailable, so the panel is never empty.
* Heavy ML imports are lazy (inside the worker threads) so the window paints
  instantly and `--selftest` stays light.

Run:        py COBOT_GUI.py
Self-test:  QT_QPA_PLATFORM=offscreen py COBOT_GUI.py --selftest
"""

from __future__ import annotations               # Treat all annotations as strings (lazy; avoids import cost).

import os                                         # File paths, env vars, making the snapshot dir.
import sys                                        # argv (the --selftest flag) and process exit.
import math                                       # Trig for gauges, pulses, and the arm sweep.
import time                                       # Timestamps for FPS, uptime, snapshots, cooldowns.
import queue                                      # Thread-safe FIFO for the audio worker.
import threading                                  # Background audio + buzzer threads.
import subprocess                                 # PowerShell call for the SAPI speech fallback.
import importlib.util                             # Detect whether optional 'pyttsx3' is installed.
from collections import deque                     # Fixed-length ring buffers for the live charts.
from dataclasses import dataclass                 # Lightweight struct for one frame's results.

import cv2                                         # OpenCV: camera capture + all in-frame drawing.
import numpy as np                                 # Arrays: frames, embeddings, math.

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QRectF, QPointF   # Qt core: threads, signals, geometry.
from PySide6.QtGui import (                        # Qt painting + input primitives...
    QImage, QPainter, QColor, QPen, QBrush, QFont, QPainterPath,
    QLinearGradient, QPolygonF, QShortcut, QKeySequence,
)
from PySide6.QtWidgets import (                    # Qt widgets used to build the window...
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSlider,
    QCheckBox, QComboBox, QHBoxLayout, QVBoxLayout, QFrame, QSizePolicy,
    QPlainTextEdit,
)


# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

CAM_INDEX        = 0                               # Which camera to open (0 = default webcam).
FRAME_WIDTH      = 640                             # Requested capture width.
FRAME_HEIGHT     = 480                             # Requested capture height (4:3; 460 was non-standard).
FLIP_VIEW        = True                            # Mirror the feed (selfie view) by default.

YOLO_MODEL_PATH  = "yolov8s.pt"                    # YOLOv8-small weights file.
YOLO_CONF        = 0.70                            # Default detection confidence threshold.

MAX_HANDS        = 2                               # Track at most two hands.
HAND_DET_CONF    = 0.60                            # MediaPipe hand DETECTION confidence gate.
HAND_TRACK_CONF  = 0.60                            # MediaPipe hand TRACKING confidence gate.

AREA_THRESHOLD   = 40_000                          # Default "too close" hand-box area (px²).
THRESHOLD_MIN    = 4_000                           # Slider minimum.
THRESHOLD_MAX    = 150_000                         # Slider maximum.
THRESHOLD_STEP   = 2_000                           # Slider / keyboard step.

URDF_PATH = r"C:\Users\original\OneDrive\Desktop\GRAD VISION\HUSSEIN\urdf\HUSSEIN.urdf"  # 3-D arm model.
ARM_W, ARM_H     = 600, 820                        # Off-screen arm render size (portrait, matches the tall arm panel).
ARM_SWEEP_DEG    = 100.0    # max swing per joint: +-100 deg (only used for joints not in RANGE_JOINTS)
# Per-joint sweep range, in degrees.  Actuated joints (from the URDF) are:
#   index 0 = BASE, 1 = SHOULDER, 2 = ELBOW, 3 = PITCH, 4 = ROLL
# Ranges are symmetric about 0 so the rest pose is the URDF zero configuration,
# which is the naturally extended, front-facing pose for this arm.  (The old
# 60-120 deg centres folded the links back on themselves.)
RANGE_JOINTS     = {                               # Explicit (min,max) sweep per joint index (degrees):
    0: (-30.0, 30.0),   # BASE  (waist)
    1: (-18.0, 18.0),   # SHOULDER
    2: (-22.0, 22.0),   # ELBOW
    3: (-15.0, 15.0),   # PITCH
    4: (-35.0, 35.0),   # ROLL
}
# joint index -> fixed angle in deg; overrides RANGE_JOINTS for that joint.
FIXED_JOINTS     = {}            # none pinned: every joint sweeps gently around the extended pose.
ARM_SPEED        = 0.45                            # Sweep phase increment per rendered arm frame.

# Face recognition (InsightFace buffalo_l + faces_db.npz)
FACE_DB_PATH     = r"C:\Users\original\OneDrive\Desktop\GRAD VISION\faces_db.npz"  # Prebuilt face DB.
FACE_THRESHOLD   = 0.50                            # Min cosine similarity to call a face "known".
FACE_EVERY_N     = 2          # run the recogniser every Nth frame (perf)

SNAPSHOT_DIR     = "snapshots"                     # Folder where PNG snapshots are written.
LOG_DIR          = "logs"                          # Folder where per-session safety-monitor logs are written.

# Voice / audio safety alerts (spoken through the laptop speaker)
VOICE_ALERTS_DEFAULT = True                        # Voice on by default.
RED_ZONE_PHRASE      = "Red zone"   # spoken once on entering the red zone
BUZZER_FREQ          = 600   # Hz - tone of the repeating red-zone buzzer
BUZZER_DUR_MS        = 300   # ms - length of each buzz
BUZZER_GAP_S         = 0.15  # s  - silence between buzzes
ESTOP_ON_PHRASE      = "Emergency stop activated."   # Spoken when e-stop engages.
ESTOP_OFF_PHRASE     = "Emergency stop released."    # Spoken when e-stop releases.

# Red-zone alert episode length: the camera flash, the voice and the buzzer all
# run for this long on entering the red zone, then stop on their own.
RED_ALERT_SECS       = 5.0                         # Duration of one red-zone alert episode.
# Don't greet the same known person again until they've been out of frame for
# this long (s) - stops brief recognition drop-outs re-triggering the greeting.
GREET_REENTRY_SECS   = 100000                      # Effectively "greet once per session" here.


# ════════════════════════════════════════════════════════════════════════════
#  THEME  -  vibrant palette + Qt stylesheet
# ════════════════════════════════════════════════════════════════════════════

class C:
    """Central colour palette."""
    BG          = "#080b15"                        # Deepest background.
    BG2         = "#0d1322"                        # Slightly lighter background (dock).
    PANEL       = "#0f1626"                        # Panel fill.
    CARD        = "#121a2e"                        # Card fill.
    CARD2       = "#0e1525"                        # Inset card fill (log box).
    BORDER      = "#283556"                        # Strong border.
    BORDER_SOFT = "#1b2440"                        # Subtle border.
    TRACK       = "#1d2742"                         # Empty gauge/slider track.
    TEXT        = "#eef2ff"                         # Primary text.
    MUTED       = "#8c96b6"                         # Secondary text.
    FAINT       = "#525d80"                         # Disabled / placeholder text.

    BLUE        = "#3b82f6"                         # Accent hues used by gauges/charts...
    CYAN        = "#22d3ee"
    VIOLET      = "#a39abd"
    GREEN       = "#22c55e"
    AMBER       = "#f59e0b"
    PINK        = "#ec4899"
    RED         = "#ef4444"
    ORANGE      = "#fb923c"
    TEAL        = "#2dd4bf"

    # semantic aliases
    ACCENT      = BLUE                              # Generic accent.
    SAFE        = GREEN                             # "Safe" state colour.
    DANGER      = RED                               # "Stop/danger" state colour.
    WARN        = AMBER                             # Warning colour.


# OpenCV BGR colours used inside the vision pipeline drawing
BGR_SAFE   = (94, 197, 34)      # green             # In-frame box colour: safe hand.
BGR_DANGER = (68, 68, 239)      # red               # In-frame box colour: hand too close.
BGR_WARN   = (11, 158, 245)     # amber             # In-frame box colour: detected objects.
BGR_CYAN   = (238, 211, 34)     # cyan              # In-frame box colour: known faces.
BGR_WHITE  = (245, 245, 245)                        # Label text.
BGR_DARK   = (20, 16, 12)                           # Label background chip.
FONT       = cv2.FONT_HERSHEY_SIMPLEX               # Font for all OpenCV text.


def build_stylesheet() -> str:                      # Returns the global Qt stylesheet (QSS) as a string.
    # NOTE: this is one big f-string of Qt Style Sheet rules; {{ }} are literal
    # braces and {C.X} interpolate palette colours. No Python runs inside it.
    return f"""
    QWidget {{
        color: {C.TEXT};
        font-family: 'Segoe UI', 'Inter', sans-serif;
        font-size: 13px;
    }}
    QWidget#root {{
        background: qlineargradient(x1:0, y1:0, x2:0.6, y2:1,
            stop:0 #080b15, stop:1 #0c1426);
    }}
    QFrame#card {{
        background: {C.CARD};
        border: 1px solid {C.BORDER_SOFT};
        border-radius: 13px;
    }}
    QFrame#dock {{
        background: {C.BG2};
        border: 1px solid {C.BORDER_SOFT};
        border-radius: 14px;
    }}
    QFrame#header {{
        background: #ffffff;
        border: 1px solid #6f4ed1;
        border-radius: 14px;
    }}
    QLabel#h1 {{ font-size: 20px; font-weight: 900; letter-spacing: 1px;
                 color: #1a0640; }}
    QLabel#h2 {{ font-size: 11px; font-weight: 700; color: #3a1d7a; letter-spacing: 3px; }}
    QLabel#section {{ color: {C.CYAN}; font-size: 11px; font-weight: 800;
                      letter-spacing: 2px; }}
    QLabel#panelTitle {{ font-size: 12px; font-weight: 800; letter-spacing: 1px; }}

    QPushButton {{
        background: {C.CARD}; border: 1px solid {C.BORDER}; border-radius: 9px;
        padding: 9px 14px; font-weight: 600; color: {C.TEXT};
    }}
    QPushButton:hover {{ border: 1px solid {C.CYAN}; }}
    QPushButton:pressed {{ background: {C.PANEL}; }}
    QPushButton:disabled {{ color: {C.FAINT}; border-color: {C.BORDER_SOFT}; }}
    QPushButton#primary {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #10b981, stop:1 #22d3ee);
        color: #05231a; border: none; font-weight: 800; font-size: 14px;
    }}
    QPushButton#primary:hover {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #14d39b, stop:1 #4be3f7); }}
    QPushButton#danger {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #ef4444, stop:1 #ec4899);
        color: #2b0710; border: none; font-weight: 800; font-size: 14px;
    }}

    QComboBox {{ background: {C.CARD}; border: 1px solid {C.BORDER};
        border-radius: 8px; padding: 6px 10px; }}
    QComboBox:hover {{ border: 1px solid {C.CYAN}; }}
    QComboBox QAbstractItemView {{ background: {C.PANEL};
        border: 1px solid {C.BORDER}; selection-background-color: {C.BLUE};
        outline: none; }}

    QCheckBox {{ color: {C.TEXT}; spacing: 8px; }}
    QCheckBox::indicator {{ width: 18px; height: 18px; border-radius: 5px;
        border: 1px solid {C.BORDER}; background: {C.CARD}; }}
    QCheckBox::indicator:checked {{ background: {C.CYAN}; border-color: {C.CYAN}; }}

    QSlider::groove:horizontal {{ height: 5px; background: {C.TRACK};
        border-radius: 3px; }}
    QSlider::sub-page:horizontal {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {C.BLUE}, stop:1 {C.CYAN});
        border-radius: 3px; }}
    QSlider::handle:horizontal {{ background: #ffffff; width: 15px; height: 15px;
        margin: -6px 0; border-radius: 8px; }}
    QSlider::handle:horizontal:hover {{ background: {C.CYAN}; }}

    QPlainTextEdit {{ background: {C.CARD2}; border: 1px solid {C.BORDER_SOFT};
        border-radius: 10px; color: {C.MUTED};
        font-family: 'Consolas', 'Cascadia Mono', monospace; font-size: 11px;
        padding: 6px; }}
    """


def qc(h: str, a: int = 255) -> QColor:             # Helper: build a QColor from "#rrggbb" with optional alpha.
    c = QColor(h); c.setAlpha(a); return c          # Create, set transparency, return.


# ════════════════════════════════════════════════════════════════════════════
#  AUDIO SAFETY ALERTS  -  spoken voice from the laptop speaker
# ════════════════════════════════════════════════════════════════════════════

class AudioAlerter:
    """Speaks short safety phrases through the laptop speaker.

    Playback runs on a daemon worker thread so the real-time GUI never blocks.
    The backend is picked at runtime - the first that works wins:

        1. pyttsx3        - in-process offline TTS (SAPI5 voice on Windows)
        2. Windows SAPI   - via PowerShell, needs no extra Python package
        3. winsound/bell  - last-resort two-tone alarm if no TTS is available

    Calls collapse: a newer phrase replaces any still-queued one, so alerts can
    never pile up behind the live loop.

    The red-zone alert is special: it says a short phrase once, then sounds a
    repeating buzzer (on its own thread) until `stop_buzzer()` is called.
    """

    def __init__(self):
        self._q = queue.Queue()                    # Phrases waiting to be spoken.
        self._thread = None                        # The speech worker thread (lazy).
        self._buzzer_stop = threading.Event()      # Flag to stop the buzzer early.
        self._buzzer_thread = None                 # The red-zone buzzer thread.
        # When a high-priority phrase (emergency stop / red zone) is queued or
        # speaking, low-priority phrases (greetings) are dropped so the safety
        # alert always wins if the two clash.
        self._lock = threading.Lock()              # Guards the priority gate + queue.
        self._priority_active = False              # True while a high-priority phrase is in flight.
        # Stay silent in headless / self-test runs (no speakers, would block).
        self._enabled = (VOICE_ALERTS_DEFAULT and  # Voice on by default...
                         os.environ.get("QT_QPA_PLATFORM") != "offscreen")  # ...muted when headless.
        if importlib.util.find_spec("pyttsx3") is not None:  # Prefer pyttsx3 if installed...
            self._backend = "pyttsx3"
        elif sys.platform.startswith("win"):       # ...else SAPI on Windows...
            self._backend = "sapi"
        else:                                      # ...else a plain beep.
            self._backend = "beep"

    def backend_name(self) -> str:
        return self._backend                       # Which backend was chosen.

    def set_enabled(self, on: bool):
        self._enabled = bool(on)                   # Toggle voice at runtime.
        if not self._enabled:
            self.stop_buzzer()                     # Silence the buzzer when disabling.

    def alert_red_zone(self, duration_s=RED_ALERT_SECS):
        """Say the red-zone phrase once, then buzz for `duration_s` seconds and
        stop on its own - a fixed-length alert, not an open-ended one."""
        if not self._enabled:                      # Skip if disabled.
            return
        self.stop_buzzer()                         # Cancel any prior buzzer.
        self._buzzer_stop.clear()                  # Reset the stop flag.
        self._buzzer_thread = threading.Thread(    # Run the episode on its own daemon thread...
            target=self._red_zone_loop, args=(duration_s,),
            name="RedZoneBuzzer", daemon=True)
        self._buzzer_thread.start()                # ...so the GUI never blocks.

    def is_alerting(self) -> bool:
        """True while the red-zone alert episode (phrase + buzzer) is running."""
        t = self._buzzer_thread                    # Snapshot the handle.
        return t is not None and t.is_alive()      # Alive → still alerting.

    def stop_buzzer(self):
        """Stop the repeating red-zone buzzer (no-op if it is not running)."""
        self._buzzer_stop.set()                    # Tell the loop to stop.
        t = self._buzzer_thread                    # Snapshot the handle.
        if t is not None and t.is_alive() and t is not threading.current_thread():  # If running & not us...
            t.join(timeout=2.0)                    # ...wait briefly.
        self._buzzer_thread = None                 # Clear the handle.

    def say(self, text: str, priority: bool = False):
        """Queue a phrase to speak, dropping any older pending phrase.

        A high-priority phrase (emergency stop / red zone) replaces whatever is
        queued and blocks low-priority phrases until it finishes. A low-priority
        phrase (greeting) is dropped if a high-priority phrase is in flight or
        the red-zone alert is sounding - so safety alerts always win a clash.
        """
        if not self._enabled:                      # Silent when disabled.
            return
        self._ensure_thread()                      # Make sure the worker exists.
        with self._lock:                           # Atomic priority check + enqueue.
            if not priority and (self._priority_active or self.is_alerting()):  # Greeting loses a clash...
                return                             # ...drop it.
            self._drain_queue()                    # Collapse any older pending phrase.
            if priority:                           # High priority...
                self._priority_active = True       # ...blocks greetings until done.
            self._q.put((priority, text))          # Enqueue for the worker.

    def _drain_queue(self):
        try:
            while True:                            # Empty the queue...
                self._q.get_nowait()
        except queue.Empty:
            pass                                   # Done.

    def shutdown(self):
        self._enabled = False                      # Stop accepting phrases.
        self.stop_buzzer()                         # Silence the buzzer.
        if self._thread is not None:
            self._q.put(None)                      # Send the worker its exit sentinel.

    # -- internals ----------------------------------------------------------
    def _red_zone_loop(self, duration_s=RED_ALERT_SECS):
        # Speak the phrase once, then buzz until the deadline (or an early stop).
        deadline = time.monotonic() + max(0.5, float(duration_s))  # When the alert ends.
        try:
            self._speak(RED_ZONE_PHRASE)           # Say "Red zone" once.
        except Exception:
            pass                                   # Never crash the thread on a speech error.
        while not self._buzzer_stop.is_set() and time.monotonic() < deadline:  # Until stopped/expired...
            self._buzz_once()                      # ...one buzz...
            self._buzzer_stop.wait(BUZZER_GAP_S)   # ...wait the gap (interruptible).
        # The episode finished on its own - drop our handle so state stays clean.
        if self._buzzer_thread is threading.current_thread():
            self._buzzer_thread = None

    def _buzz_once(self):
        try:
            import winsound                        # Windows tone API.
            winsound.Beep(BUZZER_FREQ, BUZZER_DUR_MS)  # One beep.
        except Exception:
            try:
                print("\a", end="", flush=True)    # Fallback: terminal bell.
            except Exception:
                pass

    def _ensure_thread(self):
        if self._thread is None:                   # Lazily create the speech worker...
            self._thread = threading.Thread(
                target=self._run, name="AudioAlerter", daemon=True)
            self._thread.start()

    def _run(self):
        while True:                                # Worker loop.
            item = self._q.get()                   # Block for a phrase / sentinel.
            if item is None:                       # None = shutdown.
                break
            priority, text = item                  # Unpack.
            try:
                self._speak(text)                  # Speak it (blocking on this thread only).
            except Exception:
                pass
            if priority:                           # After a high-priority phrase...
                with self._lock:
                    # Release the lock only once nothing else is waiting, so a
                    # burst of safety phrases keeps greetings out until it ends.
                    if self._q.empty():
                        self._priority_active = False

    def _speak(self, text: str):
        if self._backend == "pyttsx3":             # --- Preferred in-process TTS ---
            try:
                import pyttsx3
                eng = pyttsx3.init()               # Create engine.
                try:
                    eng.setProperty("rate", 178)   # Speed.
                    eng.setProperty("volume", 1.0) # Volume.
                except Exception:
                    pass
                eng.say(text)                      # Queue text.
                eng.runAndWait()                   # Speak synchronously.
                try:
                    eng.stop()                     # Release engine.
                except Exception:
                    pass
                return
            except Exception:
                # Demote to the platform backend and fall through.
                self._backend = "sapi" if sys.platform.startswith("win") else "beep"

        if self._backend == "sapi":                # --- Windows SAPI via PowerShell ---
            try:
                safe = text.replace("'", " ").replace('"', " ")  # Strip quotes for the PS string.
                ps = (                             # Build the System.Speech one-liner...
                    "Add-Type -AssemblyName System.Speech; "
                    "$v = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    "$v.Rate = 1; $v.Volume = 100; "
                    f"$v.Speak('{safe}')"
                )
                subprocess.run(                    # ...and run it (hidden window, bounded time).
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    timeout=12,
                )
                return
            except Exception:
                self._backend = "beep"             # Demote to a beep.

        # Last resort: an attention-grabbing two-tone alarm.
        try:
            import winsound
            winsound.Beep(920, 220)                # High tone...
            winsound.Beep(680, 320)                # ...then low tone.
        except Exception:
            try:
                print("\a", end="", flush=True)    # Terminal bell as the absolute fallback.
            except Exception:
                pass


# ════════════════════════════════════════════════════════════════════════════
#  VISION PIPELINE  -  pure computer vision, no Qt   (verified core)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class FrameResult:                                  # Immutable-ish struct returned for each processed frame.
    state: str = "GREEN"                            # "GREEN" safe or "RED" too-close.
    hands: int = 0                                  # Number of hands detected.
    objects: int = 0                                # Number of YOLO objects detected.
    collision: bool = False                         # RED state AND objects present.
    max_area: int = 0                               # Largest hand-box area (proximity proxy).
    fps: float = 0.0                                # Frames per second.
    infer_ms: float = 0.0                           # Processing time for this frame (ms).
    device: str = "cpu"                             # Compute device used.
    faces: int = 0                                  # Number of faces detected.
    identities: tuple = ()                          # Names of recognised (known) people.


def _hand_bbox(landmarks, w: int, h: int):          # Pixel bbox + area for one hand's landmarks.
    xs = [lm.x for lm in landmarks]                 # Normalised X of every landmark.
    ys = [lm.y for lm in landmarks]                 # Normalised Y of every landmark.
    x1 = max(0, int(min(xs) * w)); x2 = min(w - 1, int(max(xs) * w))  # Left/right pixel edges (clamped).
    y1 = max(0, int(min(ys) * h)); y2 = min(h - 1, int(max(ys) * h))  # Top/bottom pixel edges (clamped).
    area = max(0, x2 - x1) * max(0, y2 - y1)        # Box area.
    return x1, y1, x2, y2, area                     # Corners + area.


def _label(frame, text, x, y, bg, fg=BGR_WHITE, scale=0.5):  # Draw text on a filled chip.
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)   # Measure the text.
    y = max(y, th + 6)                              # Keep the chip on-screen (don't clip at the top).
    cv2.rectangle(frame, (x, y - th - base - 4), (x + tw + 8, y), bg, -1, cv2.LINE_AA)  # Chip background.
    cv2.putText(frame, text, (x + 4, y - 5), FONT, scale, fg, 1, cv2.LINE_AA)           # Text on top.


class FaceRecognizer:
    """InsightFace (buffalo_l) recogniser backed by faces_db.npz.

    Built lazily inside the worker thread. If insightface, the ONNX model or
    the database are unavailable it disables itself (``available == False``) so
    the rest of the vision pipeline keeps running - mirrors the 3D-arm fallback.
    """

    def __init__(self, db_path, threshold, device):
        self.available = False                     # Becomes True only if everything loads.
        self.error = ""                            # Reason for being unavailable (for the UI).
        self.threshold = threshold                 # Min similarity to accept a match.
        self.names = ()                            # Known names from the DB.
        try:
            import onnxruntime as ort              # To query GPU/CPU providers.
            from insightface.app import FaceAnalysis  # The face model wrapper.

            data = np.load(db_path, allow_pickle=False)   # Load the saved database.
            self.names = data["names"]             # Person labels.
            emb = data["embeddings"].astype(np.float32)   # Their averaged embeddings.
            # pre-normalise DB embeddings so recognition is a single matmul
            self._db = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)  # Unit-length rows.

            avail = ort.get_available_providers()  # Which backends exist here.
            if device == "cuda" and "CUDAExecutionProvider" in avail:  # GPU requested & available...
                providers, ctx = ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
            else:                                  # Otherwise CPU.
                providers, ctx = ["CPUExecutionProvider"], -1
            self.app = FaceAnalysis(name="buffalo_l",   # Build the engine (detection + recognition).
                                    allowed_modules=["detection", "recognition"],
                                    providers=providers)
            self.app.prepare(ctx_id=ctx, det_size=(640, 640))  # Initialise device + input size.
            self.available = True                  # Mark ready.
        except Exception as e:                     # Any failure (missing libs/model/DB)...
            self.error = f"{type(e).__name__}: {e}"  # ...is recorded; recognition just stays off.

    def identify(self, frame):
        """Return [(name, score, (x1, y1, x2, y2)), ...] for faces in `frame`."""
        if not self.available:                     # If recognition is off...
            return []                              # ...return nothing.
        out = []
        for face in self.app.get(frame):           # For each detected face...
            emb = face.embedding.astype(np.float32)  # Its embedding...
            emb /= (np.linalg.norm(emb) + 1e-9)    # ...normalised to unit length.
            sims = self._db @ emb                  # Cosine similarity vs every DB row (one matmul).
            idx = int(np.argmax(sims))             # Index of the best match.
            score = float(sims[idx])               # Its similarity.
            name = str(self.names[idx]) if score >= self.threshold else "Unknown"  # Accept or "Unknown".
            x1, y1, x2, y2 = face.bbox.astype(int) # Face box corners.
            out.append((name, score, (x1, y1, x2, y2)))  # Collect the result.
        return out                                 # All identified faces.


def _draw_faces(frame, faces):                      # Draw boxes + names for recognised faces.
    for name, score, (x1, y1, x2, y2) in faces:
        known = name != "Unknown"                  # Known vs unknown.
        col = BGR_CYAN if known else BGR_WARN      # Cyan for known, amber for unknown.
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)  # Box.
        _label(frame, f"{name}", x1, max(0, y1 - 4), BGR_DARK, col)    # Name label above it.


class VisionPipeline:
    """Owns the YOLO + MediaPipe models. Construct inside the worker thread."""

    def __init__(self, model_path: str, device: str):
        from ultralytics import YOLO              # Lazy import (keeps GUI startup fast).
        import mediapipe as mp                     # Lazy import.

        self.device = device                       # "cuda" or "cpu".
        self.yolo = YOLO(model_path)               # Load YOLO once.
        self.face = FaceRecognizer(FACE_DB_PATH, FACE_THRESHOLD, device)  # Build the face recogniser.
        self._frame_i = 0                          # Frame counter (for face throttling).
        self._last_faces = []                      # Cache of the last face result (re-drawn between runs).
        self._mp = mp                              # Keep a handle to mediapipe.
        self.hands = mp.solutions.hands.Hands(     # Create the hand tracker.
            static_image_mode=False, max_num_hands=MAX_HANDS, model_complexity=0,
            min_detection_confidence=HAND_DET_CONF,
            min_tracking_confidence=HAND_TRACK_CONF)
        self._draw = mp.solutions.drawing_utils    # Landmark drawing helpers.
        self._styles = mp.solutions.drawing_styles # Default skeleton styles.
        self._hands_conn = mp.solutions.hands.HAND_CONNECTIONS  # The hand bone connection map.

    def process(self, frame, conf, threshold, draw_landmarks=True, flip=True,
                recognize_faces=True, detect_objects=True):
        t0 = time.perf_counter()                   # Start timing this frame.
        if flip:                                   # Optional mirror.
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]                      # Frame size.

        objects = 0                                 # Count of YOLO detections.
        if detect_objects:                          # Object detection can be toggled off.
            kw = {"conf": conf, "verbose": False}   # YOLO inference args.
            if self.device:
                kw["device"] = self.device          # Pass device if set.
            for res in self.yolo.predict(frame, **kw):  # Run detection.
                names = res.names                   # id → name lookup.
                for box in res.boxes:               # Each detected box.
                    c = float(box.conf[0])          # Confidence.
                    if c < conf:                    # Skip low-confidence.
                        continue
                    cls = int(box.cls[0])           # Class id.
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Corners as ints.
                    cv2.rectangle(frame, (x1, y1), (x2, y2), BGR_WARN, 2, cv2.LINE_AA)  # Box.
                    _label(frame, f"{names.get(cls, cls)}",  # Class-name label.
                           x1, y1, BGR_DARK, BGR_WARN)
                    objects += 1                    # Tally it.

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe wants RGB.
        rgb.flags.writeable = False                 # Read-only → faster processing.
        hres = self.hands.process(rgb)              # Detect/track hands.
        rgb.flags.writeable = True                  # Restore.

        state = "GREEN"; n_hands = 0; max_area = 0  # Defaults for this frame.
        if hres.multi_hand_landmarks:               # If hands were found...
            n_hands = len(hres.multi_hand_landmarks)  # Count them.
            for lm in hres.multi_hand_landmarks:    # For each hand...
                if draw_landmarks:                  # Optionally draw its skeleton.
                    self._draw.draw_landmarks(
                        frame, lm, self._hands_conn,
                        self._styles.get_default_hand_landmarks_style(),
                        self._styles.get_default_hand_connections_style())
                x1, y1, x2, y2, area = _hand_bbox(lm.landmark, w, h)  # Measure the hand.
                max_area = max(max_area, area)      # Track the closest (largest) hand.
                hot = area > threshold              # "Hot" = too close.
                color = BGR_DANGER if hot else BGR_SAFE  # Box colour by proximity.
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)  # Draw the box.
                pct = min(100, int(area / threshold * 100))  # Proximity as a percentage of the limit.
                _label(frame, f"{pct}% proximity", x1, y1, BGR_DARK, color)  # Label it.
                if hot:                             # If any hand is too close...
                    state = "RED"                   # ...the frame is RED.

        # ---- face recognition (throttled; last result re-drawn between runs) --
        self._frame_i += 1                          # Advance the frame counter.
        if recognize_faces and self.face.available: # If face ID is on and ready...
            if self._frame_i % FACE_EVERY_N == 0:   # ...run it only every Nth frame (perf)...
                self._last_faces = self.face.identify(frame)  # ...refreshing the cache.
            _draw_faces(frame, self._last_faces)    # Draw the (possibly cached) faces every frame.
        else:
            self._last_faces = []                   # Otherwise clear it.
        identities = tuple(n for n, _s, _b in self._last_faces if n != "Unknown")  # Known names only.

        collision = state == "RED" and objects > 0  # Collision risk = too-close hand AND objects present.
        dt = (time.perf_counter() - t0) * 1000.0    # Processing time in ms.
        return frame, FrameResult(state=state, hands=n_hands, objects=objects,  # Annotated frame + results.
                                  collision=collision, max_area=max_area,
                                  infer_ms=dt, device=self.device,
                                  faces=len(self._last_faces), identities=identities)

    def close(self):
        try:
            self.hands.close()                      # Free MediaPipe Hands.
        except Exception:
            pass


# ════════════════════════════════════════════════════════════════════════════
#  3D ROBOTIC ARM  (URDF via pyrender)  +  2D fallback   (verified core)
# ════════════════════════════════════════════════════════════════════════════

class RobotArm3D:
    """Loads a SolidWorks-exported URDF and renders it as a live 3D arm."""

    def __init__(self, urdf_path, width, height):
        import yourdfpy                            # URDF kinematics + scene graph.
        import pyrender                            # Off-screen OpenGL renderer.
        self._pyrender = pyrender                  # Keep a handle for render().
        self.width, self.height = width, height    # Output image size.
        self.robot = yourdfpy.URDF.load(urdf_path) # Parse the URDF.
        self.joint_names = list(self.robot.actuated_joint_names)  # Names of movable joints.
        # Per-joint safe sweep: stay inside each joint's real URDF limits so the
        # links never coil through each other ("arm moving into itself"). We sweep
        # around the mid-point of each joint, using only a fraction of its range.
        # Hard cap on how far any joint may swing: +-100 deg (100 left, 100 right).
        max_swing = math.radians(ARM_SWEEP_DEG)    # Global swing cap in radians.
        self._joint_motion = []                    # Will hold (center, half-range) per joint.
        for idx, n in enumerate(self.joint_names): # For each actuated joint...
            joint = self.robot.joint_map.get(n)    # The joint object.
            lim = getattr(joint, "limit", None)    # Its limit (if any).
            lo = getattr(lim, "lower", None)       # Lower bound.
            hi = getattr(lim, "upper", None)       # Upper bound.
            if lo is not None and hi is not None and hi > lo:  # If it has real limits...
                center = 0.5 * (lo + hi)           # Sweep around the midpoint...
                half = 0.5 * (hi - lo)             # ...using half the range.
            else:  # continuous / unlimited joint -> would spin 360, so clamp it
                center = 0.0                       # Centre at 0...
                half = max_swing                   # ...and cap the swing.
            # never let a joint swing more than +-100 deg from its centre
            half = min(half, max_swing)            # Apply the global cap.
            # explicit min/max sweep range overrides the limit-based motion
            if idx in RANGE_JOINTS:                # If the config pins a range...
                lo_d, hi_d = RANGE_JOINTS[idx]     # ...use it (degrees → radians)...
                center = math.radians(0.5 * (lo_d + hi_d))
                half = math.radians(0.5 * (hi_d - lo_d))
            # held-fixed joints stay at a set angle (zero swing)
            if idx in FIXED_JOINTS:                # If the config fixes this joint...
                center = math.radians(FIXED_JOINTS[idx])  # ...hold it at that angle...
                half = 0.0                         # ...with no swing.
            self._joint_motion.append((center, half))  # Record motion for this joint.
        # Frame the camera around the FULL range of motion (not just the rest
        # pose) so the arm stays centred and never sweeps out of view.
        centers = [c for c, h in self._joint_motion]  # All centres.
        mins    = [c - h for c, h in self._joint_motion]  # All minimum angles.
        maxs    = [c + h for c, h in self._joint_motion]  # All maximum angles.
        sample_cfgs = [centers, mins, maxs]        # A few whole-arm poses to bound.
        for i in range(len(self.joint_names)):       # each joint at its extremes
            for val in (mins[i], maxs[i]):         # ...one joint swung to each extreme...
                cfg = list(centers); cfg[i] = val  # ...others at centre.
                sample_cfgs.append(cfg)
        lo_b = np.full(3, np.inf); hi_b = np.full(3, -np.inf)  # Accumulate a bounding box over all poses.
        for vals in sample_cfgs:                   # For each sampled pose...
            self.robot.update_cfg({n: v for n, v in zip(self.joint_names, vals)})  # Apply it...
            b = self.robot.scene.bounds            # ...read its bounds...
            lo_b = np.minimum(lo_b, b[0]); hi_b = np.maximum(hi_b, b[1])  # ...grow the overall box.
        self._center = 0.5 * (lo_b + hi_b)         # Centre of the motion envelope.
        self._radius = max(float(np.linalg.norm(hi_b - lo_b)), 1e-3)  # Its size (for camera distance).
        self.renderer = pyrender.OffscreenRenderer(width, height)  # Create the GL renderer.
        # One distinct vibrant colour per link, spread evenly around the hue wheel.
        import colorsys                            # HSV → RGB conversion.
        self._link_names = list(self.robot.scene.graph.nodes_geometry)  # Mesh node names.
        n_links = max(len(self._link_names), 1)    # Avoid divide-by-zero.
        self._link_colors = []                     # One RGBA colour per link.
        for k in range(n_links):                   # Evenly spaced hues...
            r, g, b = colorsys.hsv_to_rgb(k / n_links, 0.65, 1.0)
            self._link_colors.append([r, g, b, 1.0])

    def _camera_pose(self):
        # View the arm from the front so it reads as upright and extended. This
        # arm extends along -Y and is fairly planar (low Z), so looking roughly
        # down +Z with the world -Y axis as screen-up presents it standing
        # (base at the bottom, end-effector at the top).
        eyedir = np.array([0.45, -0.35, 1.0])      # Eye direction (front, slightly above).
        eyedir /= np.linalg.norm(eyedir)           # Normalise to a unit direction.
        eye = self._center + eyedir * (self._radius * 1.30)  # Eye position (1.30 = framing margin).
        fwd = self._center - eye; fwd /= np.linalg.norm(fwd)  # Unit forward direction.
        up = np.array([0.0, -1.0, 0.0])            # Screen-up = world -Y (arm stands upright).
        right = np.cross(fwd, up); right /= np.linalg.norm(right)  # Camera right axis.
        true_up = np.cross(right, fwd)             # Orthogonal up axis.
        pose = np.eye(4)                           # 4×4 camera-to-world matrix.
        pose[:3, 0] = right; pose[:3, 1] = true_up # Right + up columns.
        pose[:3, 2] = -fwd;  pose[:3, 3] = eye     # Backward column + position.
        return pose

    def render(self, phase, stopped):
        pyrender = self._pyrender                  # Local alias.
        # Swing each joint up to its capped half-range (+-100 deg), staggering the
        # phase per joint for a natural sweep.
        cfg = {}                                   # Joint-name → angle for this frame.
        for i, n in enumerate(self.joint_names):
            center, half = self._joint_motion[i]   # This joint's motion.
            cfg[n] = center + half * math.sin(phase + i * 0.7)  # Sine sweep, phase-shifted per joint.
        self.robot.update_cfg(cfg)                 # Apply forward kinematics.
        scene = pyrender.Scene(bg_color=[0.04, 0.06, 0.11, 1.0],  # New scene, dark bg...
                               ambient_light=[0.5, 0.5, 0.5])     # ...soft ambient light.
        ts = self.robot.scene                      # Posed trimesh scene.
        for i, node in enumerate(ts.graph.nodes_geometry):  # For each link mesh...
            T, gname = ts.graph.get(node)          # World transform + geometry name.
            color = self._link_colors[i % len(self._link_colors)]  # Its assigned colour.
            material = pyrender.MetallicRoughnessMaterial(  # A shiny metallic material...
                baseColorFactor=color, metallicFactor=0.45, roughnessFactor=0.5)
            mesh = pyrender.Mesh.from_trimesh(ts.geometry[gname], smooth=False,  # Build the mesh...
                                              material=material)
            scene.add(mesh, pose=np.asarray(T))    # Add it at its pose.
        cam_pose = self._camera_pose()             # Camera transform.
        scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4.0), pose=cam_pose)  # 45° camera.
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.2),   # Headlight at the camera.
                  pose=cam_pose)
        color, _ = self.renderer.render(scene)     # Render → RGB.
        return cv2.cvtColor(np.asarray(color), cv2.COLOR_RGB2BGR)  # RGB → BGR for Qt/OpenCV.

    def legend_entries(self):
        """List of (link_name, '#rrggbb') for the GUI side-panel colour key."""
        out = []
        for name, col in zip(self._link_names, self._link_colors):  # Pair each link with its colour...
            hexc = "#%02x%02x%02x" % (int(col[0] * 255),  # ...as a hex string...
                                      int(col[1] * 255), int(col[2] * 255))
            out.append((str(name), hexc))
        return out

    def close(self):
        try:
            self.renderer.delete()                 # Free the GL renderer.
        except Exception:
            pass


def render_arm_2d(phase, stopped, w=ARM_W, h=ARM_H):
    """Dependency-free 2-link arm - the fallback when 3D is unavailable."""
    canvas = np.full((h, w, 3), 18, dtype=np.uint8)  # Dark blank canvas.
    base = (w // 2, h - 70)                        # Arm base (bottom-centre).
    L1, L2 = 150, 120                              # Link lengths.
    a1 = -math.pi / 2 + 0.6 * math.sin(phase)      # Shoulder angle (sways around "up").
    a2 = -0.8 + 0.7 * math.sin(phase * 1.3)        # Elbow angle (different rate).
    bx, by = base                                  # Base coords.
    jx = int(bx + L1 * math.cos(a1)); jy = int(by + L1 * math.sin(a1))  # Elbow position.
    ex = int(jx + L2 * math.cos(a1 + a2)); ey = int(jy + L2 * math.sin(a1 + a2))  # Hand position.
    col = BGR_DANGER if stopped else BGR_SAFE      # Red when stopped, green when moving.
    cv2.circle(canvas, base, 16, BGR_WHITE, -1, cv2.LINE_AA)        # Base joint.
    cv2.line(canvas, base, (jx, jy), col, 11, cv2.LINE_AA)         # Upper arm.
    cv2.circle(canvas, (jx, jy), 10, BGR_WHITE, -1, cv2.LINE_AA)   # Elbow joint.
    cv2.line(canvas, (jx, jy), (ex, ey), col, 9, cv2.LINE_AA)      # Forearm.
    cv2.circle(canvas, (ex, ey), 12, BGR_CYAN, -1, cv2.LINE_AA)    # Gripper (fill).
    cv2.circle(canvas, (ex, ey), 12, col, 2, cv2.LINE_AA)         # Gripper (outline).
    return canvas


def _overlay_estop(canvas):
    """Stamp a bold red EMERGENCY-STOP wash, border and label over an arm frame
    so a stop is unmistakable in BOTH the 3D and the 2D renderers (the 3D render
    on its own gives no visual cue that the arm has frozen)."""
    h, w = canvas.shape[:2]                        # Canvas size.
    wash = np.empty_like(canvas); wash[:] = BGR_DANGER  # A solid-red layer.
    cv2.addWeighted(wash, 0.30, canvas, 0.70, 0, canvas)  # Blend 30% red over the frame.
    cv2.rectangle(canvas, (2, 2), (w - 3, h - 3), BGR_DANGER, 8, cv2.LINE_AA)  # Thick red border.
    txt = "EMERGENCY STOP"                         # Banner text.
    scale = max(0.8, w / 700.0)                    # Scale text to the canvas width.
    (tw, th), _ = cv2.getTextSize(txt, FONT, scale, 3)  # Measure it.
    x = (w - tw) // 2; y = th + 34                 # Centre it horizontally near the top.
    cv2.rectangle(canvas, (x - 16, y - th - 16), (x + tw + 16, y + 16),  # Dark chip behind the text.
                  BGR_DARK, -1, cv2.LINE_AA)
    cv2.putText(canvas, txt, (x, y), FONT, scale, BGR_WHITE, 3, cv2.LINE_AA)  # The text.
    return canvas


# ════════════════════════════════════════════════════════════════════════════
#  WORKER THREADS   (verified core)
# ════════════════════════════════════════════════════════════════════════════

def bgr_to_qimage(frame_bgr) -> QImage:
    """OpenCV BGR -> deep-copied QImage (safe cross-thread handoff)."""
    rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))  # BGR→RGB, contiguous memory.
    h, w, ch = rgb.shape                           # Image dimensions + channels.
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()  # Wrap + .copy() (own the bytes).


class VisionWorker(QThread):                        # Runs the whole CV pipeline off the GUI thread.
    frameReady = Signal(QImage, object)             # Emitted per frame: (image, FrameResult).
    status     = Signal(str)                        # Status/log messages.
    failed     = Signal(str)                        # Fatal error message.

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False                       # Loop control flag.
        self.cam_index = CAM_INDEX                   # Camera to open.
        self.conf = YOLO_CONF                        # Live YOLO confidence.
        self.threshold = AREA_THRESHOLD              # Live proximity threshold.
        self.draw_landmarks = True                   # Draw hand skeletons?
        self.flip = FLIP_VIEW                        # Mirror the feed?
        self.device = "cuda"                         # Requested compute device.
        self.recognize_faces = True                  # Face ID on?
        self.detect_objects = True                   # Object detection on?
        self._last_frame = None                      # Last annotated frame (for snapshots).

    def set_conf(self, v):       self.conf = float(v)        # Setters called from the GUI thread...
    def set_threshold(self, v):  self.threshold = int(v)     # ...(simple attribute writes are atomic enough).
    def set_landmarks(self, v):  self.draw_landmarks = bool(v)
    def set_flip(self, v):       self.flip = bool(v)
    def set_face(self, v):       self.recognize_faces = bool(v)
    def set_objects(self, v):    self.detect_objects = bool(v)
    def stop(self):              self._running = False       # Ask the loop to exit.

    def run(self):                                  # Thread body (runs when .start() is called).
        try:
            import torch                            # Used only to check CUDA availability.
            if self.device == "cuda" and not torch.cuda.is_available():  # Asked for GPU but none...
                self.device = "cpu"                 # ...fall back to CPU.
        except Exception:
            self.device = "cpu"                     # No torch → CPU.

        self.status.emit(f"Loading YOLOv8 + MediaPipe  ({self.device.upper()})...")  # Notify the UI.
        try:
            pipe = VisionPipeline(YOLO_MODEL_PATH, self.device)  # Build the models (heavy).
        except Exception as e:
            self.failed.emit(f"Model load failed: {e}")  # Report and abort on failure.
            return

        if pipe.face.available:                     # Tell the UI whether face ID came up.
            self.status.emit(f"Face DB loaded: {len(pipe.face.names)} identities")
        else:
            self.status.emit(f"Face recognition off ({pipe.face.error or 'no DB'})")

        self.status.emit("Opening camera...")
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)  # Open camera (DirectShow on Windows).
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)         # Request width.
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)       # Request height.
        if not cap.isOpened():                      # If DirectShow failed...
            cap = cv2.VideoCapture(self.cam_index)  # ...retry with the default backend.
        if not cap.isOpened():                      # Still no camera...
            self.failed.emit(f"Cannot open camera index {self.cam_index}")  # ...report...
            pipe.close(); return                    # ...clean up and abort.

        self.status.emit("Live")                    # We're live.
        self._running = True                        # Enter the loop.
        prev = time.perf_counter()                  # Seed FPS timer.
        while self._running:                        # Main capture loop.
            ok, frame = cap.read()                  # Grab a frame.
            if not ok:                              # Dropped frame...
                self.status.emit("Camera frame dropped"); continue  # ...skip it.
            frame, result = pipe.process(           # Run the full pipeline on it.
                frame, self.conf, self.threshold, self.draw_landmarks, self.flip,
                self.recognize_faces, self.detect_objects)
            now = time.perf_counter()               # Time now...
            result.fps = 1.0 / max(1e-6, now - prev); prev = now  # ...compute + store FPS.
            self._last_frame = frame                # Keep for snapshots.
            self.frameReady.emit(bgr_to_qimage(frame), result)  # Hand the frame to the GUI thread.

        cap.release(); pipe.close()                 # Loop ended: release camera + models.
        self.status.emit("Stopped")

    def snapshot_path(self):
        if self._last_frame is None:                # Nothing captured yet...
            return None
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)    # Ensure the folder exists.
        path = os.path.join(SNAPSHOT_DIR, f"cobot_{int(time.time())}.png")  # Timestamped filename.
        cv2.imwrite(path, self._last_frame)         # Write the PNG.
        return path                                 # Return where it went.


class ArmWorker(QThread):                            # Renders the 3-D arm off the GUI thread.
    armReady = Signal(QImage)                        # Emitted per arm frame.
    mode     = Signal(str)                           # "3D URDF" or "2D fallback".
    legend   = Signal(list)        # [(link_name, '#rrggbb'), ...]   # Colour key for the UI.

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False                        # Loop flag.
        self._stopped = False                        # Whether the arm is frozen (e-stop / red zone).
        self._phase = 0.0                            # Sweep phase accumulator.

    def set_stopped(self, v):  self._stopped = bool(v)   # Freeze/unfreeze the arm.
    def stop(self):            self._running = False     # Ask the loop to exit.

    def run(self):
        arm = None                                   # The 3-D renderer (or None → 2-D fallback).
        try:
            arm = RobotArm3D(URDF_PATH, ARM_W, ARM_H)  # Try to load the 3-D model.
            self.mode.emit("3D URDF")                # Tell the UI we're in 3-D.
            self.legend.emit(arm.legend_entries())   # Send the colour key.
        except Exception:
            arm = None                               # Failed → use the 2-D fallback.
            self.mode.emit("2D fallback")

        self._running = True
        while self._running:                         # Render loop.
            if not self._stopped:                    # Only advance the sweep while moving.
                self._phase += ARM_SPEED
            try:
                canvas = (arm.render(self._phase, self._stopped) if arm  # 3-D if available...
                          else render_arm_2d(self._phase, self._stopped))  # ...else 2-D.
            except Exception:
                arm = None                           # If 3-D rendering blows up mid-run...
                self.mode.emit("2D fallback")        # ...switch to 2-D permanently.
                canvas = render_arm_2d(self._phase, self._stopped)
            if self._stopped:                        # If frozen...
                canvas = _overlay_estop(canvas)      # ...stamp the EMERGENCY STOP overlay.
            self.armReady.emit(bgr_to_qimage(canvas))  # Send the frame to the GUI.
            self.msleep(33)                          # ~30 FPS cap.

        if arm is not None:                          # Loop ended...
            arm.close()                              # ...free the renderer.


# ════════════════════════════════════════════════════════════════════════════
#  CUSTOM WIDGETS
# ════════════════════════════════════════════════════════════════════════════

class VideoView(QWidget):
    """Aspect-preserving image surface with a rounded frame + placeholder.

    Right-clicking the surface emits `rightClicked` (used as a manual emergency
    stop); when latched it paints a red alert border + banner over the feed.
    """

    rightClicked = Signal()                          # Emitted on right-click.

    def __init__(self, placeholder="NO SIGNAL", parent=None, lock_aspect=True):
        super().__init__(parent)
        self._img = None                             # Current image (or None → show placeholder).
        self._lock_aspect = lock_aspect              # If True, the widget hugs the feed's aspect ratio.
        self._placeholder = placeholder              # Text shown when there's no image.
        self._estop = False                          # Whether the red e-stop overlay is latched.
        self._flashing = False                       # Whether the red-zone flash is active.
        self._flash_on = False                       # Blink phase (on/off) of the flash.
        self._flash_blink = QTimer(self)             # Timer that toggles the blink...
        self._flash_blink.setInterval(220)           # ...every 220 ms.
        self._flash_blink.timeout.connect(self._toggle_flash)
        self._flash_stop = QTimer(self)              # One-shot timer that ends the flash...
        self._flash_stop.setSingleShot(True)
        self._flash_stop.timeout.connect(self.stop_flash)
        self.setMinimumSize(320, 240)                # Don't collapse too small.
        # When lock_aspect is on, hug the feed's aspect ratio (no stretch / crop /
        # letterbox); the camera uses this. When off, just fill the box -- the 3D
        # arm uses this so it can grow taller than its native render ratio.
        self._aspect = FRAME_WIDTH / FRAME_HEIGHT    # Default ratio; updated from real frames.
        if lock_aspect:
            sp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            sp.setHeightForWidth(True)               # Height tracks width at the feed ratio.
            self.setSizePolicy(sp)
        else:
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Fill available space.
        self.setContextMenuPolicy(Qt.PreventContextMenu)  # Right-click is ours, not a context menu.

    def hasHeightForWidth(self):
        return self._lock_aspect                     # Only constrain height when locking aspect.

    def heightForWidth(self, w):
        return int(round(w / self._aspect)) if self._aspect > 0 else w  # Box height for a given width.

    def set_image(self, img):
        self._img = img                              # Show a new frame.
        # Track the feed's true aspect ratio so the container hugs it exactly.
        if (self._lock_aspect and img is not None and not img.isNull()
                and img.height() > 0):
            a = img.width() / img.height()
            if abs(a - self._aspect) > 1e-3:
                self._aspect = a
                self.updateGeometry()                # Re-ask the layout for height.
        self.update()
    def set_placeholder(self, t): self._placeholder = t; self._img = None; self.update()  # Show text.
    def clear(self): self._img = None; self.update()                  # Clear the surface.
    def set_estop(self, on): self._estop = bool(on); self.update()    # Toggle the e-stop overlay.

    def _toggle_flash(self): self._flash_on = not self._flash_on; self.update()  # Blink tick.

    def start_flash(self, duration_ms=5000):
        """Blink a red DANGER overlay for `duration_ms`, then stop on its own."""
        self._flashing = True; self._flash_on = True  # Begin flashing (start visible).
        self._flash_blink.start()                     # Start blinking.
        self._flash_stop.start(int(duration_ms))      # Schedule the auto-stop.
        self.update()

    def stop_flash(self):
        self._flash_blink.stop(); self._flash_stop.stop()  # Stop both timers.
        self._flashing = False; self._flash_on = False     # Clear flash state.
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton:             # Right-click...
            self.rightClicked.emit(); e.accept()     # ...emit our signal.
        else:
            super().mousePressEvent(e)               # Other buttons: default handling.

    def paintEvent(self, _):                          # Custom painting of the surface.
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)  # Smooth drawing.
        r = self.rect().adjusted(0, 0, -1, -1)        # Drawable rect (1px inset).
        path = QPainterPath(); path.addRoundedRect(QRectF(r), 10, 10)  # Rounded clip path.
        p.setClipPath(path)                           # Clip everything to the rounded rect.
        p.fillRect(r, qc("#06080f"))                  # Dark backing fill.
        if self._img is not None and not self._img.isNull():  # If we have a frame...
            s = self._img.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)  # Fit it.
            p.drawImage((self.width() - s.width()) // 2,  # Centre it horizontally...
                        (self.height() - s.height()) // 2, s)  # ...and vertically.
        else:                                         # No frame → draw the placeholder text.
            p.setPen(qc(C.FAINT))
            f = QFont("Segoe UI", 11); f.setLetterSpacing(QFont.AbsoluteSpacing, 2)
            p.setFont(f); p.drawText(r, Qt.AlignCenter, self._placeholder)
        p.setClipping(False)                          # Stop clipping for the borders.
        if self._estop:                               # If e-stop is latched...
            p.setPen(QPen(qc(C.RED), 3)); p.setBrush(Qt.NoBrush); p.drawPath(path)  # Red border.
            p.setClipPath(path)
            band = QRectF(r.left(), r.top(), r.width(), 34)  # Top banner area.
            p.fillRect(band, qc(C.RED, 220))          # Filled red banner.
            p.setPen(qc("#ffffff"))
            f = QFont("Segoe UI", 12, QFont.Black)
            f.setLetterSpacing(QFont.AbsoluteSpacing, 1)
            p.setFont(f)
            p.drawText(band, Qt.AlignCenter,          # Banner text.
                       "EMERGENCY STOP  -  right-click to release")
            p.setClipping(False)
        else:                                         # Normal state → subtle border.
            p.setPen(QPen(qc(C.BORDER), 1)); p.setBrush(Qt.NoBrush); p.drawPath(path)
        if self._flashing and self._flash_on:         # During the red-zone flash (on phase)...
            p.setClipPath(path)
            p.fillRect(r, qc(C.RED, 95))              # Translucent red wash.
            p.setPen(QPen(qc(C.RED), 6)); p.setBrush(Qt.NoBrush); p.drawPath(path)  # Bold red border.
            band = QRectF(r.left(), r.bottom() - 33, r.width(), 33)  # Bottom banner.
            p.fillRect(band, qc(C.RED, 235))
            p.setPen(qc("#ffffff"))
            f = QFont("Segoe UI", 12, QFont.Black)
            f.setLetterSpacing(QFont.AbsoluteSpacing, 1)
            p.setFont(f)
            p.drawText(band, Qt.AlignCenter, "!  RED ZONE  -  DANGER  !")  # Warning text.
            p.setClipping(False)


class StatusPill(QWidget):
    """Big animated SAFE / STOP indicator with a pulsing halo."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(312, 60)                    # Fixed pill size.
        self._safe = True                             # Current safe state.
        self._estop = False                           # E-stop latch.
        self._pulse = 0.0                             # Animation phase for the halo.
        t = QTimer(self); t.timeout.connect(self._tick); t.start(40)  # ~25 FPS animation.

    def set_safe(self, safe): self._safe = safe; self.update()        # Update safe state.
    def set_estop(self, on): self._estop = bool(on); self.update()    # Update e-stop.
    def _tick(self): self._pulse = (self._pulse + 0.08) % (2 * math.pi); self.update()  # Advance pulse.

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)  # Inset rect.
        base = qc(C.SAFE if (self._safe and not self._estop) else C.DANGER)  # Colour by state.
        bg = QColor(base); bg.setAlpha(45)            # Translucent fill of that colour.
        p.setBrush(bg); p.setPen(QPen(base, 1.6)); p.drawRoundedRect(r, 15, 15)  # Pill body.
        glow = 0.5 + 0.5 * math.sin(self._pulse)      # 0..1 pulsing value.
        cx, cy = 28, self.height() / 2                # Halo centre (left side).
        halo = QColor(base); halo.setAlpha(int(80 * glow))  # Halo alpha pulses.
        p.setPen(Qt.NoPen); p.setBrush(halo)
        p.drawEllipse(QPointF(cx, cy), 14, 14)        # Outer pulsing halo.
        p.setBrush(base); p.drawEllipse(QPointF(cx, cy), 6.5, 6.5)  # Solid inner dot.
        p.setPen(base)
        f = QFont("Segoe UI", 18, QFont.Black); f.setLetterSpacing(QFont.AbsoluteSpacing, 1.5)
        p.setFont(f)
        text = ("EMERGENCY STOP" if self._estop else  # Label depends on state.
                "SYSTEM RUNNING" if self._safe else "SYSTEM STOP")
        p.drawText(QRectF(50, 0, self.width() - 56, self.height()),  # Draw the label.
                   Qt.AlignVCenter | Qt.AlignLeft, text)


class RingGauge(QFrame):
    """Colour-coded circular gauge: title on top, big value in the ring."""

    def __init__(self, title, unit="", color=C.BLUE, vmax=100.0,
                 dynamic=False, parent=None):
        super().__init__(parent)
        self.setObjectName("card")                    # Use the "card" style.
        self._title = title; self._unit = unit; self._color = color  # Labels + colour.
        self._vmax = max(float(vmax), 1e-6); self._dynamic = dynamic  # Full-scale value + dynamic colour.
        self._val = 0.0                               # Current value.
        self.setMinimumSize(150, 132)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_value(self, v): self._val = float(v); self.update()  # Update + repaint.

    def _col(self):
        if self._dynamic:                             # Dynamic gauge: colour by value...
            return qc(C.GREEN if self._val < 60 else  # ...green < 60...
                      C.AMBER if self._val < 100 else C.RED)  # ...amber < 100, else red.
        return qc(self._color)                        # Static gauge: fixed colour.

    def paintEvent(self, e):
        super().paintEvent(e)                         # Draw the card background first.
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        p.setPen(qc(C.MUTED)); p.setFont(QFont("Segoe UI", 8, QFont.Bold))
        p.drawText(QRectF(0, 11, W, 14), Qt.AlignHCenter, self._title.upper())  # Title.

        tw = 11                                       # Ring thickness.
        d = max(min(W - 46, H - 60), 36)              # Ring diameter (fits the card).
        cx, cy = W / 2.0, 30 + d / 2.0                # Ring centre.
        rect = QRectF(cx - d / 2, cy - d / 2, d, d)   # Ring bounding rect.
        p.setPen(QPen(qc(C.TRACK), tw)); p.drawArc(rect, 0, 360 * 16)  # Full grey track (angles in 1/16°).
        col = self._col()                             # Value colour.
        frac = max(0.0, min(self._val / self._vmax, 1.0))  # Fraction filled (0..1).
        pen = QPen(col, tw); pen.setCapStyle(Qt.RoundCap); p.setPen(pen)  # Rounded coloured arc.
        p.drawArc(rect, 90 * 16, -int(frac * 360 * 16))  # Filled arc, starting at top, going clockwise.

        p.setPen(col); p.setFont(QFont("Segoe UI", 21, QFont.Bold))
        p.drawText(rect, Qt.AlignCenter, f"{self._val:.0f}")  # Big numeric value.
        if self._unit:                                # Optional unit under the value.
            p.setPen(qc(C.MUTED)); p.setFont(QFont("Segoe UI", 8))
            p.drawText(QRectF(rect.left(), cy + d * 0.18, d, 13),
                       Qt.AlignHCenter, self._unit)


class StatChip(QFrame):
    """Small tinted pill: a coloured dot, a LABEL and a value."""

    def __init__(self, label, color=C.BLUE, parent=None):
        super().__init__(parent)
        self._label = label; self._color = color; self._val = "--"  # Label, colour, current value.
        self.setMinimumHeight(38)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set(self, val, color=None):
        self._val = str(val)                          # Update the value text.
        if color: self._color = color                 # Optionally recolour.
        self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        base = qc(self._color); bg = QColor(base); bg.setAlpha(38)  # Tinted background.
        p.setBrush(bg); p.setPen(QPen(base, 1.2)); p.drawRoundedRect(r, 9, 9)  # Pill body.
        cy = r.center().y()
        p.setPen(Qt.NoPen); p.setBrush(base)
        p.drawEllipse(QPointF(r.left() + 14, cy), 5, 5)  # Coloured status dot.
        p.setPen(qc(C.MUTED)); p.setFont(QFont("Segoe UI", 8, QFont.Bold))
        p.drawText(QRectF(r.left() + 25, r.top(), r.width() - 62, r.height()),  # Label (left).
                   Qt.AlignVCenter | Qt.AlignLeft, self._label.upper())
        p.setPen(base); p.setFont(QFont("Segoe UI", 12, QFont.Bold))
        p.drawText(QRectF(r.left(), r.top(), r.width() - 13, r.height()),  # Value (right).
                   Qt.AlignVCenter | Qt.AlignRight, self._val)


class LinkLegend(QWidget):
    """Colour key for the 3D arm, painted directly: a big swatch + link name per
    link, flowing left-to-right and wrapping to new rows as needed."""

    SW   = 16          # swatch size
    PADX = 12                                         # Horizontal padding.
    PADY = 3                                          # Vertical padding.
    GAPX = 22          # gap between entries
    ROWH = 22          # row height

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries = []                            # List of (name, hexcolour).
        # vertical size is Fixed -> no blank space; height tracks the row count
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._apply_height(1)                         # Start one row tall.

    def _apply_height(self, rows):
        h = rows * self.ROWH + 2 * self.PADY          # Height = rows × rowH + padding.
        self.setFixedHeight(h)                        # Lock the widget height.

    def set_entries(self, entries):
        self._entries = list(entries)                 # Store the colour key.
        rows = self._layout_rows(max(self.width(), 600))[1]  # Compute how many rows it needs.
        self._apply_height(rows)                      # Resize to fit.
        self.update()

    def _layout_rows(self, width):
        """Return list of (x, name, hexc) plus total rows for a given width."""
        f = QFont("Segoe UI", 11, QFont.Bold)         # Font for the labels.
        fm = self.fontMetrics() if not self._entries else None  # (kept from original; recomputed below)
        from PySide6.QtGui import QFontMetrics
        fm = QFontMetrics(f)                          # Measure text widths with this font.
        placed, x, row = [], self.PADX, 0             # Placement list, running X, current row.
        for name, hexc in self._entries:              # Lay out each entry...
            tw = fm.horizontalAdvance(str(name))      # Width of the name text.
            step = self.SW + 8 + tw + self.GAPX       # Total width this entry consumes.
            if x + step > width - self.PADX and x > self.PADX:  # If it won't fit on this row...
                row += 1; x = self.PADX               # ...wrap to the next row.
            placed.append((x, row, name, hexc, tw))   # Record where it goes.
            x += step                                 # Advance X.
        return placed, row + 1, f                     # Placements, row count, font.

    def resizeEvent(self, e):
        if self._entries:                             # On resize, recompute the row count...
            self._apply_height(self._layout_rows(self.width())[1])
        super().resizeEvent(e)

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        if not self._entries:                         # No legend yet...
            p.setPen(qc(C.MUTED)); p.setFont(QFont("Segoe UI", 10))
            p.drawText(self.rect(), Qt.AlignCenter, "waiting for arm…")  # ...show a placeholder.
            return
        placed, rows, font = self._layout_rows(self.width())  # Compute placements.
        p.setFont(font)
        for x, row, name, hexc, tw in placed:         # Draw each entry...
            y = self.PADY + row * self.ROWH           # Row Y.
            col = QColor(hexc)
            p.setBrush(col); p.setPen(QPen(QColor("#ffffff"), 2))
            p.drawRoundedRect(QRectF(x, y, self.SW, self.SW), 5, 5)  # Colour swatch.
            p.setPen(qc(C.TEXT))
            p.drawText(QRectF(x + self.SW + 8, y - 4, tw + 6, self.SW + 8),  # Link name beside it.
                       Qt.AlignVCenter | Qt.AlignLeft, str(name))


class Sparkline(QFrame):
    """Live time-series sparkline drawn with QPainter."""

    def __init__(self, title="FPS", color=C.BLUE, maxlen=120, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setMinimumHeight(96)
        self._title = title; self._color = color      # Title + line colour.
        self._data = deque(maxlen=maxlen)             # Ring buffer of recent values.

    def push(self, v): self._data.append(float(v)); self.update()  # Add a sample + repaint.

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        m = 12                                        # Margin.
        area = self.rect().adjusted(m, m + 14, -m, -m)  # Plot area (leave room for the title).
        col = qc(self._color)
        p.setPen(qc(C.MUTED)); p.setFont(QFont("Segoe UI", 8, QFont.Bold))
        cur = self._data[-1] if self._data else 0.0   # Latest value.
        p.drawText(QRectF(m, 7, self.width() - 2 * m, 14), Qt.AlignLeft, self._title)  # Title (left).
        p.setPen(col)
        p.drawText(QRectF(m, 7, self.width() - 2 * m, 14), Qt.AlignRight, f"{cur:5.1f}")  # Value (right).
        if len(self._data) < 2:                       # Need ≥2 points to draw a line.
            return
        vmax = max(max(self._data), 1.0); vmin = min(self._data)  # Y-range.
        rng = max(vmax - vmin, 1e-3); n = len(self._data)         # Range + count.
        w = area.width(); h = area.height(); x0 = area.left(); y0 = area.top()  # Plot geometry.
        pts = [QPointF(x0 + w * i / (n - 1), y0 + h - (v - vmin) / rng * h)  # Map samples → points.
               for i, v in enumerate(self._data)]
        fill = QPolygonF(pts + [QPointF(x0 + w, y0 + h), QPointF(x0, y0 + h)])  # Close the area under it.
        grad = QLinearGradient(0, y0, 0, y0 + h)      # Vertical fade for the fill.
        c0 = QColor(col); c0.setAlpha(110); c1 = QColor(col); c1.setAlpha(8)
        grad.setColorAt(0, c0); grad.setColorAt(1, c1)
        p.setPen(Qt.NoPen); p.setBrush(QBrush(grad)); p.drawPolygon(fill)  # Filled gradient area.
        p.setPen(QPen(col, 2)); p.setBrush(Qt.NoBrush); p.drawPolyline(QPolygonF(pts))  # The line.


class SafetyTimeline(QFrame):
    """Horizontal strip of recent GREEN/RED states."""

    def __init__(self, maxlen=200, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setMinimumHeight(70)
        self._data = deque(maxlen=maxlen)             # Ring buffer of safe(True)/unsafe(False).

    def push(self, safe): self._data.append(bool(safe)); self.update()  # Record one state.

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        m = 6
        p.setPen(qc(C.MUTED)); p.setFont(QFont("Segoe UI", 8, QFont.Bold))
        p.drawText(QRectF(m, 4, 240, 14), Qt.AlignLeft, "SAFETY TIMELINE")  # Title.
        if not self._data:
            return
        area = self.rect().adjusted(m, m + 14, -m, -m)
        # Clip the bars to a rounded rect so they stay inside the card border.
        clip = QPainterPath(); clip.addRoundedRect(QRectF(area), 8, 8)
        p.setClipPath(clip)
        n = self._data.maxlen; bw = area.width() / n  # One column per possible sample.
        for i, safe in enumerate(self._data):         # Draw a green/red bar per recorded state.
            p.fillRect(QRectF(area.left() + i * bw, area.top(),
                              max(bw - 0.4, 1), area.height()),
                       qc(C.SAFE if safe else C.DANGER))


class LogoBadge(QWidget):
    """A painted 'vision' emblem (cyan ring + white pupil) - no glyph fonts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(36, 36)                     # Fixed emblem size.

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(3, 3, 30, 30)
        p.setPen(Qt.NoPen)
        p.setBrush(qc("#22d3ee")); p.drawEllipse(r)                    # Cyan outer ring.
        p.setBrush(qc("#1e2a78")); p.drawEllipse(r.adjusted(6, 6, -6, -6))   # Dark iris.
        p.setBrush(qc("#ffffff")); p.drawEllipse(r.adjusted(12, 12, -12, -12))  # White pupil.


def titled_card(title, body, color=C.TEXT):
    """Rounded card with a coloured title bar over a body widget."""
    f = QFrame(); f.setObjectName("card")             # The card frame.
    lay = QVBoxLayout(f); lay.setContentsMargins(12, 10, 12, 12); lay.setSpacing(8)  # Vertical layout.
    t = QLabel(title); t.setObjectName("panelTitle"); t.setStyleSheet(f"color:{color};")  # Title label.
    f.title_label = t                                 # Expose the label so callers can retitle it later.
    lay.addWidget(t); lay.addWidget(body, 1)          # Title on top, body fills the rest.
    return f


# ════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW  -  mission control layout
# ════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Safety Rated Monitor Stop")  # Window title.
        self.resize(1520, 940)                        # Initial size.
        self.setMinimumSize(1240, 820)                # Minimum size.

        self.vision = VisionWorker(self)              # The CV worker thread.
        self.arm = ArmWorker(self)                    # The arm-render worker thread.
        self.audio = AudioAlerter()                   # The voice/alert engine.
        self._t_start = None                          # Monitor start time (for uptime).
        self._estop = False            # manual emergency-stop latch (right-click)
        self._was_red = False          # red-zone edge tracker for voice alerts
        self._last_seen = {}           # known-name -> last-seen time (for greetings)
        self._log_file = None          # open file handle for the current session's log (or None)
        self._log_path = None          # path of the current session log file
        self._last_safe = None         # previous overall safe state (to log only transitions)
        self._last_collision = False   # previous collision-risk state (to log only the onset)

        self._build_ui()                              # Construct all widgets/layout.
        self._connect_workers()                       # Wire worker signals → handlers.
        self._wire_shortcuts()                        # Set up keyboard shortcuts.

        self._clock = QTimer(self); self._clock.timeout.connect(self._update_clock)  # Uptime clock...
        self._clock.start(1000)                       # ...ticks every second.

    # ------------------------------------------------------------------ UI ---
    def _build_ui(self):
        root = QWidget(); root.setObjectName("root"); self.setCentralWidget(root)  # Root container.
        outer = QVBoxLayout(root)                     # Top-level vertical layout.
        outer.setContentsMargins(14, 14, 14, 14); outer.setSpacing(12)

        outer.addWidget(self._build_header())         # Header bar.

        body = QHBoxLayout(); body.setSpacing(12)     # Three-column body.
        body.addWidget(self._build_left_rail(), 0)    # Left: gauges.
        body.addWidget(self._build_center(), 1)       # Centre: camera (stretches).
        body.addWidget(self._build_right_rail(), 0)   # Right: arm + chips.
        outer.addLayout(body, 1)

        self.link_legend = LinkLegend()               # The arm colour key widget.
        self.legend_card = titled_card("ROBOT LINK COLOUR KEY", self.link_legend, C.CYAN)  # Wrapped in a card.
        outer.addWidget(self.legend_card, 0)

        outer.addWidget(self._build_telemetry(), 0)   # Charts strip.
        outer.addWidget(self._build_control_dock(), 0)  # Controls dock.

    def _build_header(self):
        h = QFrame(); h.setObjectName("header"); h.setFixedHeight(76)  # Header frame.
        lay = QHBoxLayout(h); lay.setContentsMargins(22, 10, 18, 10); lay.setSpacing(16)
        logo = LogoBadge()                            # Painted logo.
        tb = QVBoxLayout(); tb.setSpacing(0)
        t1 = QLabel("SAFETY RATED MONITOR"); t1.setObjectName("h1")  # Title text.
        tb.addWidget(t1)
        lay.addWidget(logo); lay.addLayout(tb); lay.addStretch()  # Logo + title, then push right.

        self.lbl_uptime = QLabel("IDLE")              # Uptime label (top-right).
        self.lbl_uptime.setStyleSheet("color:#3a1d7a; font-size:11px; font-weight:700;")
        cb = QVBoxLayout(); cb.setSpacing(0)
        cb.addWidget(self.lbl_uptime, 0, Qt.AlignRight)
        lay.addLayout(cb)
        self.pill = StatusPill(); lay.addWidget(self.pill)  # The big SAFE/STOP pill.
        return h

    def _build_left_rail(self):
        rail = QWidget(); rail.setFixedWidth(244)     # Fixed-width left column.
        lay = QVBoxLayout(rail); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(12)
        lab = QLabel("LIVE TELEMETRY"); lab.setObjectName("section")  # Section heading.
        lay.addWidget(lab)
        self.g_fps = RingGauge("Throughput", "fps", C.BLUE, vmax=60)             # FPS gauge.
        self.g_prox = RingGauge("Proximity", "% of limit", C.GREEN, vmax=100, dynamic=True)  # Proximity.
        self.g_lat = RingGauge("Inference", "ms", C.VIOLET, vmax=80)             # Latency gauge.
        for g in (self.g_fps, self.g_prox, self.g_lat):  # Stack the three gauges.
            lay.addWidget(g, 1)
        return rail

    def _build_center(self):
        self.view_cam = VideoView("CAMERA OFFLINE   -   press START")  # The camera surface.
        return titled_card("LIVE CAMERA   •   YOLOv8 + HAND TRACKING",  # Wrapped in a titled card.
                           self.view_cam, C.CYAN)

    def _build_right_rail(self):
        rail = QWidget(); rail.setFixedWidth(560)     # Fixed-width right column.
        lay = QVBoxLayout(rail); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(12)
        # Don't aspect-lock the arm: let it fill its box so it can grow taller.
        self.view_arm = VideoView("ROBOTIC ARM STANDBY", lock_aspect=False)  # The arm surface.
        self.arm_card = titled_card("ROBOTIC ARM   •   3D URDF MODEL",  # Wrapped in a card.
                                    self.view_arm, C.VIOLET)
        lay.addWidget(self.arm_card, 1)

        chips = QHBoxLayout(); chips.setSpacing(10)   # Row of status chips.
        self.chip_state = StatChip("Arm", C.GREEN)    # Arm state chip.
        self.chip_obj = StatChip("Objects", C.AMBER)  # Object count chip.
        self.chip_hands = StatChip("Hands", C.CYAN)   # Hand count chip.
        for c in (self.chip_state, self.chip_obj, self.chip_hands):
            chips.addWidget(c)
        lay.addLayout(chips, 0)
        return rail

    def _build_telemetry(self):
        strip = QWidget(); strip.setFixedHeight(122)  # Fixed-height charts strip.
        lay = QHBoxLayout(strip); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(12)
        self.spark = Sparkline("THROUGHPUT  (fps)", C.BLUE)        # FPS chart.
        self.spark_lat = Sparkline("INFERENCE LATENCY  (ms)", C.VIOLET)  # Latency chart.
        self.timeline = SafetyTimeline()              # Green/red history strip.
        lay.addWidget(self.spark, 1)
        lay.addWidget(self.spark_lat, 1)
        lay.addWidget(self.timeline, 1)
        return strip

    def _build_control_dock(self):
        dock = QFrame(); dock.setObjectName("dock"); dock.setFixedHeight(116)  # Bottom controls dock.
        lay = QHBoxLayout(dock); lay.setContentsMargins(16, 12, 16, 12); lay.setSpacing(18)

        self.btn_start = QPushButton("START")         # Start/Stop button.
        self.btn_start.setObjectName("primary")       # Green "primary" style.
        self.btn_start.setFixedWidth(150)
        self.btn_start.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.btn_start.clicked.connect(self.toggle_monitor)  # Click → start/stop.
        lay.addWidget(self.btn_start)

        sliders = QVBoxLayout(); sliders.setSpacing(8)
        # YOLO confidence is hidden from the UI (kept internal at the default value)
        self.sld_conf, conf_box = self._slider_row(   # Build a (hidden) confidence slider.
            "YOLO CONFIDENCE", int(YOLO_CONF * 100), 10, 90, self._on_conf)
        conf_box.setVisible(False)                    # Hide it.
        self.sld_thr, thr_box = self._slider_row(     # Safety-distance (area threshold) slider.
            "SAFETY DISTANCE", AREA_THRESHOLD, THRESHOLD_MIN, THRESHOLD_MAX,
            self._on_threshold, step=THRESHOLD_STEP, is_area=True)
        self.chk_audio = QCheckBox("Voice alerts (hello + red zone)")  # Voice toggle.
        self.chk_audio.setChecked(VOICE_ALERTS_DEFAULT)
        self.chk_audio.toggled.connect(self._on_audio_toggled)
        sliders.addWidget(thr_box); sliders.addWidget(self.chk_audio); sliders.addStretch()
        sw = QWidget(); sw.setLayout(sliders); sw.setMinimumWidth(280)
        lay.addWidget(sw, 1)

        opts = QVBoxLayout(); opts.setSpacing(8)      # Column of feature toggles.
        self.chk_landmarks = QCheckBox("Hand skeleton"); self.chk_landmarks.setChecked(True)  # Skeletons.
        self.chk_landmarks.toggled.connect(self.vision.set_landmarks)
        self.chk_flip = QCheckBox("Mirror camera"); self.chk_flip.setChecked(FLIP_VIEW)  # Mirror.
        self.chk_flip.toggled.connect(self.vision.set_flip)
        self.chk_face = QCheckBox("Face ID"); self.chk_face.setChecked(True)  # Face recognition.
        self.chk_face.toggled.connect(self.vision.set_face)
        self.chk_objects = QCheckBox("Object detection"); self.chk_objects.setChecked(True)  # YOLO.
        self.chk_objects.toggled.connect(self._on_objects_toggled)
        opts.addWidget(self.chk_landmarks); opts.addWidget(self.chk_flip)
        opts.addWidget(self.chk_face); opts.addWidget(self.chk_objects)
        ow = QWidget(); ow.setLayout(opts); lay.addWidget(ow)

        right = QVBoxLayout(); right.setSpacing(8)    # Device selector + snapshot.
        self.cmb_device = QComboBox(); self.cmb_device.addItems(["CUDA (GPU)", "CPU"])  # Compute device.
        self.cmb_device.currentIndexChanged.connect(self._on_device)
        self.btn_snapshot = QPushButton("Snapshot")   # Snapshot button.
        self.btn_snapshot.clicked.connect(self.take_snapshot)
        self.btn_snapshot.setEnabled(False)           # Disabled until live.
        right.addWidget(self.cmb_device); right.addWidget(self.btn_snapshot)
        rw = QWidget(); rw.setLayout(right); rw.setFixedWidth(160); lay.addWidget(rw)

        self.log = QPlainTextEdit(); self.log.setReadOnly(True)  # Read-only log panel.
        self.log.setFixedWidth(320)
        lay.addWidget(self.log)
        return dock

    # --------------------------------------------------------- UI helpers ---
    def _slider_row(self, name, val, lo, hi, cb, step=1, is_area=False):
        box = QWidget(); v = QVBoxLayout(box)         # Container + vertical layout.
        v.setContentsMargins(0, 0, 0, 0); v.setSpacing(3)
        head = QHBoxLayout()                          # Header row: name (left) + value (right).
        lbl = QLabel(name); lbl.setStyleSheet(f"color:{C.MUTED}; font-size:10px; font-weight:700; letter-spacing:1px;")
        val_lbl = QLabel(); val_lbl.setStyleSheet(f"color:{C.CYAN}; font-weight:800;")  # Live value label.
        head.addWidget(lbl); head.addStretch(); head.addWidget(val_lbl)
        s = QSlider(Qt.Horizontal); s.setRange(lo, hi); s.setValue(val)  # The slider.
        if step > 1:
            s.setSingleStep(step)                     # Step size for area sliders.

        def fmt(x): return f"{x/1000:.0f}k" if is_area else f"{x}%"  # Format: "40k" or "50%".
        val_lbl.setText(fmt(val))                     # Initial value text.

        def on_change(x): val_lbl.setText(fmt(x)); cb(x)  # On move: update label + call back.
        s.valueChanged.connect(on_change)
        v.addLayout(head); v.addWidget(s)
        return s, box                                 # Return the slider + its container.

    def _wire_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Space), self, self.toggle_monitor)  # Space → start/stop.
        QShortcut(QKeySequence(Qt.Key_Q), self, self.close)               # Q → quit.
        QShortcut(QKeySequence(Qt.Key_Escape), self, self.close)          # Esc → quit.
        QShortcut(QKeySequence(Qt.Key_Plus), self,                        # + → raise threshold.
                  lambda: self.sld_thr.setValue(self.sld_thr.value() + THRESHOLD_STEP))
        QShortcut(QKeySequence(Qt.Key_Minus), self,                       # - → lower threshold.
                  lambda: self.sld_thr.setValue(self.sld_thr.value() - THRESHOLD_STEP))
        QShortcut(QKeySequence(Qt.Key_S), self, self.take_snapshot)       # S → snapshot.

    def _connect_workers(self):
        self.vision.frameReady.connect(self._on_frame)   # New frame → update UI.
        self.vision.status.connect(self._on_status)      # Status text → log/placeholder.
        self.vision.failed.connect(self._on_failed)      # Fatal error → log + stop.
        self.arm.armReady.connect(self._on_arm)          # New arm frame → show it.
        self.arm.mode.connect(self._on_arm_mode)         # Arm renderer mode → retitle.
        self.arm.legend.connect(self.link_legend.set_entries)  # Colour key → legend widget.
        # Right-click either feed = manual emergency stop.
        self.view_cam.rightClicked.connect(self._toggle_estop)
        self.view_arm.rightClicked.connect(self._toggle_estop)
        tip = "Right-click to toggle EMERGENCY STOP"
        self.view_cam.setToolTip(tip); self.view_arm.setToolTip(tip)  # Tooltips on both feeds.

    # --------------------------------------------------------- callbacks ---
    def toggle_monitor(self):
        self._stop_monitor() if self.vision.isRunning() else self._start_monitor()  # Toggle running state.

    def _restyle(self, btn, name):
        btn.setObjectName(name)                       # Change the QSS object name...
        self.style().unpolish(btn); self.style().polish(btn)  # ...and force a restyle.

    def _start_monitor(self):
        self._t_start = time.time()                   # Record start time.
        self._last_safe = None                        # Reset safety-transition tracker (no prior state yet).
        self._last_collision = False                  # Reset collision-risk tracker.
        self._open_session_log()                      # Open a fresh timestamped log file for this session.
        self.btn_start.setText("STOP"); self._restyle(self.btn_start, "danger")  # Button → red STOP.
        self.btn_snapshot.setEnabled(True)            # Enable snapshots.
        self.view_cam.set_placeholder("INITIALISING  •  loading YOLOv8 + MediaPipe ...")
        self.view_arm.set_placeholder("LOADING ROBOTIC ARM ...")
        self._log("Starting subsystems...")
        self._reset_estop()                           # Clear any latched e-stop.
        self._was_red = False                         # Reset red-zone edge.
        self._last_seen.clear()                       # Reset greeting memory.
        self._log(f"Voice alerts: {self.audio.backend_name()} "
                  f"({'ON' if self.chk_audio.isChecked() else 'OFF'})")
        self.vision.start(); self.arm.start()         # Launch both worker threads.

    def _stop_monitor(self):
        self._log("Stopping subsystems...")
        self.vision.stop(); self.vision.wait(2000)    # Stop CV worker, wait up to 2 s.
        self.arm.stop(); self.arm.wait(2000)          # Stop arm worker, wait up to 2 s.
        self.btn_start.setText("START"); self._restyle(self.btn_start, "primary")  # Button → green START.
        self.btn_snapshot.setEnabled(False)           # Disable snapshots.
        self.view_cam.clear(); self.view_arm.clear()  # Clear both surfaces.
        self.audio.stop_buzzer(); self._was_red = False  # Silence buzzer; reset edge.
        self.view_arm.stop_flash()                    # Stop any flashing.
        self._last_seen.clear()                       # Reset greeting memory.
        self._reset_estop()                           # Clear e-stop.
        self.pill.set_safe(True)                      # Pill back to safe.
        self.chip_state.set("--", C.GREEN)            # Reset chips.
        self._close_session_log()                     # Write the session-end footer and close the log file.

    def _on_frame(self, img, r: FrameResult):
        self.view_cam.set_image(img)                  # Show the new frame.
        self._maybe_voice(r)                          # Maybe fire red-zone alert.
        self._maybe_greet(r)                          # Maybe greet known people.
        # The manual emergency-stop latch overrides the proximity state.
        safe = (r.state == "GREEN") and not self._estop  # Overall safe state.
        self.pill.set_safe(safe); self.arm.set_stopped(not safe)  # Update pill + freeze arm if unsafe.

        # Log a line ONLY when the safety state actually changes (not every frame),
        # so the session log reads as a clean timeline of the safety monitor.
        if safe != self._last_safe:                   # State flipped since last frame...
            if self._last_safe is not None:           # ...(skip the very first frame, which has no prior).
                if safe:                              # Became safe again...
                    self._log("SAFETY  ->  SAFE (green zone, arm resumed)")
                elif self._estop:                     # Unsafe because of the manual latch...
                    self._log("SAFETY  ->  EMERGENCY STOP (manual latch)")
                else:                                 # Unsafe because a hand is too close...
                    self._log("SAFETY  ->  STOP (red zone - hand too close, arm frozen)")
            self._last_safe = safe                    # Remember for the next frame.
        if r.collision and not self._last_collision:  # Collision risk just began (rising edge)...
            self._log(f"SAFETY  ->  COLLISION RISK ({r.objects} object(s) near a stopped arm)")
        self._last_collision = r.collision            # Remember collision state.

        self.g_fps.set_value(r.fps)                   # FPS gauge.
        self.g_lat.set_value(r.infer_ms)              # Latency gauge.
        prox = min(999, int(r.max_area / max(self.vision.threshold, 1) * 100))  # Proximity % of limit.
        self.g_prox.set_value(prox)                   # Proximity gauge.

        self.chip_state.set("RUN" if safe else ("ESTOP" if self._estop else "HOLD"),  # Arm-state chip.
                            C.GREEN if safe else C.RED)
        self.chip_obj.set(f"{r.objects} !" if r.collision else r.objects,  # Object chip (! on collision).
                          C.RED if r.collision else C.AMBER)
        self.chip_hands.set(r.hands, C.CYAN)          # Hand-count chip.

        self.spark.push(r.fps); self.spark_lat.push(r.infer_ms)  # Feed the charts.
        self.timeline.push(safe)                      # Feed the safety timeline.

    def _on_arm(self, img): self.view_arm.set_image(img)  # Show the latest arm frame.

    def _on_arm_mode(self, mode):
        label = ("ROBOTIC ARM   •   3D URDF MODEL" if mode == "3D URDF"  # Title by renderer mode.
                 else "ROBOTIC ARM   •   2D KINEMATIC (3D unavailable)")
        self.arm_card.title_label.setText(label)      # Retitle the arm card.
        self._log(f"Robotic arm renderer: {mode}")

    def _on_status(self, s):
        self._log(s)                                  # Log the status line.
        if any(k in s for k in ("Loading", "Opening", "dropped")):  # During load/issues...
            self.view_cam.set_placeholder(s.upper())  # ...show it on the camera surface.

    def _on_failed(self, msg):
        self._log("ERROR  " + msg); self._stop_monitor()  # Log error + stop everything.

    def _on_conf(self, v):      self.vision.set_conf(v / 100.0)  # Slider % → 0..1 confidence.
    def _on_threshold(self, v): self.vision.set_threshold(v)     # Apply new area threshold.

    def _on_objects_toggled(self, on):
        self.vision.set_objects(on)                   # Toggle YOLO in the worker.
        self._log(f"Object detection {'enabled' if on else 'disabled'}")

    # ---------------------------------------------- safety alerts / e-stop ---
    def _maybe_voice(self, r: FrameResult):
        """On *entering* the red zone, fire the alert: flash the camera feed red
        and sound the voice + buzzer (capped at RED_ALERT_SECS). As soon as the
        hand leaves the red zone, the buzzer and flashing are stopped early."""
        in_red = (r.state == "RED")                   # Is a hand too close this frame?
        if in_red and not self._was_red:              # Rising edge → just entered the red zone.
            self.audio.alert_red_zone(RED_ALERT_SECS) # Fire the voice + buzzer alert.
            self.view_arm.start_flash(int(RED_ALERT_SECS * 1000))  # Flash the arm feed red.
        elif not in_red and self._was_red:            # Falling edge → hand just left the red zone.
            self.audio.stop_buzzer()                  # Silence the buzzer immediately.
            self.view_arm.stop_flash()                # Stop the red flashing immediately.
        self._was_red = in_red                        # Remember for next frame.

    def _maybe_greet(self, r: FrameResult):
        """Say "hello <name>" when a known person appears. The same person is not
        greeted again until they've been out of frame for GREET_REENTRY_SECS, so
        brief recognition drop-outs don't cause repeated greetings.

        Emergency stop and the red-zone alert take priority: while either is
        active we don't greet, so the safety alert is never talked over."""
        if not r.identities or self._estop or self.audio.is_alerting():  # Nothing to greet / safety active.
            return
        now = time.monotonic()
        entered = []                                  # Newly-appeared known people.
        for name in r.identities:
            last = self._last_seen.get(name)          # When last seen.
            if last is None or (now - last) > GREET_REENTRY_SECS:  # New or back after cooldown.
                entered.append(name)
            self._last_seen[name] = now               # Update last-seen time.
        if entered:
            # One "Hello <name>" per newly-seen person, spoken as a single phrase
            # so none get dropped from the (collapsing) audio queue.
            self.audio.say(". ".join(f"Hello {name}" for name in entered))  # Speak the greeting(s).
            self._log("Greeted: " + ", ".join(entered))

    def _toggle_estop(self):
        """Mouse right-click on a feed toggles the emergency-stop latch."""
        self._set_estop(not self._estop)              # Flip the latch.

    def _set_estop(self, on: bool):
        self._estop = bool(on)                        # Store the latch state.
        self.view_cam.set_estop(self._estop)          # Update the camera overlay.
        self.pill.set_estop(self._estop)              # Update the status pill.
        if self._estop:                               # Engaging...
            self.arm.set_stopped(True)                # ...freeze the arm...
            self.pill.set_safe(False)                 # ...pill unsafe...
            self.chip_state.set("ESTOP", C.RED)       # ...chip shows ESTOP...
            self._log("EMERGENCY STOP engaged  (mouse right-click)")
            self.audio.say(ESTOP_ON_PHRASE, priority=True)  # ...announce (high priority).
        else:                                         # Releasing...
            self._log("Emergency stop released  (mouse right-click)")
            self.audio.say(ESTOP_OFF_PHRASE, priority=True)  # ...announce release...
            if not self.vision.isRunning():           # ...and if not live...
                self.pill.set_safe(True)              # ...reset the pill...
                self.chip_state.set("--", C.GREEN)    # ...and the chip.

    def _reset_estop(self):
        self._estop = False                           # Clear the latch...
        self.view_cam.set_estop(False)                # ...and its overlays.
        self.pill.set_estop(False)

    def _on_audio_toggled(self, on: bool):
        self.audio.set_enabled(on)                    # Enable/disable voice.
        self._log(f"Voice alerts {'enabled' if on else 'disabled'} "
                  f"({self.audio.backend_name()})")

    def _on_device(self, idx):
        self.vision.device = "cuda" if idx == 0 else "cpu"  # Index 0 = GPU, 1 = CPU.
        self._log(f"Compute target -> {self.vision.device.upper()} (applies on next START)")

    def take_snapshot(self):
        path = self.vision.snapshot_path()            # Ask the worker to save the last frame.
        self._log(f"Snapshot saved  ->  {path}" if path  # Log where it went...
                  else "Snapshot unavailable (no live frame)")  # ...or why it failed.

    def _update_clock(self):
        if self._t_start and self.vision.isRunning(): # While live...
            el = int(time.time() - self._t_start)     # ...elapsed seconds...
            self.lbl_uptime.setText(f"UP {el//60:02d}:{el%60:02d}")  # ...shown as UP MM:SS.
        else:
            self.lbl_uptime.setText("IDLE")           # Otherwise idle.

    def _open_session_log(self):
        """Open a new timestamped log file for this monitoring session.

        One file per START press, named logs/session_YYYYMMDD_HHMMSS.log. Every
        line that goes to the on-screen log panel is also appended here, so each
        session leaves a permanent, time-stamped record of the safety monitor."""
        self._close_session_log()                     # Close any previous session's file first.
        try:
            os.makedirs(LOG_DIR, exist_ok=True)       # Make sure the logs/ folder exists.
            stamp = time.strftime("%Y%m%d_%H%M%S")    # Session-start timestamp for the filename.
            self._log_path = os.path.join(LOG_DIR, f"session_{stamp}.log")  # e.g. logs/session_20260619_145601.log
            self._log_file = open(self._log_path, "a", encoding="utf-8")    # Append mode, UTF-8.
            self._log_file.write(                     # Write a clear session-start banner.
                f"===== SESSION START  {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            self._log_file.flush()                    # Flush so the header hits disk immediately.
        except Exception:
            self._log_file = None                     # On any I/O error, just disable file logging.
        if self._log_file is not None:                # If the file opened...
            self._log(f"Session log -> {self._log_path}")  # ...note its path (also written into it).

    def _close_session_log(self):
        """Write the session-end footer (with duration) and close the log file."""
        if self._log_file is not None:                # Only if a file is open...
            try:
                el = int(time.time() - self._t_start) if self._t_start else 0  # Session duration (s).
                self._log_file.write(                 # Footer with end time + MM:SS duration.
                    f"===== SESSION END    {time.strftime('%Y-%m-%d %H:%M:%S')}"
                    f"  (duration {el//60:02d}:{el%60:02d}) =====\n\n")
                self._log_file.flush(); self._log_file.close()  # Flush + close.
            except Exception:
                pass                                  # Ignore errors during close.
            self._log_file = None                     # Drop the handle.

    def _log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"  # Timestamped line (HH:MM:SS).
        self.log.appendPlainText(line)                # Show it in the on-screen log panel.
        if self._log_file is not None:                # If a session log is open...
            try:
                self._log_file.write(line + "\n"); self._log_file.flush()  # ...also persist it to disk.
            except Exception:
                pass                                  # Never let logging crash the GUI.

    def closeEvent(self, e):                          # Called when the window is closing.
        try:
            self.audio.shutdown()                     # Stop audio.
            self.vision.stop(); self.vision.wait(2000)  # Stop + join the CV worker.
            self.arm.stop(); self.arm.wait(2000)      # Stop + join the arm worker.
            self._close_session_log()                 # Make sure the session log is finalised.
        finally:
            e.accept()                                # Allow the close.


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def _apply_app_style(app):
    app.setStyleSheet(build_stylesheet())             # Apply the global QSS.
    app.setFont(QFont("Segoe UI", 10))                # Default app font.


def selftest():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # Render headlessly (no real window).
    app = QApplication.instance() or QApplication(sys.argv)  # Reuse or create the QApplication.
    _apply_app_style(app)
    win = MainWindow(); win.show()                    # Build + show the window.
    win.audio.set_enabled(False)   # never speak during the headless self-test

    dummy = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8); dummy[:] = (40, 30, 25)  # Fake frame.
    cv2.putText(dummy, "SELFTEST", (60, 240), FONT, 2.0, (200, 200, 200), 3)  # Label it.
    qimg = bgr_to_qimage(dummy)                       # Convert to a QImage once.
    for i in range(30):                               # Feed 30 synthetic frames...
        r = FrameResult(state=("RED" if i % 7 == 0 else "GREEN"),  # ...periodically RED...
                        hands=i % 3, objects=i % 4, collision=(i % 7 == 0),
                        max_area=30_000 + i * 1500, fps=28 + (i % 5),
                        infer_ms=18 + (i % 6), device="cpu")
        win._on_frame(qimg, r)                        # ...through the real frame handler.
    win._on_arm(bgr_to_qimage(render_arm_2d(1.0, False)))  # Exercise the arm view.
    win._on_arm_mode("3D URDF")                       # Exercise the retitle path.
    win._toggle_estop(); win._on_frame(qimg, r)   # engage E-STOP, repaint feed
    win._toggle_estop()                            # release E-STOP
    win.take_snapshot()                               # Exercise the snapshot path.
    print("[selftest] UI constructed + updated OK; widgets:",  # Report widget count.
          len(win.findChildren(QWidget)))
    QTimer.singleShot(0, app.quit); app.exec()        # Run one event loop tick, then quit.
    print("[selftest] PASS")
    return 0                                           # Success exit code.


def main():
    if "--selftest" in sys.argv:                      # If launched with --selftest...
        sys.exit(selftest())                          # ...run the headless test and exit.
    app = QApplication(sys.argv)                      # Otherwise start the real app.
    app.setApplicationName("Safety Rated Monitor Stop")
    _apply_app_style(app)
    win = MainWindow(); win.show()                    # Build + show the window.
    sys.exit(app.exec())                              # Run the Qt event loop until quit.


if __name__ == "__main__":                            # Only run main() when executed directly.
    main()

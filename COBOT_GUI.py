"""
 SAFETY RATED MONITOR STOP - GRADUATION PROJECT
"""
from __future__ import annotations

import os
import sys
import math
import time
import queue
import threading
import subprocess
import importlib.util
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QRectF, QPointF
from PySide6.QtGui import (
    QImage, QPainter, QColor, QPen, QBrush, QFont, QPainterPath,
    QLinearGradient, QPolygonF, QShortcut, QKeySequence,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSlider,
    QCheckBox, QComboBox, QHBoxLayout, QVBoxLayout, QFrame, QSizePolicy,
    QPlainTextEdit,
)



CAM_INDEX        = 0
FRAME_WIDTH      = 640
FRAME_HEIGHT     = 480
FLIP_VIEW        = True

YOLO_MODEL_PATH  = "yolov8s.pt"
YOLO_CONF        = 0.70

MAX_HANDS        = 2
HAND_DET_CONF    = 0.60
HAND_TRACK_CONF  = 0.60

AREA_THRESHOLD   = 40_000
THRESHOLD_MIN    = 4_000
THRESHOLD_MAX    = 150_000
THRESHOLD_STEP   = 2_000

URDF_PATH = r"C:\Users\original\OneDrive\Desktop\GRAD VISION\HUSSEIN\urdf\HUSSEIN.urdf"
ARM_W, ARM_H     = 600, 820
ARM_SWEEP_DEG    = 100.0
RANGE_JOINTS     = {
    0: (-30.0, 30.0),
    1: (-18.0, 18.0),
    2: (-22.0, 22.0),
    3: (-15.0, 15.0),
    4: (-35.0, 35.0),
}
FIXED_JOINTS     = {}
ARM_SPEED        = 0.45

FACE_DB_PATH     = r"C:\Users\original\OneDrive\Desktop\GRAD VISION\Models\faces_db.npz"
FACE_THRESHOLD   = 0.50
FACE_EVERY_N     = 2

SNAPSHOT_DIR     = "snapshots"
LOG_DIR          = "logs"

VOICE_ALERTS_DEFAULT = True
RED_ZONE_PHRASE      = "Red zone"
BUZZER_FREQ          = 600
BUZZER_DUR_MS        = 300
BUZZER_GAP_S         = 0.15
ESTOP_ON_PHRASE      = "Emergency stop activated."
ESTOP_OFF_PHRASE     = "Emergency stop released."

RED_ALERT_SECS       = 5.0
GREET_REENTRY_SECS   = 100000



class C:
    """Central colour palette."""
    BG          = "#080b15"
    BG2         = "#0d1322"
    PANEL       = "#0f1626"
    CARD        = "#121a2e"
    CARD2       = "#0e1525"
    BORDER      = "#283556"
    BORDER_SOFT = "#1b2440"
    TRACK       = "#1d2742"
    TEXT        = "#eef2ff"
    MUTED       = "#8c96b6"
    FAINT       = "#525d80"

    BLUE        = "#3b82f6"
    CYAN        = "#22d3ee"
    VIOLET      = "#a39abd"
    GREEN       = "#22c55e"
    AMBER       = "#f59e0b"
    PINK        = "#ec4899"
    RED         = "#ef4444"
    ORANGE      = "#fb923c"
    TEAL        = "#2dd4bf"

    ACCENT      = BLUE
    SAFE        = GREEN
    DANGER      = RED
    WARN        = AMBER


BGR_SAFE   = (94, 197, 34)
BGR_DANGER = (68, 68, 239)
BGR_WARN   = (11, 158, 245)
BGR_CYAN   = (238, 211, 34)
BGR_WHITE  = (245, 245, 245)
BGR_DARK   = (20, 16, 12)
FONT       = cv2.FONT_HERSHEY_SIMPLEX


def build_stylesheet() -> str:
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


def qc(h: str, a: int = 255) -> QColor:
    c = QColor(h); c.setAlpha(a); return c



class AudioAlerter:

    def __init__(self):
        self._q = queue.Queue()
        self._thread = None
        self._buzzer_stop = threading.Event()
        self._buzzer_thread = None
        self._lock = threading.Lock()
        self._priority_active = False
        self._enabled = (VOICE_ALERTS_DEFAULT and
                         os.environ.get("QT_QPA_PLATFORM") != "offscreen")
        if importlib.util.find_spec("pyttsx3") is not None:
            self._backend = "pyttsx3"
        elif sys.platform.startswith("win"):
            self._backend = "sapi"
        else:
            self._backend = "beep"

    def backend_name(self) -> str:
        return self._backend

    def set_enabled(self, on: bool):
        self._enabled = bool(on)
        if not self._enabled:
            self.stop_buzzer()

    def alert_red_zone(self, duration_s=RED_ALERT_SECS):
        """Say the red-zone phrase once, then buzz for `duration_s` seconds and
        stop on its own - a fixed-length alert, not an open-ended one."""
        if not self._enabled:
            return
        self.stop_buzzer()
        self._buzzer_stop.clear()
        self._buzzer_thread = threading.Thread(
            target=self._red_zone_loop, args=(duration_s,),
            name="RedZoneBuzzer", daemon=True)
        self._buzzer_thread.start()

    def is_alerting(self) -> bool:
        """True while the red-zone alert episode (phrase + buzzer) is running."""
        t = self._buzzer_thread
        return t is not None and t.is_alive()

    def stop_buzzer(self):
        """Stop the repeating red-zone buzzer (no-op if it is not running)."""
        self._buzzer_stop.set()
        t = self._buzzer_thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)
        self._buzzer_thread = None

    def say(self, text: str, priority: bool = False):
        """Queue a phrase to speak, dropping any older pending phrase.

        A high-priority phrase (emergency stop / red zone) replaces whatever is
        queued and blocks low-priority phrases until it finishes. A low-priority
        phrase (greeting) is dropped if a high-priority phrase is in flight or
        the red-zone alert is sounding - so safety alerts always win a clash.
        """
        if not self._enabled:
            return
        self._ensure_thread()
        with self._lock:
            if not priority and (self._priority_active or self.is_alerting()):
                return
            self._drain_queue()
            if priority:
                self._priority_active = True
            self._q.put((priority, text))

    def _drain_queue(self):
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def shutdown(self):
        self._enabled = False
        self.stop_buzzer()
        if self._thread is not None:
            self._q.put(None)

    def _red_zone_loop(self, duration_s=RED_ALERT_SECS):
        deadline = time.monotonic() + max(0.5, float(duration_s))
        try:
            self._speak(RED_ZONE_PHRASE)
        except Exception:
            pass
        while not self._buzzer_stop.is_set() and time.monotonic() < deadline:
            self._buzz_once()
            self._buzzer_stop.wait(BUZZER_GAP_S)
        if self._buzzer_thread is threading.current_thread():
            self._buzzer_thread = None

    def _buzz_once(self):
        try:
            import winsound
            winsound.Beep(BUZZER_FREQ, BUZZER_DUR_MS)
        except Exception:
            try:
                print("\a", end="", flush=True)
            except Exception:
                pass

    def _ensure_thread(self):
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="AudioAlerter", daemon=True)
            self._thread.start()

    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            priority, text = item
            try:
                self._speak(text)
            except Exception:
                pass
            if priority:
                with self._lock:
                    if self._q.empty():
                        self._priority_active = False

    def _speak(self, text: str):
        if self._backend == "pyttsx3":
            try:
                import pyttsx3
                eng = pyttsx3.init()
                try:
                    eng.setProperty("rate", 178)
                    eng.setProperty("volume", 1.0)
                except Exception:
                    pass
                eng.say(text)
                eng.runAndWait()
                try:
                    eng.stop()
                except Exception:
                    pass
                return
            except Exception:
                self._backend = "sapi" if sys.platform.startswith("win") else "beep"

        if self._backend == "sapi":
            try:
                safe = text.replace("'", " ").replace('"', " ")
                ps = (
                    "Add-Type -AssemblyName System.Speech; "
                    "$v = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    "$v.Rate = 1; $v.Volume = 100; "
                    f"$v.Speak('{safe}')"
                )
                subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    timeout=12,
                )
                return
            except Exception:
                self._backend = "beep"

        try:
            import winsound
            winsound.Beep(920, 220)
            winsound.Beep(680, 320)
        except Exception:
            try:
                print("\a", end="", flush=True)
            except Exception:
                pass



@dataclass
class FrameResult:
    state: str = "GREEN"
    hands: int = 0
    objects: int = 0
    collision: bool = False
    max_area: int = 0
    fps: float = 0.0
    infer_ms: float = 0.0
    device: str = "cpu"
    faces: int = 0
    identities: tuple = ()


def _hand_bbox(landmarks, w: int, h: int):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x1 = max(0, int(min(xs) * w)); x2 = min(w - 1, int(max(xs) * w))
    y1 = max(0, int(min(ys) * h)); y2 = min(h - 1, int(max(ys) * h))
    area = max(0, x2 - x1) * max(0, y2 - y1)
    return x1, y1, x2, y2, area


def _label(frame, text, x, y, bg, fg=BGR_WHITE, scale=0.5):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    y = max(y, th + 6)
    cv2.rectangle(frame, (x, y - th - base - 4), (x + tw + 8, y), bg, -1, cv2.LINE_AA)
    cv2.putText(frame, text, (x + 4, y - 5), FONT, scale, fg, 1, cv2.LINE_AA)


class FaceRecognizer:


    def __init__(self, db_path, threshold, device):
        self.available = False
        self.error = ""
        self.threshold = threshold
        self.names = ()
        try:
            import onnxruntime as ort
            from insightface.app import FaceAnalysis

            data = np.load(db_path, allow_pickle=False)
            self.names = data["names"]
            emb = data["embeddings"].astype(np.float32)
            self._db = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

            avail = ort.get_available_providers()
            if device == "cuda" and "CUDAExecutionProvider" in avail:
                providers, ctx = ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
            else:
                providers, ctx = ["CPUExecutionProvider"], -1
            self.app = FaceAnalysis(name="buffalo_l",
                                    allowed_modules=["detection", "recognition"],
                                    providers=providers)
            self.app.prepare(ctx_id=ctx, det_size=(640, 640))
            self.available = True
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"

    def identify(self, frame):
        """Return [(name, score, (x1, y1, x2, y2)), ...] for faces in `frame`."""
        if not self.available:
            return []
        out = []
        for face in self.app.get(frame):
            emb = face.embedding.astype(np.float32)
            emb /= (np.linalg.norm(emb) + 1e-9)
            sims = self._db @ emb
            idx = int(np.argmax(sims))
            score = float(sims[idx])
            name = str(self.names[idx]) if score >= self.threshold else "Unknown"
            x1, y1, x2, y2 = face.bbox.astype(int)
            out.append((name, score, (x1, y1, x2, y2)))
        return out


def _draw_faces(frame, faces):
    for name, score, (x1, y1, x2, y2) in faces:
        known = name != "Unknown"
        col = BGR_CYAN if known else BGR_WARN
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
        _label(frame, f"{name}", x1, max(0, y1 - 4), BGR_DARK, col)


class VisionPipeline:


    def __init__(self, model_path: str, device: str):
        from ultralytics import YOLO
        import mediapipe as mp

        self.device = device
        self.yolo = YOLO(model_path)
        self.face = FaceRecognizer(FACE_DB_PATH, FACE_THRESHOLD, device)
        self._frame_i = 0
        self._last_faces = []
        self._mp = mp
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=MAX_HANDS, model_complexity=0,
            min_detection_confidence=HAND_DET_CONF,
            min_tracking_confidence=HAND_TRACK_CONF)
        self._draw = mp.solutions.drawing_utils
        self._styles = mp.solutions.drawing_styles
        self._hands_conn = mp.solutions.hands.HAND_CONNECTIONS

    def process(self, frame, conf, threshold, draw_landmarks=True, flip=True,
                recognize_faces=True, detect_objects=True):
        t0 = time.perf_counter()
        if flip:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        objects = 0
        if detect_objects:
            kw = {"conf": conf, "verbose": False}
            if self.device:
                kw["device"] = self.device
            for res in self.yolo.predict(frame, **kw):
                names = res.names
                for box in res.boxes:
                    c = float(box.conf[0])
                    if c < conf:
                        continue
                    cls = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), BGR_WARN, 2, cv2.LINE_AA)
                    _label(frame, f"{names.get(cls, cls)}",
                           x1, y1, BGR_DARK, BGR_WARN)
                    objects += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        hres = self.hands.process(rgb)
        rgb.flags.writeable = True

        state = "GREEN"; n_hands = 0; max_area = 0
        if hres.multi_hand_landmarks:
            n_hands = len(hres.multi_hand_landmarks)
            for lm in hres.multi_hand_landmarks:
                if draw_landmarks:
                    self._draw.draw_landmarks(
                        frame, lm, self._hands_conn,
                        self._styles.get_default_hand_landmarks_style(),
                        self._styles.get_default_hand_connections_style())
                x1, y1, x2, y2, area = _hand_bbox(lm.landmark, w, h)
                max_area = max(max_area, area)
                hot = area > threshold
                color = BGR_DANGER if hot else BGR_SAFE
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                pct = min(100, int(area / threshold * 100))
                _label(frame, f"{pct}% proximity", x1, y1, BGR_DARK, color)
                if hot:
                    state = "RED"

        self._frame_i += 1
        if recognize_faces and self.face.available:
            if self._frame_i % FACE_EVERY_N == 0:
                self._last_faces = self.face.identify(frame)
            _draw_faces(frame, self._last_faces)
        else:
            self._last_faces = []
        identities = tuple(n for n, _s, _b in self._last_faces if n != "Unknown")

        collision = state == "RED" and objects > 0
        dt = (time.perf_counter() - t0) * 1000.0
        return frame, FrameResult(state=state, hands=n_hands, objects=objects,
                                  collision=collision, max_area=max_area,
                                  infer_ms=dt, device=self.device,
                                  faces=len(self._last_faces), identities=identities)

    def close(self):
        try:
            self.hands.close()
        except Exception:
            pass



class RobotArm3D:


    def __init__(self, urdf_path, width, height):
        import yourdfpy
        import pyrender
        self._pyrender = pyrender
        self.width, self.height = width, height
        self.robot = yourdfpy.URDF.load(urdf_path)
        self.joint_names = list(self.robot.actuated_joint_names)
        max_swing = math.radians(ARM_SWEEP_DEG)
        self._joint_motion = []
        for idx, n in enumerate(self.joint_names):
            joint = self.robot.joint_map.get(n)
            lim = getattr(joint, "limit", None)
            lo = getattr(lim, "lower", None)
            hi = getattr(lim, "upper", None)
            if lo is not None and hi is not None and hi > lo:
                center = 0.5 * (lo + hi)
                half = 0.5 * (hi - lo)
            else:
                center = 0.0
                half = max_swing
            half = min(half, max_swing)
            if idx in RANGE_JOINTS:
                lo_d, hi_d = RANGE_JOINTS[idx]
                center = math.radians(0.5 * (lo_d + hi_d))
                half = math.radians(0.5 * (hi_d - lo_d))
            if idx in FIXED_JOINTS:
                center = math.radians(FIXED_JOINTS[idx])
                half = 0.0
            self._joint_motion.append((center, half))
        centers = [c for c, h in self._joint_motion]
        mins    = [c - h for c, h in self._joint_motion]
        maxs    = [c + h for c, h in self._joint_motion]
        sample_cfgs = [centers, mins, maxs]
        for i in range(len(self.joint_names)):
            for val in (mins[i], maxs[i]):
                cfg = list(centers); cfg[i] = val
                sample_cfgs.append(cfg)
        lo_b = np.full(3, np.inf); hi_b = np.full(3, -np.inf)
        for vals in sample_cfgs:
            self.robot.update_cfg({n: v for n, v in zip(self.joint_names, vals)})
            b = self.robot.scene.bounds
            lo_b = np.minimum(lo_b, b[0]); hi_b = np.maximum(hi_b, b[1])
        self._center = 0.5 * (lo_b + hi_b)
        self._radius = max(float(np.linalg.norm(hi_b - lo_b)), 1e-3)
        self.renderer = pyrender.OffscreenRenderer(width, height)
        import colorsys
        self._link_names = list(self.robot.scene.graph.nodes_geometry)
        n_links = max(len(self._link_names), 1)
        self._link_colors = []
        for k in range(n_links):
            r, g, b = colorsys.hsv_to_rgb(k / n_links, 0.65, 1.0)
            self._link_colors.append([r, g, b, 1.0])

    def _camera_pose(self):
        eyedir = np.array([0.45, -0.35, 1.0])
        eyedir /= np.linalg.norm(eyedir)
        eye = self._center + eyedir * (self._radius * 1.30)
        fwd = self._center - eye; fwd /= np.linalg.norm(fwd)
        up = np.array([0.0, -1.0, 0.0])
        right = np.cross(fwd, up); right /= np.linalg.norm(right)
        true_up = np.cross(right, fwd)
        pose = np.eye(4)
        pose[:3, 0] = right; pose[:3, 1] = true_up
        pose[:3, 2] = -fwd;  pose[:3, 3] = eye
        return pose

    def render(self, phase, stopped):
        pyrender = self._pyrender
        cfg = {}
        for i, n in enumerate(self.joint_names):
            center, half = self._joint_motion[i]
            cfg[n] = center + half * math.sin(phase + i * 0.7)
        self.robot.update_cfg(cfg)
        scene = pyrender.Scene(bg_color=[0.04, 0.06, 0.11, 1.0],
                               ambient_light=[0.5, 0.5, 0.5])
        ts = self.robot.scene
        for i, node in enumerate(ts.graph.nodes_geometry):
            T, gname = ts.graph.get(node)
            color = self._link_colors[i % len(self._link_colors)]
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=color, metallicFactor=0.45, roughnessFactor=0.5)
            mesh = pyrender.Mesh.from_trimesh(ts.geometry[gname], smooth=False,
                                              material=material)
            scene.add(mesh, pose=np.asarray(T))
        cam_pose = self._camera_pose()
        scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4.0), pose=cam_pose)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.2),
                  pose=cam_pose)
        color, _ = self.renderer.render(scene)
        return cv2.cvtColor(np.asarray(color), cv2.COLOR_RGB2BGR)

    def legend_entries(self):
        """List of (link_name, '#rrggbb') for the GUI side-panel colour key."""
        out = []
        for name, col in zip(self._link_names, self._link_colors):
            hexc = "#%02x%02x%02x" % (int(col[0] * 255),
                                      int(col[1] * 255), int(col[2] * 255))
            out.append((str(name), hexc))
        return out

    def close(self):
        try:
            self.renderer.delete()
        except Exception:
            pass


def render_arm_2d(phase, stopped, w=ARM_W, h=ARM_H):
    """Dependency-free 2-link arm - the fallback when 3D is unavailable."""
    canvas = np.full((h, w, 3), 18, dtype=np.uint8)
    base = (w // 2, h - 70)
    L1, L2 = 150, 120
    a1 = -math.pi / 2 + 0.6 * math.sin(phase)
    a2 = -0.8 + 0.7 * math.sin(phase * 1.3)
    bx, by = base
    jx = int(bx + L1 * math.cos(a1)); jy = int(by + L1 * math.sin(a1))
    ex = int(jx + L2 * math.cos(a1 + a2)); ey = int(jy + L2 * math.sin(a1 + a2))
    col = BGR_DANGER if stopped else BGR_SAFE
    cv2.circle(canvas, base, 16, BGR_WHITE, -1, cv2.LINE_AA)
    cv2.line(canvas, base, (jx, jy), col, 11, cv2.LINE_AA)
    cv2.circle(canvas, (jx, jy), 10, BGR_WHITE, -1, cv2.LINE_AA)
    cv2.line(canvas, (jx, jy), (ex, ey), col, 9, cv2.LINE_AA)
    cv2.circle(canvas, (ex, ey), 12, BGR_CYAN, -1, cv2.LINE_AA)
    cv2.circle(canvas, (ex, ey), 12, col, 2, cv2.LINE_AA)
    return canvas


def _overlay_estop(canvas):
    """Stamp a bold red EMERGENCY-STOP wash, border and label over an arm frame
    so a stop is unmistakable in BOTH the 3D and the 2D renderers (the 3D render
    on its own gives no visual cue that the arm has frozen)."""
    h, w = canvas.shape[:2]
    wash = np.empty_like(canvas); wash[:] = BGR_DANGER
    cv2.addWeighted(wash, 0.30, canvas, 0.70, 0, canvas)
    cv2.rectangle(canvas, (2, 2), (w - 3, h - 3), BGR_DANGER, 8, cv2.LINE_AA)
    txt = "EMERGENCY STOP"
    scale = max(0.8, w / 700.0)
    (tw, th), _ = cv2.getTextSize(txt, FONT, scale, 3)
    x = (w - tw) // 2; y = th + 34
    cv2.rectangle(canvas, (x - 16, y - th - 16), (x + tw + 16, y + 16),
                  BGR_DARK, -1, cv2.LINE_AA)
    cv2.putText(canvas, txt, (x, y), FONT, scale, BGR_WHITE, 3, cv2.LINE_AA)
    return canvas



def bgr_to_qimage(frame_bgr) -> QImage:
    """OpenCV BGR -> deep-copied QImage (safe cross-thread handoff)."""
    rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


class VisionWorker(QThread):
    frameReady = Signal(QImage, object)
    status     = Signal(str)
    failed     = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self.cam_index = CAM_INDEX
        self.conf = YOLO_CONF
        self.threshold = AREA_THRESHOLD
        self.draw_landmarks = True
        self.flip = FLIP_VIEW
        self.device = "cuda"
        self.recognize_faces = True
        self.detect_objects = True
        self._last_frame = None

    def set_conf(self, v):       self.conf = float(v)
    def set_threshold(self, v):  self.threshold = int(v)
    def set_landmarks(self, v):  self.draw_landmarks = bool(v)
    def set_flip(self, v):       self.flip = bool(v)
    def set_face(self, v):       self.recognize_faces = bool(v)
    def set_objects(self, v):    self.detect_objects = bool(v)
    def stop(self):              self._running = False

    def run(self):
        try:
            import torch
            if self.device == "cuda" and not torch.cuda.is_available():
                self.device = "cpu"
        except Exception:
            self.device = "cpu"

        self.status.emit(f"Loading YOLOv8 + MediaPipe  ({self.device.upper()})...")
        try:
            pipe = VisionPipeline(YOLO_MODEL_PATH, self.device)
        except Exception as e:
            self.failed.emit(f"Model load failed: {e}")
            return

        if pipe.face.available:
            self.status.emit(f"Face DB loaded: {len(pipe.face.names)} identities")
        else:
            self.status.emit(f"Face recognition off ({pipe.face.error or 'no DB'})")

        self.status.emit("Opening camera...")
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            self.failed.emit(f"Cannot open camera index {self.cam_index}")
            pipe.close(); return

        self.status.emit("Live")
        self._running = True
        prev = time.perf_counter()
        while self._running:
            ok, frame = cap.read()
            if not ok:
                self.status.emit("Camera frame dropped"); continue
            frame, result = pipe.process(
                frame, self.conf, self.threshold, self.draw_landmarks, self.flip,
                self.recognize_faces, self.detect_objects)
            now = time.perf_counter()
            result.fps = 1.0 / max(1e-6, now - prev); prev = now
            self._last_frame = frame
            self.frameReady.emit(bgr_to_qimage(frame), result)

        cap.release(); pipe.close()
        self.status.emit("Stopped")

    def snapshot_path(self):
        if self._last_frame is None:
            return None
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        path = os.path.join(SNAPSHOT_DIR, f"cobot_{int(time.time())}.png")
        cv2.imwrite(path, self._last_frame)
        return path


class ArmWorker(QThread):
    armReady = Signal(QImage)
    mode     = Signal(str)
    legend   = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._stopped = False
        self._phase = 0.0

    def set_stopped(self, v):  self._stopped = bool(v)
    def stop(self):            self._running = False

    def run(self):
        arm = None
        try:
            arm = RobotArm3D(URDF_PATH, ARM_W, ARM_H)
            self.mode.emit("3D URDF")
            self.legend.emit(arm.legend_entries())
        except Exception:
            arm = None
            self.mode.emit("2D fallback")

        self._running = True
        while self._running:
            if not self._stopped:
                self._phase += ARM_SPEED
            try:
                canvas = (arm.render(self._phase, self._stopped) if arm
                          else render_arm_2d(self._phase, self._stopped))
            except Exception:
                arm = None
                self.mode.emit("2D fallback")
                canvas = render_arm_2d(self._phase, self._stopped)
            if self._stopped:
                canvas = _overlay_estop(canvas)
            self.armReady.emit(bgr_to_qimage(canvas))
            self.msleep(33)

        if arm is not None:
            arm.close()



class VideoView(QWidget):


    rightClicked = Signal()

    def __init__(self, placeholder="NO SIGNAL", parent=None, lock_aspect=True):
        super().__init__(parent)
        self._img = None
        self._lock_aspect = lock_aspect
        self._placeholder = placeholder
        self._estop = False
        self._flashing = False
        self._flash_on = False
        self._flash_blink = QTimer(self)
        self._flash_blink.setInterval(220)
        self._flash_blink.timeout.connect(self._toggle_flash)
        self._flash_stop = QTimer(self)
        self._flash_stop.setSingleShot(True)
        self._flash_stop.timeout.connect(self.stop_flash)
        self.setMinimumSize(320, 240)
        self._aspect = FRAME_WIDTH / FRAME_HEIGHT
        if lock_aspect:
            sp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            sp.setHeightForWidth(True)
            self.setSizePolicy(sp)
        else:
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setContextMenuPolicy(Qt.PreventContextMenu)

    def hasHeightForWidth(self):
        return self._lock_aspect

    def heightForWidth(self, w):
        return int(round(w / self._aspect)) if self._aspect > 0 else w

    def set_image(self, img):
        self._img = img
        if (self._lock_aspect and img is not None and not img.isNull()
                and img.height() > 0):
            a = img.width() / img.height()
            if abs(a - self._aspect) > 1e-3:
                self._aspect = a
                self.updateGeometry()
        self.update()
    def set_placeholder(self, t): self._placeholder = t; self._img = None; self.update()
    def clear(self): self._img = None; self.update()
    def set_estop(self, on): self._estop = bool(on); self.update()

    def _toggle_flash(self): self._flash_on = not self._flash_on; self.update()

    def start_flash(self, duration_ms=5000):
        """Blink a red DANGER overlay for `duration_ms`, then stop on its own."""
        self._flashing = True; self._flash_on = True
        self._flash_blink.start()
        self._flash_stop.start(int(duration_ms))
        self.update()

    def stop_flash(self):
        self._flash_blink.stop(); self._flash_stop.stop()
        self._flashing = False; self._flash_on = False
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton:
            self.rightClicked.emit(); e.accept()
        else:
            super().mousePressEvent(e)

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(0, 0, -1, -1)
        path = QPainterPath(); path.addRoundedRect(QRectF(r), 10, 10)
        p.setClipPath(path)
        p.fillRect(r, qc("#06080f"))
        if self._img is not None and not self._img.isNull():
            s = self._img.scaled(self.size(), Qt.KeepAspectRatio,
                                 Qt.SmoothTransformation)
            p.drawImage((self.width() - s.width()) // 2,
                        (self.height() - s.height()) // 2, s)
        else:
            p.setPen(qc(C.FAINT))
            f = QFont("Segoe UI", 11); f.setLetterSpacing(QFont.AbsoluteSpacing, 2)
            p.setFont(f); p.drawText(r, Qt.AlignCenter, self._placeholder)
        p.setClipping(False)
        if self._estop:
            p.setPen(QPen(qc(C.RED), 3)); p.setBrush(Qt.NoBrush); p.drawPath(path)
            p.setClipPath(path)
            band = QRectF(r.left(), r.top(), r.width(), 34)
            p.fillRect(band, qc(C.RED, 220))
            p.setPen(qc("#ffffff"))
            f = QFont("Segoe UI", 12, QFont.Black)
            f.setLetterSpacing(QFont.AbsoluteSpacing, 1)
            p.setFont(f)
            p.drawText(band, Qt.AlignCenter,
                       "EMERGENCY STOP  -  right-click to release")
            p.setClipping(False)
        else:
            p.setPen(QPen(qc(C.BORDER), 1)); p.setBrush(Qt.NoBrush); p.drawPath(path)
        if self._flashing and self._flash_on:
            p.setClipPath(path)
            p.fillRect(r, qc(C.RED, 95))
            p.setPen(QPen(qc(C.RED), 6)); p.setBrush(Qt.NoBrush); p.drawPath(path)
            band = QRectF(r.left(), r.bottom() - 33, r.width(), 33)
            p.fillRect(band, qc(C.RED, 235))
            p.setPen(qc("#ffffff"))
            f = QFont("Segoe UI", 12, QFont.Black)
            f.setLetterSpacing(QFont.AbsoluteSpacing, 1)
            p.setFont(f)
            p.drawText(band, Qt.AlignCenter, "!  RED ZONE  -  DANGER  !")
            p.setClipping(False)


class StatusPill(QWidget):


    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(312, 60)
        self._safe = True
        self._estop = False
        self._pulse = 0.0
        t = QTimer(self); t.timeout.connect(self._tick); t.start(40)

    def set_safe(self, safe): self._safe = safe; self.update()
    def set_estop(self, on): self._estop = bool(on); self.update()
    def _tick(self): self._pulse = (self._pulse + 0.08) % (2 * math.pi); self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        base = qc(C.SAFE if (self._safe and not self._estop) else C.DANGER)
        bg = QColor(base); bg.setAlpha(45)
        p.setBrush(bg); p.setPen(QPen(base, 1.6)); p.drawRoundedRect(r, 15, 15)
        glow = 0.5 + 0.5 * math.sin(self._pulse)
        cx, cy = 28, self.height() / 2
        halo = QColor(base); halo.setAlpha(int(80 * glow))
        p.setPen(Qt.NoPen); p.setBrush(halo)
        p.drawEllipse(QPointF(cx, cy), 14, 14)
        p.setBrush(base); p.drawEllipse(QPointF(cx, cy), 6.5, 6.5)
        p.setPen(base)
        f = QFont("Segoe UI", 18, QFont.Black); f.setLetterSpacing(QFont.AbsoluteSpacing, 1.5)
        p.setFont(f)
        text = ("EMERGENCY STOP" if self._estop else
                "SYSTEM RUNNING" if self._safe else "SYSTEM STOP")
        p.drawText(QRectF(50, 0, self.width() - 56, self.height()),
                   Qt.AlignVCenter | Qt.AlignLeft, text)


class RingGauge(QFrame):


    def __init__(self, title, unit="", color=C.BLUE, vmax=100.0,
                 dynamic=False, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self._title = title; self._unit = unit; self._color = color
        self._vmax = max(float(vmax), 1e-6); self._dynamic = dynamic
        self._val = 0.0
        self.setMinimumSize(182, 158)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_value(self, v): self._val = float(v); self.update()

    def _col(self):
        if self._dynamic:
            return qc(C.GREEN if self._val < 60 else
                      C.AMBER if self._val < 100 else C.RED)
        return qc(self._color)

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        p.setPen(qc(C.MUTED)); p.setFont(QFont("Segoe UI", 8, QFont.Bold))
        p.drawText(QRectF(0, 11, W, 14), Qt.AlignHCenter, self._title.upper())

        tw = 11
        d = max(min(W - 46, H - 60), 36)
        cx, cy = W / 2.0, 30 + d / 2.0
        rect = QRectF(cx - d / 2, cy - d / 2, d, d)
        p.setPen(QPen(qc(C.TRACK), tw)); p.drawArc(rect, 0, 360 * 16)
        col = self._col()
        frac = max(0.0, min(self._val / self._vmax, 1.0))
        pen = QPen(col, tw); pen.setCapStyle(Qt.RoundCap); p.setPen(pen)
        p.drawArc(rect, 90 * 16, -int(frac * 360 * 16))

        p.setPen(col); p.setFont(QFont("Segoe UI", 21, QFont.Bold))
        p.drawText(rect, Qt.AlignCenter, f"{self._val:.0f}")
        if self._unit:
            p.setPen(qc(C.MUTED)); p.setFont(QFont("Segoe UI", 8))
            p.drawText(QRectF(rect.left(), cy + d * 0.18, d, 13),
                       Qt.AlignHCenter, self._unit)


class StatChip(QFrame):


    def __init__(self, label, color=C.BLUE, parent=None):
        super().__init__(parent)
        self._label = label; self._color = color; self._val = "--"
        self.setMinimumHeight(38)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set(self, val, color=None):
        self._val = str(val)
        if color: self._color = color
        self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        base = qc(self._color); bg = QColor(base); bg.setAlpha(38)
        p.setBrush(bg); p.setPen(QPen(base, 1.2)); p.drawRoundedRect(r, 9, 9)
        cy = r.center().y()
        p.setPen(Qt.NoPen); p.setBrush(base)
        p.drawEllipse(QPointF(r.left() + 14, cy), 5, 5)
        p.setPen(qc(C.MUTED)); p.setFont(QFont("Segoe UI", 8, QFont.Bold))
        p.drawText(QRectF(r.left() + 25, r.top(), r.width() - 62, r.height()),
                   Qt.AlignVCenter | Qt.AlignLeft, self._label.upper())
        p.setPen(base); p.setFont(QFont("Segoe UI", 12, QFont.Bold))
        p.drawText(QRectF(r.left(), r.top(), r.width() - 13, r.height()),
                   Qt.AlignVCenter | Qt.AlignRight, self._val)


class LinkLegend(QWidget):


    SW   = 16
    PADX = 12
    PADY = 3
    GAPX = 22
    ROWH = 22

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries = []
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._apply_height(1)

    def _apply_height(self, rows):
        h = rows * self.ROWH + 2 * self.PADY
        self.setFixedHeight(h)

    def set_entries(self, entries):
        self._entries = list(entries)
        rows = self._layout_rows(max(self.width(), 600))[1]
        self._apply_height(rows)
        self.update()

    def _layout_rows(self, width):
        """Return list of (x, name, hexc) plus total rows for a given width."""
        f = QFont("Segoe UI", 11, QFont.Bold)
        fm = self.fontMetrics() if not self._entries else None
        from PySide6.QtGui import QFontMetrics
        fm = QFontMetrics(f)
        placed, x, row = [], self.PADX, 0
        for name, hexc in self._entries:
            tw = fm.horizontalAdvance(str(name))
            step = self.SW + 8 + tw + self.GAPX
            if x + step > width - self.PADX and x > self.PADX:
                row += 1; x = self.PADX
            placed.append((x, row, name, hexc, tw))
            x += step
        return placed, row + 1, f

    def resizeEvent(self, e):
        if self._entries:
            self._apply_height(self._layout_rows(self.width())[1])
        super().resizeEvent(e)

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        if not self._entries:
            p.setPen(qc(C.MUTED)); p.setFont(QFont("Segoe UI", 10))
            p.drawText(self.rect(), Qt.AlignCenter, "waiting for arm…")
            return
        placed, rows, font = self._layout_rows(self.width())
        p.setFont(font)
        for x, row, name, hexc, tw in placed:
            y = self.PADY + row * self.ROWH
            col = QColor(hexc)
            p.setBrush(col); p.setPen(QPen(QColor("#ffffff"), 2))
            p.drawRoundedRect(QRectF(x, y, self.SW, self.SW), 5, 5)
            p.setPen(qc(C.TEXT))
            p.drawText(QRectF(x + self.SW + 8, y - 4, tw + 6, self.SW + 8),
                       Qt.AlignVCenter | Qt.AlignLeft, str(name))


class Sparkline(QFrame):


    def __init__(self, title="FPS", color=C.BLUE, maxlen=120, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setMinimumHeight(96)
        self._title = title; self._color = color
        self._data = deque(maxlen=maxlen)

    def push(self, v): self._data.append(float(v)); self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        m = 12
        
        area = self.rect().adjusted(m, m + 14, -m, -m)
        col = qc(self._color)
        p.setPen(qc(C.MUTED)); p.setFont(QFont("Segoe UI", 8, QFont.Bold))
        cur = self._data[-1] if self._data else 0.0
        p.drawText(QRectF(m, 7, self.width() - 2 * m, 14), Qt.AlignLeft, self._title)
        p.setPen(col)
        p.drawText(QRectF(m, 7, self.width() - 2 * m, 14), Qt.AlignRight, f"{cur:5.1f}")
        if len(self._data) < 2:
            return
        vmax = max(max(self._data), 1.0); vmin = min(self._data)
        rng = max(vmax - vmin, 1e-3); n = len(self._data)
        w = area.width(); h = area.height(); x0 = area.left(); y0 = area.top()
        pts = [QPointF(x0 + w * i / (n - 1), y0 + h - (v - vmin) / rng * h)
               for i, v in enumerate(self._data)]
        fill = QPolygonF(pts + [QPointF(x0 + w, y0 + h), QPointF(x0, y0 + h)])
        grad = QLinearGradient(0, y0, 0, y0 + h)
        c0 = QColor(col); c0.setAlpha(110); c1 = QColor(col); c1.setAlpha(8)
        grad.setColorAt(0, c0); grad.setColorAt(1, c1)
        p.setPen(Qt.NoPen); p.setBrush(QBrush(grad)); p.drawPolygon(fill)
        p.setPen(QPen(col, 2)); p.setBrush(Qt.NoBrush); p.drawPolyline(QPolygonF(pts))


class SafetyTimeline(QFrame):

    def __init__(self, maxlen=200, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setMinimumHeight(70)
        self._data = deque(maxlen=maxlen)

    def push(self, safe): self._data.append(bool(safe)); self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        m = 6
        p.setPen(qc(C.MUTED)); p.setFont(QFont("Segoe UI", 8, QFont.Bold))
        p.drawText(QRectF(m, 4, 240, 14), Qt.AlignLeft, "SAFETY TIMELINE")
        if not self._data:
            return
        area = self.rect().adjusted(m, m + 14, -m, -m)
        clip = QPainterPath(); clip.addRoundedRect(QRectF(area), 8, 8)
        p.setClipPath(clip)
        n = self._data.maxlen; bw = area.width() / n
        for i, safe in enumerate(self._data):
            p.fillRect(QRectF(area.left() + i * bw, area.top(),
                              max(bw - 0.4, 1), area.height()),
                       qc(C.SAFE if safe else C.DANGER))


class LogoBadge(QWidget):


    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(36, 36)

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(3, 3, 30, 30)
        p.setPen(Qt.NoPen)
        p.setBrush(qc("#22d3ee")); p.drawEllipse(r)
        p.setBrush(qc("#1e2a78")); p.drawEllipse(r.adjusted(6, 6, -6, -6))
        p.setBrush(qc("#ffffff")); p.drawEllipse(r.adjusted(12, 12, -12, -12))


def titled_card(title, body, color=C.TEXT):

    f = QFrame(); f.setObjectName("card")
    lay = QVBoxLayout(f); lay.setContentsMargins(12, 10, 12, 12); lay.setSpacing(8)
    t = QLabel(title); t.setObjectName("panelTitle"); t.setStyleSheet(f"color:{color};")
    f.title_label = t
    lay.addWidget(t); lay.addWidget(body, 1)
    return f



class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Safety Rated Monitor Stop")
        self.resize(1680, 1000)
        self.setMinimumSize(1320, 860)

        self.vision = VisionWorker(self)
        self.arm = ArmWorker(self)
        self.audio = AudioAlerter()
        self._t_start = None
        self._estop = False
        self._was_red = False
        self._last_seen = {}
        self._log_file = None
        self._log_path = None
        self._last_safe = None
        self._last_collision = False

        self._build_ui()
        self._connect_workers()
        self._wire_shortcuts()

        self._clock = QTimer(self); self._clock.timeout.connect(self._update_clock)
        self._clock.start(1000)

    def _build_ui(self):
        root = QWidget(); root.setObjectName("root"); self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(14, 14, 14, 14); outer.setSpacing(12)

        outer.addWidget(self._build_header())

        body = QHBoxLayout(); body.setSpacing(12)
        body.addWidget(self._build_center(), 1)
        body.addWidget(self._build_right_rail(), 0)
        outer.addLayout(body, 1)

        outer.addWidget(self._build_control_dock(), 0)

    def _build_header(self):
        h = QFrame(); h.setObjectName("header"); h.setFixedHeight(76)
        lay = QHBoxLayout(h); lay.setContentsMargins(22, 10, 18, 10); lay.setSpacing(16)
        logo = LogoBadge()
        tb = QVBoxLayout(); tb.setSpacing(0)
        t1 = QLabel("SAFETY RATED MONITOR"); t1.setObjectName("h1")
        tb.addWidget(t1)
        lay.addWidget(logo); lay.addLayout(tb); lay.addStretch()

        self.lbl_uptime = QLabel("IDLE")
        self.lbl_uptime.setStyleSheet("color:#3a1d7a; font-size:11px; font-weight:700;")
        cb = QVBoxLayout(); cb.setSpacing(0)
        cb.addWidget(self.lbl_uptime, 0, Qt.AlignRight)
        lay.addLayout(cb)
        self.pill = StatusPill(); lay.addWidget(self.pill)
        return h

    def _build_center(self):
        col = QWidget()
        lay = QVBoxLayout(col); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(12)
        self.view_cam = VideoView("CAMERA OFFLINE   -   press START")
        cam_card = titled_card("LIVE CAMERA   •   YOLOv8 + HAND TRACKING",
                               self.view_cam, C.CYAN)
        lay.addWidget(cam_card, 0, Qt.AlignTop)

        self.link_legend = LinkLegend()
        self.legend_card = titled_card("ROBOT LINK COLOUR KEY",
                                       self.link_legend, C.CYAN)
        self.legend_card.layout().setAlignment(self.link_legend, Qt.AlignTop)
        self.timeline = SafetyTimeline()
        self.timeline_card = titled_card("SAFETY MONITOR", self.timeline, C.VIOLET)
        lay.addWidget(self.legend_card)
        lay.addWidget(self.timeline_card)
        lay.addStretch(1)
        return col

    def _build_right_rail(self):
        rail = QWidget(); rail.setFixedWidth(480)
        lay = QVBoxLayout(rail); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(12)
        self.view_arm = VideoView("ROBOTIC ARM STANDBY", lock_aspect=False)
        self.arm_card = titled_card("ROBOTIC ARM   •   3D URDF MODEL",
                                    self.view_arm, C.VIOLET)
        lay.addWidget(self.arm_card, 1)

        chips = QHBoxLayout(); chips.setSpacing(10)
        self.chip_state = StatChip("Arm", C.GREEN)
        self.chip_obj = StatChip("Objects", C.AMBER)
        self.chip_hands = StatChip("Hands", C.CYAN)
        for c in (self.chip_state, self.chip_obj, self.chip_hands):
            chips.addWidget(c)
        lay.addLayout(chips, 0)
        return rail

    def _build_control_dock(self):
        dock = QFrame(); dock.setObjectName("dock"); dock.setFixedHeight(116)
        lay = QHBoxLayout(dock); lay.setContentsMargins(16, 12, 16, 12); lay.setSpacing(18)

        self.btn_start = QPushButton("START")
        self.btn_start.setObjectName("primary")
        self.btn_start.setFixedWidth(150)
        self.btn_start.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.btn_start.clicked.connect(self.toggle_monitor)
        lay.addWidget(self.btn_start)

        sliders = QVBoxLayout(); sliders.setSpacing(8)
        self.sld_conf, conf_box = self._slider_row(
            "YOLO CONFIDENCE", int(YOLO_CONF * 100), 10, 90, self._on_conf)
        conf_box.setVisible(False)
        self.sld_thr, thr_box = self._slider_row(
            "SAFETY DISTANCE", AREA_THRESHOLD, THRESHOLD_MIN, THRESHOLD_MAX,
            self._on_threshold, step=THRESHOLD_STEP, is_area=True)
        self.chk_audio = QCheckBox("Voice alerts (hello + red zone)")
        self.chk_audio.setChecked(VOICE_ALERTS_DEFAULT)
        self.chk_audio.toggled.connect(self._on_audio_toggled)
        sliders.addWidget(thr_box); sliders.addWidget(self.chk_audio); sliders.addStretch()
        sw = QWidget(); sw.setLayout(sliders); sw.setMinimumWidth(280)
        lay.addWidget(sw, 1)

        opts = QVBoxLayout(); opts.setSpacing(8)
        self.chk_landmarks = QCheckBox("Hand skeleton"); self.chk_landmarks.setChecked(True)
        self.chk_landmarks.toggled.connect(self.vision.set_landmarks)
        self.chk_flip = QCheckBox("Mirror camera"); self.chk_flip.setChecked(FLIP_VIEW)
        self.chk_flip.toggled.connect(self.vision.set_flip)
        self.chk_face = QCheckBox("Face ID"); self.chk_face.setChecked(True)
        self.chk_face.toggled.connect(self.vision.set_face)
        self.chk_objects = QCheckBox("Object detection"); self.chk_objects.setChecked(True)
        self.chk_objects.toggled.connect(self._on_objects_toggled)
        opts.addWidget(self.chk_landmarks); opts.addWidget(self.chk_flip)
        opts.addWidget(self.chk_face); opts.addWidget(self.chk_objects)
        ow = QWidget(); ow.setLayout(opts); lay.addWidget(ow)

        right = QVBoxLayout(); right.setSpacing(8)
        self.cmb_device = QComboBox(); self.cmb_device.addItems(["CUDA (GPU)", "CPU"])
        self.cmb_device.currentIndexChanged.connect(self._on_device)
        self.btn_snapshot = QPushButton("Snapshot")
        self.btn_snapshot.clicked.connect(self.take_snapshot)
        self.btn_snapshot.setEnabled(False)
        right.addWidget(self.cmb_device); right.addWidget(self.btn_snapshot)
        rw = QWidget(); rw.setLayout(right); rw.setFixedWidth(160); lay.addWidget(rw)

        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        self.log.setFixedWidth(320)
        lay.addWidget(self.log)
        return dock

    def _slider_row(self, name, val, lo, hi, cb, step=1, is_area=False):
        box = QWidget(); v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0); v.setSpacing(3)
        head = QHBoxLayout()
        lbl = QLabel(name); lbl.setStyleSheet(f"color:{C.MUTED}; font-size:10px; font-weight:700; letter-spacing:1px;")
        val_lbl = QLabel(); val_lbl.setStyleSheet(f"color:{C.CYAN}; font-weight:800;")
        head.addWidget(lbl); head.addStretch(); head.addWidget(val_lbl)
        s = QSlider(Qt.Horizontal); s.setRange(lo, hi); s.setValue(val)
        if step > 1:
            s.setSingleStep(step)

        def fmt(x): return f"{x/1000:.0f}k" if is_area else f"{x}%"
        val_lbl.setText(fmt(val))

        def on_change(x): val_lbl.setText(fmt(x)); cb(x)
        s.valueChanged.connect(on_change)
        v.addLayout(head); v.addWidget(s)
        return s, box

    def _wire_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Space), self, self.toggle_monitor)
        QShortcut(QKeySequence(Qt.Key_Q), self, self.close)
        QShortcut(QKeySequence(Qt.Key_Escape), self, self.close)
        QShortcut(QKeySequence(Qt.Key_Plus), self,
                  lambda: self.sld_thr.setValue(self.sld_thr.value() + THRESHOLD_STEP))
        QShortcut(QKeySequence(Qt.Key_Minus), self,
                  lambda: self.sld_thr.setValue(self.sld_thr.value() - THRESHOLD_STEP))
        QShortcut(QKeySequence(Qt.Key_S), self, self.take_snapshot)

    def _connect_workers(self):
        self.vision.frameReady.connect(self._on_frame)
        self.vision.status.connect(self._on_status)
        self.vision.failed.connect(self._on_failed)
        self.arm.armReady.connect(self._on_arm)
        self.arm.mode.connect(self._on_arm_mode)
        self.arm.legend.connect(self.link_legend.set_entries)
        self.view_cam.rightClicked.connect(self._toggle_estop)
        self.view_arm.rightClicked.connect(self._toggle_estop)
        tip = "Right-click to toggle EMERGENCY STOP"
        self.view_cam.setToolTip(tip); self.view_arm.setToolTip(tip)

    def toggle_monitor(self):
        self._stop_monitor() if self.vision.isRunning() else self._start_monitor()

    def _restyle(self, btn, name):
        btn.setObjectName(name)
        self.style().unpolish(btn); self.style().polish(btn)

    def _start_monitor(self):
        self._t_start = time.time()
        self._last_safe = None
        self._last_collision = False
        self._open_session_log()
        self.btn_start.setText("STOP"); self._restyle(self.btn_start, "danger")
        self.btn_snapshot.setEnabled(True)
        self.view_cam.set_placeholder("INITIALISING  •  loading YOLOv8 + MediaPipe ...")
        self.view_arm.set_placeholder("LOADING ROBOTIC ARM ...")
        self._log("Starting subsystems...")
        self._reset_estop()
        self._was_red = False
        self._last_seen.clear()
        self._log(f"Voice alerts: {self.audio.backend_name()} "
                  f"({'ON' if self.chk_audio.isChecked() else 'OFF'})")
        self.vision.start(); self.arm.start()

    def _stop_monitor(self):
        self._log("Stopping subsystems...")
        self.vision.stop(); self.vision.wait(2000)
        self.arm.stop(); self.arm.wait(2000)
        self.btn_start.setText("START"); self._restyle(self.btn_start, "primary")
        self.btn_snapshot.setEnabled(False)
        self.view_cam.clear(); self.view_arm.clear()
        self.audio.stop_buzzer(); self._was_red = False
        self.view_arm.stop_flash()
        self._last_seen.clear()
        self._reset_estop()
        self.pill.set_safe(True)
        self.chip_state.set("--", C.GREEN)
        self._close_session_log()

    def _on_frame(self, img, r: FrameResult):
        self.view_cam.set_image(img)
        self._maybe_voice(r)
        self._maybe_greet(r)
        safe = (r.state == "GREEN") and not self._estop
        self.pill.set_safe(safe); self.arm.set_stopped(not safe)

        if safe != self._last_safe:
            if self._last_safe is not None:
                if safe:
                    self._log("SAFETY  ->  SAFE (green zone, arm resumed)")
                elif self._estop:
                    self._log("SAFETY  ->  EMERGENCY STOP (manual latch)")
                else:
                    self._log("SAFETY  ->  STOP (red zone - hand too close, arm frozen)")
            self._last_safe = safe
        if r.collision and not self._last_collision:
            self._log(f"SAFETY  ->  COLLISION RISK ({r.objects} object(s) near a stopped arm)")
        self._last_collision = r.collision

        self.chip_state.set("RUN" if safe else ("ESTOP" if self._estop else "HOLD"),
                            C.GREEN if safe else C.RED)
        self.chip_obj.set(f"{r.objects} !" if r.collision else r.objects,
                          C.RED if r.collision else C.AMBER)
        self.chip_hands.set(r.hands, C.CYAN)

        self.timeline.push(safe)

    def _on_arm(self, img): self.view_arm.set_image(img)

    def _on_arm_mode(self, mode):
        label = ("ROBOTIC ARM   •   3D URDF MODEL" if mode == "3D URDF"
                 else "ROBOTIC ARM   •   2D KINEMATIC (3D unavailable)")
        self.arm_card.title_label.setText(label)
        self._log(f"Robotic arm renderer: {mode}")

    def _on_status(self, s):
        self._log(s)
        if any(k in s for k in ("Loading", "Opening", "dropped")):
            self.view_cam.set_placeholder(s.upper())

    def _on_failed(self, msg):
        self._log("ERROR  " + msg); self._stop_monitor()

    def _on_conf(self, v):      self.vision.set_conf(v / 100.0)
    def _on_threshold(self, v): self.vision.set_threshold(v)

    def _on_objects_toggled(self, on):
        self.vision.set_objects(on)
        self._log(f"Object detection {'enabled' if on else 'disabled'}")

    def _maybe_voice(self, r: FrameResult):
        """On *entering* the red zone, fire the alert: flash the camera feed red
        and sound the voice + buzzer (capped at RED_ALERT_SECS). As soon as the
        hand leaves the red zone, the buzzer and flashing are stopped early."""
        in_red = (r.state == "RED")
        if in_red and not self._was_red:
            self.audio.alert_red_zone(RED_ALERT_SECS)
            self.view_arm.start_flash(int(RED_ALERT_SECS * 1000))
        elif not in_red and self._was_red:
            self.audio.stop_buzzer()
            self.view_arm.stop_flash()
        self._was_red = in_red

    def _maybe_greet(self, r: FrameResult):
        """Say "hello <name>" when a known person appears. The same person is not
        greeted again until they've been out of frame for GREET_REENTRY_SECS, so
        brief recognition drop-outs don't cause repeated greetings.

        Emergency stop and the red-zone alert take priority: while either is
        active we don't greet, so the safety alert is never talked over."""
        if not r.identities or self._estop or self.audio.is_alerting():
            return
        now = time.monotonic()
        entered = []
        for name in r.identities:
            last = self._last_seen.get(name)
            if last is None or (now - last) > GREET_REENTRY_SECS:
                entered.append(name)
            self._last_seen[name] = now
        if entered:
            self.audio.say(". ".join(f"Hello {name}" for name in entered))
            self._log("Greeted: " + ", ".join(entered))

    def _toggle_estop(self):
        """Mouse right-click on a feed toggles the emergency-stop latch."""
        self._set_estop(not self._estop)

    def _set_estop(self, on: bool):
        self._estop = bool(on)
        self.view_cam.set_estop(self._estop)
        self.pill.set_estop(self._estop)
        if self._estop:
            self.arm.set_stopped(True)
            self.pill.set_safe(False)
            self.chip_state.set("ESTOP", C.RED)
            self._log("EMERGENCY STOP engaged  (mouse right-click)")
            self.audio.say(ESTOP_ON_PHRASE, priority=True)
        else:
            self._log("Emergency stop released  (mouse right-click)")
            self.audio.say(ESTOP_OFF_PHRASE, priority=True)
            if not self.vision.isRunning():
                self.pill.set_safe(True)
                self.chip_state.set("--", C.GREEN)

    def _reset_estop(self):
        self._estop = False
        self.view_cam.set_estop(False)
        self.pill.set_estop(False)

    def _on_audio_toggled(self, on: bool):
        self.audio.set_enabled(on)
        self._log(f"Voice alerts {'enabled' if on else 'disabled'} "
                  f"({self.audio.backend_name()})")

    def _on_device(self, idx):
        self.vision.device = "cuda" if idx == 0 else "cpu"
        self._log(f"Compute target -> {self.vision.device.upper()} (applies on next START)")

    def take_snapshot(self):
        path = self.vision.snapshot_path()
        self._log(f"Snapshot saved  ->  {path}" if path
                  else "Snapshot unavailable (no live frame)")

    def _update_clock(self):
        if self._t_start and self.vision.isRunning():
            el = int(time.time() - self._t_start)
            self.lbl_uptime.setText(f"UP {el//60:02d}:{el%60:02d}")
        else:
            self.lbl_uptime.setText("IDLE")

    def _open_session_log(self):
        self._close_session_log()
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self._log_path = os.path.join(LOG_DIR, f"session_{stamp}.log")
            self._log_file = open(self._log_path, "a", encoding="utf-8")
            self._log_file.write(
                f"===== SESSION START  {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            self._log_file.flush()
        except Exception:
            self._log_file = None
        if self._log_file is not None:
            self._log(f"Session log -> {self._log_path}")

    def _close_session_log(self):
        if self._log_file is not None:
            try:
                el = int(time.time() - self._t_start) if self._t_start else 0
                self._log_file.write(
                    f"===== SESSION END    {time.strftime('%Y-%m-%d %H:%M:%S')}"
                    f"  (duration {el//60:02d}:{el%60:02d}) =====\n\n")
                self._log_file.flush(); self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    def _log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self.log.appendPlainText(line)
        if self._log_file is not None:
            try:
                self._log_file.write(line + "\n"); self._log_file.flush()
            except Exception:
                pass

    def closeEvent(self, e):
        try:
            self.audio.shutdown()
            self.vision.stop(); self.vision.wait(2000)
            self.arm.stop(); self.arm.wait(2000)
            self._close_session_log()
        finally:
            e.accept()



def _apply_app_style(app):
    app.setStyleSheet(build_stylesheet())
    app.setFont(QFont("Segoe UI", 10))


def selftest():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication(sys.argv)
    _apply_app_style(app)
    win = MainWindow(); win.show()
    win.audio.set_enabled(False)

    dummy = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8); dummy[:] = (40, 30, 25)
    cv2.putText(dummy, "SELFTEST", (60, 240), FONT, 2.0, (200, 200, 200), 3)
    qimg = bgr_to_qimage(dummy)
    for i in range(30):
        r = FrameResult(state=("RED" if i % 7 == 0 else "GREEN"),
                        hands=i % 3, objects=i % 4, collision=(i % 7 == 0),
                        max_area=30_000 + i * 1500, fps=28 + (i % 5),
                        infer_ms=18 + (i % 6), device="cpu")
        win._on_frame(qimg, r)
    win._on_arm(bgr_to_qimage(render_arm_2d(1.0, False)))
    win._on_arm_mode("3D URDF")
    win._toggle_estop(); win._on_frame(qimg, r)
    win._toggle_estop()
    win.take_snapshot()
    print("[selftest] UI constructed + updated OK; widgets:",
          len(win.findChildren(QWidget)))
    QTimer.singleShot(0, app.quit); app.exec()
    print("[selftest] PASS")
    return 0


def main():
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    app = QApplication(sys.argv)
    app.setApplicationName("Safety Rated Monitor Stop")
    _apply_app_style(app)
    win = MainWindow(); win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

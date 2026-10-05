"""PHASE 17 - Camera-guided straight-line drawing for the 3-DOF cobot.

The camera finds the sheet of paper (corners P0..P3) and the red reference pin
that marks the robot's origin. The operator types a line in millimetres measured
on the sheet from P0. The panel maps that line onto the robot, plans the whole
pen path, checks every pose of it, and only then streams joint angles to the arm
while the 3-D view follows along.

Frames
    paper   mm from corner P0; x along P0->P1, y along P0->P3 (what is typed in)
    image   camera pixels; +x right, +y down
    robot   metres from the red pin; +x = image right, +y = image up

Where the accuracy comes from
    * paper -> image goes through a perspective transform fitted to all four
      detected corners, so a point typed at (width, height) lands on the far
      corner even when the sheet is seen slightly at an angle
    * sheet corners and the pin are averaged over recent frames, which takes out
      the frame-to-frame detection jitter
    * the pin is fixed to the table, so while the arm or a hand hides it its last
      averaged position is used - never a different origin
    * the whole path is planned before the arm moves: every IK answer is checked
      with forward kinematics and the joint limits, and joint jumps (IK branch
      flips) are refused, so a bad line is rejected instead of stopping half-drawn
    * every pen pose lies exactly on the straight line, with a smooth
      minimum-jerk speed profile instead of stop-and-go every 2 mm
    * the pen is lifted for travel moves, so it only touches the paper on the line
"""

import ctypes
import html
import os
import sys
import threading
import time
import traceback
from collections import deque

import cv2
import numpy as np

# Silence nanobind C++ cleanup warnings on shutdown
os.environ["NANOBIND_LEAK_WARNINGS"] = "0"

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
plt.ioff()  # Disable interactive popups to prevent extra "Figure 1" windows

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtCore import QLocale, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QAbstractSpinBox, QApplication, QDoubleSpinBox, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QMainWindow, QPlainTextEdit, QPushButton,
    QSizePolicy, QSplitter, QVBoxLayout, QWidget,
)

# Import local modules
from robot_math import robot, solve_ik, z_drawing
from serial_comm import SerialController
from camera_workspace import CameraWorkspace

# ==========================================
# 0. Configuration
# ==========================================
SERIAL_PORT = "COM3"
SERIAL_BAUD = 115200

HOME_Q = np.array([0.0, np.deg2rad(90), np.deg2rad(-90)])

TICK_S = 0.015              # time between two joint commands sent to the arm
LINE_SPEED_MM_S = 40.0      # peak pen speed along the line
PEN_SPEED_MM_S = 25.0       # peak speed while lowering / lifting the pen
JOINT_SPEED_DEG_S = 90.0    # peak joint speed for free travel moves
PEN_LIFT_M = 0.015          # pen clearance for travel moves (0 = never lift)
PEN_DOWN_TOL_M = 0.001      # the pen counts as touching below z_drawing + this

IK_TOL_MM = 1.0             # max IK position error accepted (checked with FK)
MAX_JOINT_STEP_DEG = 8.0    # a bigger jump between two commands = IK branch flip
MIN_LINE_MM = 0.5           # shorter "lines" are refused

PAPER_MAX_AGE_S = 1.0       # a paper detection older than this is not trusted
PAPER_HISTORY = 20          # detections averaged while the sheet is still
PAPER_STILL_PX = 3.0        # corner movement that counts as the sheet moving
PIN_HISTORY = 30            # red-pin detections averaged (the pin never moves)
PLOT_EPS_M = 0.0005         # smaller changes are invisible in the 3-D view: no redraw


class C:
    """Palette shared with COBOT_GUI.py."""
    BG = "#080b15"
    BG2 = "#0d1322"
    CARD = "#121a2e"
    CARD2 = "#0e1525"
    BORDER = "#283556"
    BORDER_SOFT = "#1b2440"
    TRACK = "#1d2742"
    TEXT = "#eef2ff"
    MUTED = "#8c96b6"
    FAINT = "#525d80"
    BLUE = "#3b82f6"
    CYAN = "#22d3ee"
    GREEN = "#22c55e"
    AMBER = "#f59e0b"
    PINK = "#ec4899"
    RED = "#ef4444"
    ORANGE = "#fb923c"


def bgr(hex_color):
    """'#rrggbb' -> OpenCV BGR tuple."""
    v = hex_color.lstrip("#")
    return int(v[4:6], 16), int(v[2:4], 16), int(v[0:2], 16)


def rgba(hex_color, alpha):
    """'#rrggbb' -> Qt stylesheet 'rgba(r, g, b, alpha)' (Qt reads 8-digit hex as #AARRGGBB)."""
    b, g, r = bgr(hex_color)
    return f"rgba({r}, {g}, {b}, {alpha})"


# ==========================================
# 1. Paper Pose (camera -> paper -> robot)
# ==========================================
class PaperPose:
    """One steadied measurement of the sheet and the pin, and the maps between frames.

    width_mm is the P0->P1 side and height_mm the P0->P3 side, so the operator's
    x/y always run along the same physical edges they are validated against.
    """

    def __init__(self, vertices_px, width_mm, height_mm, mm_per_px, pin_px):
        self.vertices_px = np.asarray(vertices_px, dtype=np.float64).reshape(4, 2)
        self.width_mm = float(width_mm)
        self.height_mm = float(height_mm)
        self.mm_per_px = float(mm_per_px)
        self.pin_px = np.asarray(pin_px, dtype=np.float64).reshape(2)

        # Perspective transform through all four corners: paper mm -> image px.
        paper_rect = np.array([[0.0, 0.0], [self.width_mm, 0.0],
                               [self.width_mm, self.height_mm], [0.0, self.height_mm]],
                              dtype=np.float32)
        self._paper_to_image = cv2.getPerspectiveTransform(
            paper_rect, self.vertices_px.astype(np.float32))

    @classmethod
    def build(cls, vertices_px, width_mm, height_mm, mm_per_px, pin_px):
        """A PaperPose, or None when the measurement cannot be trusted."""
        v = np.asarray(vertices_px, dtype=np.float64).reshape(4, 2)
        if not (np.all(np.isfinite(v)) and mm_per_px > 0
                and width_mm > 1.0 and height_mm > 1.0):
            return None
        quad = v.astype(np.float32)
        if cv2.contourArea(quad) < 100.0 or not cv2.isContourConvex(quad):
            return None

        # Keep width on the P0->P1 edge: if the reported size fits the detected
        # edges better the other way round, the two were handed over swapped.
        side_01 = (np.linalg.norm(v[1] - v[0]) + np.linalg.norm(v[2] - v[3])) / 2 * mm_per_px
        side_03 = (np.linalg.norm(v[3] - v[0]) + np.linalg.norm(v[2] - v[1])) / 2 * mm_per_px
        if (abs(width_mm - side_03) + abs(height_mm - side_01)
                < abs(width_mm - side_01) + abs(height_mm - side_03)):
            width_mm, height_mm = height_mm, width_mm

        try:
            return cls(v, width_mm, height_mm, mm_per_px, pin_px)
        except cv2.error:
            return None

    def contains(self, x_mm, y_mm):
        return 0.0 <= x_mm <= self.width_mm and 0.0 <= y_mm <= self.height_mm

    def paper_to_image(self, points_mm):
        pts = np.asarray(points_mm, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self._paper_to_image).reshape(-1, 2)

    def image_to_robot(self, points_px):
        d = np.asarray(points_px, dtype=np.float64).reshape(-1, 2) - self.pin_px
        # Image y grows downwards; robot y grows "up" the image.
        return np.column_stack([d[:, 0], -d[:, 1]]) * self.mm_per_px / 1000.0

    def paper_to_robot(self, points_mm):
        """Paper mm (from P0) -> robot XY metres (from the red pin)."""
        return self.image_to_robot(self.paper_to_image(points_mm))

    def outline_robot(self):
        return self.image_to_robot(self.vertices_px)


class WorkspaceTracker:
    """Steadies the camera's paper and pin measurements over recent frames.

    The pin is fixed to the table, so its position is the median of the last
    detections and it is remembered while it is hidden. The sheet can be moved,
    so its corners are only averaged while it holds still - any movement starts
    the average again, so the pose never lags a moved sheet.
    """

    def __init__(self):
        self._papers = deque(maxlen=PAPER_HISTORY)
        self._pins = deque(maxlen=PIN_HISTORY)
        self.paper_seen_at = None
        self.pin_seen_at = None

    def update(self, data, now):
        if not data:
            return
        try:
            vertices = np.asarray(data["vertices_px"], dtype=np.float64).reshape(4, 2)
            sample = (vertices, float(data["width_mm"]), float(data["height_mm"]),
                      float(data["scale_mm_per_px"]))
        except (KeyError, TypeError, ValueError):
            return

        if self._papers:
            reference = np.mean([s[0] for s in self._papers], axis=0)
            if np.max(np.linalg.norm(vertices - reference, axis=1)) > PAPER_STILL_PX:
                self._papers.clear()
        self._papers.append(sample)
        self.paper_seen_at = now

        red = data.get("red_px")
        if red is not None:
            try:
                pin = np.asarray(red, dtype=np.float64).ravel()[:2]   # (x, y[, extras])
            except (TypeError, ValueError):
                pin = None
            if pin is not None and pin.size == 2 and np.all(np.isfinite(pin)):
                self._pins.append(pin)
                self.pin_seen_at = now

    def paper_visible(self, now):
        return self.paper_seen_at is not None and now - self.paper_seen_at <= PAPER_MAX_AGE_S

    def pin_age(self, now):
        """Seconds since the pin was last seen, or None if it never was."""
        return None if self.pin_seen_at is None else now - self.pin_seen_at

    def pose(self, now):
        """(PaperPose, None) or (None, reason it cannot be used)."""
        if not self.paper_visible(now):
            return None, "the camera does not see the paper"
        if not self._pins:
            return None, "the red reference pin has not been seen yet"

        vertices = np.median([s[0] for s in self._papers], axis=0)
        width, height, scale = np.median([s[1:] for s in self._papers], axis=0)
        pin = np.median(self._pins, axis=0)
        pose = PaperPose.build(vertices, width, height, scale, pin)
        if pose is None:
            return None, "the detected paper outline is not a valid sheet"
        return pose, None


# ==========================================
# 2. Motion Planning
# ==========================================
class PlanError(Exception):
    """The motion cannot be carried out; nothing has been sent to the arm."""


def tick_count(distance, peak_speed):
    """Commands needed to cover distance with a min-jerk profile peaking at peak_speed."""
    # A minimum-jerk move peaks at 1.875x its average speed.
    duration = 1.875 * distance / peak_speed
    return max(2, int(np.ceil(duration / TICK_S)) + 1)


def min_jerk(n):
    """n path fractions from 0 to 1 that start and stop with zero speed and acceleration."""
    tau = np.linspace(0.0, 1.0, n)
    return tau ** 3 * (10.0 - 15.0 * tau + 6.0 * tau ** 2)


def fk_position(q):
    return np.asarray(robot.fkine(q).t, dtype=float).reshape(3)


def fmt_mm(p):
    return f"({p[0] * 1000:.1f}, {p[1] * 1000:.1f}, {p[2] * 1000:.1f}) mm"


def _joint_limits():
    try:
        qlim = np.asarray(robot.qlim, dtype=float)
    except Exception:       # robot model without usable limits
        return None
    return qlim if qlim.shape == (2, len(HOME_Q)) else None


JOINT_LIMITS = _joint_limits()


def ik_checked(p, q_seed, what):
    """Joint angles that put the pen at p (metres); raises PlanError otherwise.

    solve_ik's own flag is not the last word: the answer must lie inside the
    joint limits and, pushed back through forward kinematics, land within
    IK_TOL_MM of p.
    """
    p = np.asarray(p, dtype=float)
    q, ok = solve_ik(robot, float(p[0]), float(p[1]), float(p[2]), q_seed)
    if not ok or q is None:
        raise PlanError(f"{what} {fmt_mm(p)} is out of the arm's reach.")
    q = np.asarray(q, dtype=float).reshape(-1)
    if not np.all(np.isfinite(q)):
        raise PlanError(f"{what} {fmt_mm(p)}: IK returned invalid angles.")
    if JOINT_LIMITS is not None:
        outside = (q < JOINT_LIMITS[0] - 1e-6) | (q > JOINT_LIMITS[1] + 1e-6)
        if np.any(outside):
            j = int(np.argmax(outside))
            raise PlanError(f"{what} {fmt_mm(p)} needs joint {j + 1} at "
                            f"{np.rad2deg(q[j]):.1f} deg, outside its limits.")
    miss_mm = np.linalg.norm(fk_position(q) - p) * 1000.0
    if miss_mm > IK_TOL_MM:
        raise PlanError(f"{what} {fmt_mm(p)}: IK misses by {miss_mm:.2f} mm "
                        f"(limit IK_TOL_MM = {IK_TOL_MM} mm).")
    return q


def cartesian_path(p_from, p_to, q_seed, peak_speed_mm_s, what):
    """Joint waypoints that keep the pen exactly on the straight segment p_from -> p_to."""
    p_from = np.asarray(p_from, dtype=float)
    p_to = np.asarray(p_to, dtype=float)
    distance_mm = np.linalg.norm(p_to - p_from) * 1000.0
    max_step = np.deg2rad(MAX_JOINT_STEP_DEG)

    path = []
    q_prev = np.asarray(q_seed, dtype=float)
    for s in min_jerk(tick_count(distance_mm, peak_speed_mm_s)):
        p = p_from + s * (p_to - p_from)
        q = ik_checked(p, q_prev, what)
        if np.max(np.abs(q - q_prev)) > max_step:
            raise PlanError(f"{what}: the arm would have to flip its joints near {fmt_mm(p)} "
                            f"(IK branch change or singularity).")
        path.append(q)
        q_prev = q
    return np.array(path)


def joint_path(q_from, q_to):
    """Smooth joint-space move; same min-jerk profile as roboticstoolbox's jtraj."""
    q_from = np.asarray(q_from, dtype=float)
    q_to = np.asarray(q_to, dtype=float)
    span_deg = np.rad2deg(np.max(np.abs(q_to - q_from)))
    s = min_jerk(tick_count(span_deg, JOINT_SPEED_DEG_S))
    return q_from + s[:, None] * (q_to - q_from)


def lowest_z(path):
    return min(fk_position(q)[2] for q in path)


def windows_timer_resolution(enable):
    """Windows sleeps in ~15.6 ms steps by default; ask for 1 ms while streaming."""
    if sys.platform != "win32":
        return
    try:
        winmm = ctypes.WinDLL("winmm")
        (winmm.timeBeginPeriod if enable else winmm.timeEndPeriod)(1)
    except (OSError, AttributeError):
        pass


# ==========================================
# 3. Background Workers
# ==========================================
class TrajectoryWorker(QThread):
    """Plans a whole motion, checks it, then streams it to the arm.

    When the thread ends, self.completed says whether the motion ran to the end.
    """

    step_signal = pyqtSignal(object)    # joint angles just commanded
    log_signal = pyqtSignal(str, str)   # (text, level)

    def __init__(self, mode, q_curr, serial_ctrl=None, line_xy=None):
        super().__init__()
        self.mode = mode
        self.q_curr = np.asarray(q_curr, dtype=float).copy()
        self.serial_ctrl = serial_ctrl
        self.line_xy = line_xy      # ((x0, y0), (x1, y1)) robot metres
        self.completed = False

    def run(self):
        try:
            segments = self._plan_line() if self.mode == "line" else self._plan_home()
            windows_timer_resolution(True)
            try:
                self.completed = self._execute(segments)
            finally:
                windows_timer_resolution(False)
        except PlanError as exc:
            self.log_signal.emit(f"REJECTED: {exc} The arm was not moved.", "error")
        except Exception as exc:   # never leave the panel locked
            traceback.print_exc()
            self.log_signal.emit(f"Motion aborted: {exc!r}", "error")

    # --- planning ---
    def _plan_line(self):
        (x0, y0), (x1, y1) = self.line_xy
        start = np.array([x0, y0, z_drawing])
        end = np.array([x1, y1, z_drawing])
        lift = np.array([0.0, 0.0, PEN_LIFT_M])

        self.log_signal.emit("Planning and checking the pen path...", "info")
        segments = []
        q = self._lift_if_down(self.q_curr, segments)

        # Travel with the pen up, then lower it straight down onto the start.
        q_above = None
        if PEN_LIFT_M > 0:
            try:
                q_above = ik_checked(start + lift, q, "Point above the line start")
            except PlanError as exc:
                self.log_signal.emit(f"{exc} Approaching with the pen down.", "warn")

        if q_above is not None:
            segments.append(("Moving above line start", self._travel(q, q_above)))
            lower = cartesian_path(start + lift, start, q_above, PEN_SPEED_MM_S, "Pen-down move")
            segments.append(("Lowering pen", lower))
            q = lower[-1]
        else:
            q_start = ik_checked(start, q, "Line start")
            segments.append(("Moving to line start", joint_path(q, q_start)))
            q = q_start

        draw = cartesian_path(start, end, q, LINE_SPEED_MM_S, "Line point")
        segments.append(("Drawing line", draw))

        if PEN_LIFT_M > 0:
            try:
                up = cartesian_path(end, end + lift, draw[-1], PEN_SPEED_MM_S, "Pen-up move")
                segments.append(("Lifting pen", up))
            except PlanError as exc:
                self.log_signal.emit(f"{exc} The pen stays down at the line end.", "warn")

        commands = sum(len(path) - 1 for _, path in segments)
        self.log_signal.emit(
            f"Path OK: {len(draw)} poses on the line, {commands} commands, "
            f"about {commands * TICK_S:.1f} s.", "info")
        return segments

    def _plan_home(self):
        segments = []
        q = self._lift_if_down(self.q_curr, segments)
        segments.append(("Homing robot", joint_path(q, HOME_Q)))
        return segments

    def _lift_if_down(self, q, segments):
        """If the pen is on the paper (e.g. after a stop), raise it before travelling."""
        if PEN_LIFT_M <= 0:
            return q
        p = fk_position(q)
        if p[2] > z_drawing + PEN_LIFT_M / 2:
            return q
        target = np.array([p[0], p[1], z_drawing + PEN_LIFT_M])
        try:
            path = cartesian_path(p, target, q, PEN_SPEED_MM_S, "Pen-up move")
        except PlanError as exc:
            self.log_signal.emit(f"{exc} Travelling without lifting the pen.", "warn")
            return q
        segments.append(("Lifting pen", path))
        return path[-1]

    def _travel(self, q_from, q_to):
        """Pen-up joint move that must not touch the paper; detours via home if needed."""
        clearance = z_drawing + PEN_DOWN_TOL_M
        path = joint_path(q_from, q_to)
        if fk_position(q_from)[2] < clearance:
            return path     # the pen could not be lifted (already warned about)
        if lowest_z(path) >= clearance:
            return path
        detour = np.vstack([joint_path(q_from, HOME_Q), joint_path(HOME_Q, q_to)[1:]])
        if lowest_z(detour) >= clearance:
            self.log_signal.emit("Direct travel would touch the paper - going via home.", "warn")
            return detour
        raise PlanError("Every travel move to the line start would touch the paper.")

    # --- execution ---
    def _execute(self, segments):
        for label, path in segments:
            self.log_signal.emit(f"{label}...", "info")
            t0 = time.perf_counter()
            # path[0] is where the previous segment already left the arm.
            for i, q_step in enumerate(path[1:], start=1):
                if self.isInterruptionRequested():
                    self.log_signal.emit("STOPPED by operator - the arm holds its pose.", "warn")
                    return False
                if self.serial_ctrl is not None:
                    self.serial_ctrl.send_angles(q_step)
                self.step_signal.emit(q_step.copy())
                # Absolute deadlines: IK/serial time does not stretch the move.
                delay = t0 + i * TICK_S - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
        return True


class CameraWorker(QThread):
    """Owns the camera: opens, reads and closes it on this one thread.

    The blocking frame grab stays off the GUI thread, and only the newest frame
    is kept, so the window never falls behind the camera.
    """

    error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._latest = None     # (seq, frame, data)

    def latest(self):
        with self._lock:
            return self._latest

    def run(self):
        try:
            camera = CameraWorkspace()
        except Exception as exc:
            traceback.print_exc()
            self.error_signal.emit(f"Camera could not start: {exc}")
            return

        seq = 0
        try:
            while not self.isInterruptionRequested():
                frame, data = camera.read_workspace()
                if frame is None:
                    self.msleep(5)
                    continue
                seq += 1
                with self._lock:
                    self._latest = (seq, frame, data)
        except Exception as exc:
            traceback.print_exc()
            self.error_signal.emit(f"Camera stopped: {exc}")
        finally:
            camera.close()


# ==========================================
# 4. UI Building Blocks
# ==========================================
STYLE = f"""
QMainWindow, QWidget#root {{ background: {C.BG}; }}
QWidget {{ color: {C.TEXT}; font-family: 'Segoe UI'; font-size: 10pt; }}
QToolTip {{ background: {C.CARD}; color: {C.TEXT}; border: 1px solid {C.BORDER}; padding: 4px; }}

QFrame#header {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1a1f4a, stop:1 {C.CARD2});
    border: 1px solid {C.BORDER}; border-radius: 12px;
}}
QLabel#title {{ font-size: 15pt; font-weight: 800; letter-spacing: 1px; }}
QLabel#subtitle {{ color: {C.MUTED}; font-size: 9pt; }}
QLabel#pill {{ border-radius: 13px; padding: 4px 16px; font-weight: 800; font-size: 9pt; }}

QFrame#card {{ background: {C.CARD}; border: 1px solid {C.BORDER_SOFT}; border-radius: 12px; }}
QLabel#cardTitle {{ color: {C.MUTED}; font-size: 8pt; font-weight: 700; letter-spacing: 2px; }}
QLabel#hint {{ color: {C.FAINT}; font-size: 8pt; }}
QLabel#axisHead {{ color: {C.FAINT}; font-size: 8pt; font-weight: 700; }}
QLabel#rowHead {{ color: {C.MUTED}; font-weight: 600; }}
QLabel#chipName {{ color: {C.FAINT}; font-size: 7pt; font-weight: 700; letter-spacing: 1px; }}
QLabel#metric {{ color: {C.MUTED}; font-size: 9pt; }}
QLabel#telemetry {{ color: {C.MUTED}; font-family: Consolas, 'Cascadia Mono', monospace; font-size: 9pt; }}

QDoubleSpinBox {{
    background: {C.CARD2}; border: 1px solid {C.BORDER}; border-radius: 8px;
    padding: 6px 8px; font-size: 11pt; font-weight: 600;
    selection-background-color: {C.BLUE};
}}
QDoubleSpinBox:focus {{ border-color: {C.CYAN}; }}
QDoubleSpinBox:disabled {{ color: {C.FAINT}; border-color: {C.BORDER_SOFT}; }}

QPushButton {{ border-radius: 10px; padding: 10px 14px; font-weight: 700; }}
QPushButton#primary {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10b981, stop:1 {C.CYAN});
    color: #05231a; border: none; font-size: 11pt;
}}
QPushButton#primary:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #14d39b, stop:1 #4be3f7);
}}
QPushButton#secondary {{ background: {C.CARD2}; border: 1px solid {C.BORDER}; color: {C.TEXT}; }}
QPushButton#secondary:hover {{ border-color: {C.BLUE}; }}
QPushButton#danger {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {C.RED}, stop:1 {C.PINK});
    color: #2b0710; border: none;
}}
QPushButton#primary:disabled, QPushButton#secondary:disabled, QPushButton#danger:disabled {{
    background: {C.TRACK}; color: {C.FAINT}; border: none;
}}
QPushButton#tool {{
    background: transparent; border: 1px solid {C.BORDER}; border-radius: 7px;
    padding: 3px 10px; font-size: 8pt; color: {C.MUTED};
}}
QPushButton#tool:hover {{ border-color: {C.BLUE}; color: {C.TEXT}; }}
QPushButton#tool:checked {{ background: {C.BLUE}; border-color: {C.BLUE}; color: white; }}
QPushButton#tool:disabled {{ color: {C.FAINT}; border-color: {C.BORDER_SOFT}; }}

QPlainTextEdit#log {{
    background: {C.BG2}; border: 1px solid {C.BORDER_SOFT}; border-radius: 8px;
    font-family: Consolas, 'Cascadia Mono', monospace; font-size: 9pt; padding: 4px;
}}
QLabel#video {{ background: transparent; color: {C.FAINT}; }}
QSplitter::handle {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {C.BORDER}; border-radius: 4px; min-height: 24px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
"""

LEVEL_COLORS = {"info": C.TEXT, "ok": C.GREEN, "warn": C.AMBER, "error": C.RED}
TONE_COLORS = {"ok": C.GREEN, "warn": C.AMBER, "bad": C.RED, "idle": C.FAINT}

# Robot state -> (pill text, pill colour)
STATES = {
    "idle": ("READY", C.GREEN),
    "line": ("DRAWING", C.CYAN),
    "home": ("HOMING", C.BLUE),
    "stopping": ("STOPPING", C.RED),
}


def card(title, tools=()):
    """A rounded panel with a small caps title; returns (frame, body layout)."""
    frame = QFrame(objectName="card")
    outer = QVBoxLayout(frame)
    outer.setContentsMargins(14, 12, 14, 14)
    outer.setSpacing(10)
    head = QHBoxLayout()
    head.addWidget(QLabel(title, objectName="cardTitle"))
    head.addStretch(1)
    for tool in tools:
        head.addWidget(tool)
    outer.addLayout(head)
    return frame, outer


class StatusChip(QWidget):
    """Small name/value pair with a coloured dot, e.g. 'RED PIN  ● Visible'."""

    def __init__(self, name):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(QLabel(name, objectName="chipName"))
        self.value = QLabel()
        self.value.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.value)
        self._shown = None
        self.set("idle", "-")

    def set(self, tone, text):
        if self._shown == (tone, text):
            return
        self._shown = (tone, text)
        self.value.setText(f'<span style="color:{TONE_COLORS[tone]}">&#9679;</span>&nbsp;'
                           f'{html.escape(text)}')


class VideoView(QLabel):
    """Shows camera frames scaled to fit while keeping their aspect ratio."""

    def __init__(self):
        super().__init__("Starting camera...", objectName="video")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._pixmap = None

    def set_frame(self, rgb):
        h, w, _ = rgb.shape
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(image)   # copies, so rgb may be freed
        self._rescale()

    def show_message(self, text):
        self._pixmap = None     # so a resize cannot bring back a stale frame
        self.setText(text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self):
        if self._pixmap is not None:
            self.setPixmap(self._pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))


def to_fixed(points_px):
    """Pixel points -> OpenCV drawing points with 4 bits of sub-pixel precision."""
    return [(int(round(x * 16)), int(round(y * 16)))
            for x, y in np.asarray(points_px, dtype=float).reshape(-1, 2)]


def draw_overlay(frame, pose, start_mm, end_mm, fits):
    """Camera frame with the sheet, P0's axes, the pin and the requested line drawn in."""
    view = frame.copy()
    if pose is None:
        return view
    aa, sub, font = cv2.LINE_AA, 4, cv2.FONT_HERSHEY_SIMPLEX

    quad = np.array(to_fixed(pose.vertices_px), dtype=np.int32)
    cv2.polylines(view, [quad], True, bgr(C.GREEN), 1, aa, sub)

    axis_mm = max(10.0, 0.25 * min(pose.width_mm, pose.height_mm))
    p0, px, py = to_fixed(pose.paper_to_image([[0, 0], [axis_mm, 0], [0, axis_mm]]))
    for tip, label in ((px, "x"), (py, "y")):
        cv2.arrowedLine(view, p0, tip, bgr(C.ORANGE), 2, aa, sub, 0.2)
        cv2.putText(view, label, (tip[0] // 16 + 5, tip[1] // 16 + 5), font, 0.5,
                    bgr(C.ORANGE), 2, aa)
    cv2.putText(view, "P0", (p0[0] // 16 - 28, p0[1] // 16 - 8), font, 0.5,
                bgr(C.ORANGE), 2, aa)

    pin = to_fixed([pose.pin_px])[0]
    cv2.drawMarker(view, (pin[0] // 16, pin[1] // 16), bgr(C.CYAN),
                   cv2.MARKER_CROSS, 18, 2, aa)

    color = bgr(C.GREEN if fits else C.RED)
    a, b = to_fixed(pose.paper_to_image([start_mm, end_mm]))
    cv2.line(view, a, b, color, 2, aa, sub)
    cv2.circle(view, a, 5 * 16, color, -1, aa, sub)     # start: filled
    cv2.circle(view, b, 5 * 16, color, 2, aa, sub)      # end: ring
    return view


# ==========================================
# 5. Main GUI Control Panel
# ==========================================
class CobotControlPanel(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3-DOF Cobot - Camera Line Drawing")
        self.resize(1480, 860)

        # Robot & workspace state
        self.q_curr = HOME_Q.copy()
        self.tracker = WorkspaceTracker()
        self.pose = None            # latest steadied paper pose
        self.locked_pose = None     # pose the running line was planned with
        self.locked_line = None     # ((sx, sy), (ex, ey)) paper mm of that line
        self.worker = None
        self.state = "idle"
        self.trace = deque(maxlen=20000)   # pen-down tip positions; NaN rows break the line
        self._pen_was_down = False
        self._trace_dirty = False
        self._plot_dirty = True
        self._shown_outline = None
        self._shown_target = None
        self._frame_seq = 0
        self._frame_times = deque(maxlen=30)
        self._camera_error = None

        self._build_ui()
        self.update_arm_geometry(self.q_curr)
        self._set_state("idle")
        self._update_line_feedback()

        self.log("Control panel initialised. Assuming the arm starts at the home pose.")
        self.serial_ctrl = self._open_serial()
        self.camera_thread = CameraWorker()
        self.camera_thread.error_signal.connect(self._on_camera_error)
        self.camera_thread.start()
        self.log("Camera starting. Coordinates are mm on the sheet, measured from P0.")

        # The GUI picks up the newest camera frame and redraws the 3-D view on
        # timers, so neither can queue up behind a slow frame.
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.update_camera_workspace)
        self.camera_timer.start(30)
        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self._render_plot)
        self.render_timer.start(40)

    # --- layout ---
    def _build_ui(self):
        root = QWidget(objectName="root")
        self.setCentralWidget(root)
        page = QVBoxLayout(root)
        page.setContentsMargins(14, 14, 14, 14)
        page.setSpacing(12)
        page.addWidget(self._build_header())

        left = QWidget()
        left.setMinimumWidth(360)
        left.setMaximumWidth(520)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)
        left_layout.addWidget(self._build_line_card())
        left_layout.addWidget(self._build_status_card())
        left_layout.addWidget(self._build_log_card(), 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(12)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(left)
        splitter.addWidget(self._build_camera_card())
        splitter.addWidget(self._build_plot_card())
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([380, 560, 520])
        page.addWidget(splitter, 1)

        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self.start_line)
        QShortcut(QKeySequence("Ctrl+Enter"), self, activated=self.start_line)
        QShortcut(QKeySequence("Ctrl+H"), self, activated=self.start_home)
        QShortcut(QKeySequence("Esc"), self, activated=self.stop_motion)

    def _build_header(self):
        header = QFrame(objectName="header")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(18, 12, 18, 12)
        text = QVBoxLayout()
        text.setSpacing(0)
        text.addWidget(QLabel("CAMERA LINE DRAWING", objectName="title"))
        text.addWidget(QLabel("3-DOF cobot  ·  straight lines placed on the camera-detected "
                              "sheet  ·  robot origin = red pin", objectName="subtitle"))
        layout.addLayout(text, 1)
        self.pill = QLabel(objectName="pill")
        self.pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pill.setMinimumWidth(120)
        layout.addWidget(self.pill)
        return header

    def _build_line_card(self):
        self.btn_swap = QPushButton("⇄ Swap", objectName="tool")
        self.btn_swap.setToolTip("Swap the start and end points")
        self.btn_swap.clicked.connect(self._swap_ends)
        frame, layout = card("LINE ON PAPER", (self.btn_swap,))
        layout.addWidget(QLabel("Millimetres on the sheet, measured from corner P0.",
                                objectName="hint"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.addWidget(QLabel("X  (P0 → P1)", objectName="axisHead"), 0, 1)
        grid.addWidget(QLabel("Y  (P0 → P3)", objectName="axisHead"), 0, 2)
        self.input_start_x = self._mm_spin(10.0)
        self.input_start_y = self._mm_spin(10.0)
        self.input_end_x = self._mm_spin(50.0)
        self.input_end_y = self._mm_spin(50.0)
        for row, (name, sx, sy) in enumerate(
                (("Start", self.input_start_x, self.input_start_y),
                 ("End", self.input_end_x, self.input_end_y)), start=1):
            grid.addWidget(QLabel(name, objectName="rowHead"), row, 0)
            grid.addWidget(sx, row, 1)
            grid.addWidget(sy, row, 2)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        layout.addLayout(grid)

        self.line_metric = QLabel(objectName="metric")
        self.line_check = QLabel()
        self.line_check.setWordWrap(True)
        self.line_robot = QLabel(objectName="hint")
        for w in (self.line_metric, self.line_check, self.line_robot):
            layout.addWidget(w)

        self.btn_move = QPushButton("Draw line", objectName="primary")
        self.btn_move.setToolTip("Plan, check and draw the line  (Ctrl+Enter)")
        self.btn_move.clicked.connect(self.start_line)
        self.btn_home = QPushButton("Home", objectName="secondary")
        self.btn_home.setToolTip("Lift the pen and return to the home pose  (Ctrl+H)")
        self.btn_home.clicked.connect(self.start_home)
        self.btn_stop = QPushButton("STOP", objectName="danger")
        self.btn_stop.setToolTip("Stop sending commands; the arm holds its pose  (Esc)")
        self.btn_stop.clicked.connect(self.stop_motion)
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addWidget(self.btn_move, 2)
        buttons.addWidget(self.btn_home, 1)
        buttons.addWidget(self.btn_stop, 1)
        layout.addLayout(buttons)
        layout.addWidget(QLabel("Ctrl+Enter draw  ·  Ctrl+H home  ·  Esc stop", objectName="hint"))
        return frame

    def _mm_spin(self, value):
        spin = QDoubleSpinBox()
        spin.setLocale(QLocale.c())     # '.' decimals and Latin digits on any Windows locale
        spin.setRange(0.0, 1000.0)
        spin.setDecimals(1)
        spin.setSingleStep(1.0)
        spin.setSuffix(" mm")
        spin.setValue(value)
        spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        spin.valueChanged.connect(self._update_line_feedback)
        return spin

    def _build_status_card(self):
        frame, layout = card("WORKSPACE")
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(10)
        self.chip_camera = StatusChip("CAMERA")
        self.chip_paper = StatusChip("PAPER")
        self.chip_pin = StatusChip("RED PIN (ORIGIN)")
        self.chip_serial = StatusChip("SERIAL")
        for i, chip in enumerate((self.chip_camera, self.chip_paper,
                                  self.chip_pin, self.chip_serial)):
            grid.addWidget(chip, i // 2, i % 2)
        layout.addLayout(grid)
        self.telemetry = QLabel(objectName="telemetry")
        layout.addWidget(self.telemetry)
        return frame

    def _build_log_card(self):
        clear = QPushButton("Clear", objectName="tool")
        frame, layout = card("SYSTEM LOG", (clear,))
        self.console = QPlainTextEdit(objectName="log")
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(500)
        clear.clicked.connect(self.console.clear)
        layout.addWidget(self.console, 1)
        return frame

    def _build_camera_card(self):
        self.fps_label = QLabel(objectName="hint")
        frame, layout = card("CAMERA", (self.fps_label,))
        self.camera_label = VideoView()
        layout.addWidget(self.camera_label, 1)
        return frame

    def _build_plot_card(self):
        top = QPushButton("Top view", objectName="tool")
        top.setCheckable(True)
        top.toggled.connect(self._set_top_view)
        clear = QPushButton("Clear trace", objectName="tool")
        clear.clicked.connect(self._clear_trace)
        frame, layout = card("ROBOT WORKSPACE", (top, clear))

        self.fig = Figure(figsize=(6, 6), facecolor=C.CARD)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumSize(320, 320)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.subplots_adjust(left=0.0, right=1.0, bottom=0.12, top=1.0)
        self._style_axes()

        ax = self.ax
        self.paper_line, = ax.plot([], [], [], "-", lw=1.8, color=C.GREEN, label="Paper (camera)")
        self.p0_point, = ax.plot([], [], [], "o", ms=7, color=C.ORANGE, linestyle="None",
                                 label="P0")
        self.red_pin_point, = ax.plot([0.0], [0.0], [z_drawing], "X", ms=10, color=C.RED,
                                      linestyle="None", label="Red pin (origin)")
        self.target_line, = ax.plot([], [], [], "--", lw=2, color=C.AMBER, label="Planned line")
        self.trace_line, = ax.plot([], [], [], "-", lw=2.5, color=C.PINK, label="Pen trace")
        self.arm_line, = ax.plot([], [], [], "-", lw=5, color=C.CYAN,
                                 solid_capstyle="round", label="Arm links")
        self.joint_points, = ax.plot([], [], [], "o", ms=7, color=C.TEXT,
                                     markeredgecolor=C.BLUE, linestyle="None")
        self.ee_point, = ax.plot([], [], [], "o", ms=9, color=C.PINK, linestyle="None",
                                 label="Pen tip")
        # Legend under the plot, so it never covers the arm or the sheet.
        self.fig.legend(loc="lower center", ncol=4, fontsize=8, frameon=False,
                        labelcolor=C.MUTED, handlelength=1.6, columnspacing=1.2)

        layout.addWidget(self.canvas, 1)
        return frame

    def _style_axes(self):
        ax = self.ax
        ax.set_facecolor(C.CARD)
        ax.set_xlim([-0.25, 0.25])
        ax.set_ylim([-0.25, 0.25])
        ax.set_zlim([0.0, 0.3])
        ax.set_box_aspect((0.5, 0.5, 0.3), zoom=1.1)   # true proportions: the arm is not stretched
        ax.view_init(elev=28, azim=-60)
        for axis, name in ((ax.xaxis, "X (m)"), (ax.yaxis, "Y (m)"), (ax.zaxis, "Z (m)")):
            axis.set_pane_color((1.0, 1.0, 1.0, 0.025))
            axis.pane.set_alpha(None)       # else the 0.025 alpha also hides the edge
            axis.pane.set_edgecolor(C.BORDER_SOFT)
            axis.line.set_color(C.BORDER)
            axis._axinfo["grid"].update(color=C.BORDER_SOFT, linewidth=0.6)
            axis.set_label_text(name, color=C.MUTED, fontsize=8)
        ax.tick_params(colors=C.FAINT, labelsize=7)

    # --- logging & state ---
    def log(self, text, level="info"):
        stamp = time.strftime("%H:%M:%S")
        color = LEVEL_COLORS.get(level, C.TEXT)
        self.console.appendHtml(
            f'<span style="color:{C.FAINT}">{stamp}</span>&nbsp;&nbsp;'
            f'<span style="color:{color}">{html.escape(text)}</span>')

    def _set_state(self, state):
        self.state = state
        text, color = STATES[state]
        self.pill.setText(text)
        self.pill.setStyleSheet(f"background:{rgba(color, 40)}; color:{color}; "
                                f"border:1px solid {color};")

        idle = state == "idle"
        self.btn_move.setEnabled(idle)
        self.btn_home.setEnabled(idle)
        self.btn_swap.setEnabled(idle)
        self.btn_stop.setEnabled(not idle)
        for spin in self._spins():
            spin.setEnabled(idle)

    def _busy(self):
        return self.worker is not None and self.worker.isRunning()

    def _open_serial(self):
        try:
            ctrl = SerialController(comPort=SERIAL_PORT, baudRate=SERIAL_BAUD, useHardware=True)
        except Exception as exc:
            self.log(f"Serial {SERIAL_PORT} unavailable ({exc}) - simulation only.", "warn")
            self.chip_serial.set("warn", "Simulation only")
            return None
        self.chip_serial.set("ok", f"{SERIAL_PORT} · {SERIAL_BAUD}")
        return ctrl

    # --- line input ---
    def _spins(self):
        return (self.input_start_x, self.input_start_y, self.input_end_x, self.input_end_y)

    def _line_mm(self):
        sx, sy, ex, ey = (spin.value() for spin in self._spins())
        return (sx, sy), (ex, ey)

    def _swap_ends(self):
        (sx, sy), (ex, ey) = self._line_mm()
        for spin, value in zip(self._spins(), (ex, ey, sx, sy)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self._update_line_feedback()

    def _check_line(self, pose, start, end):
        """(ok, message) for a line in paper mm against a paper pose."""
        if pose is None:
            _, reason = self.tracker.pose(time.monotonic())
            return False, f"Cannot place the line: {reason}."
        outside = [name for name, (x, y) in (("start", start), ("end", end))
                   if not pose.contains(x, y)]
        size = f"{pose.width_mm:.1f} × {pose.height_mm:.1f} mm sheet"
        if outside:
            return False, f"The {' and '.join(outside)} point is outside the {size}."
        if np.hypot(end[0] - start[0], end[1] - start[1]) < MIN_LINE_MM:
            return False, "Start and end are the same point."
        # A sheet is convex: both ends on it means the whole line is on it.
        return True, f"Fits on the {size}."

    def _update_line_feedback(self):
        start, end = self._line_mm()
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = np.hypot(dx, dy)
        self.line_metric.setText(
            f"Length {length:.1f} mm  ·  angle {np.degrees(np.arctan2(dy, dx)):.1f}°  ·  "
            f"draws in ~{1.875 * length / LINE_SPEED_MM_S:.1f} s")

        ok, message = self._check_line(self.pose, start, end)
        color = C.GREEN if ok else (C.RED if self.pose is not None else C.AMBER)
        self.line_check.setText(f'<span style="color:{color}">{"✔" if ok else "✖"}&nbsp; '
                                f'{html.escape(message)}</span>')

        if self.pose is not None:
            a, b = self.pose.paper_to_robot([start, end]) * 1000.0
            self.line_robot.setText(f"Robot frame: ({a[0]:.1f}, {a[1]:.1f}) → "
                                    f"({b[0]:.1f}, {b[1]:.1f}) mm from the red pin")
        else:
            self.line_robot.setText("Robot frame: -")

        if not self._busy():
            self._show_target(self.pose, start, end, ok)

    def _show_target(self, pose, start, end, ok):
        xy = None if pose is None else pose.paper_to_robot([start, end])
        shown = self._shown_target
        if xy is None and shown is None:
            return
        if (xy is not None and shown is not None and shown[1] == ok
                and np.allclose(xy, shown[0], atol=PLOT_EPS_M)):
            return
        self._shown_target = None if xy is None else (xy, ok)
        if xy is None:
            self.target_line.set_data_3d([], [], [])
        else:
            self.target_line.set_data_3d(xy[:, 0], xy[:, 1], [z_drawing, z_drawing])
            self.target_line.set_color(C.AMBER if ok else C.RED)
        self._plot_dirty = True

    # --- 3-D view ---
    def update_arm_geometry(self, q):
        """Forward kinematics -> arm links, joints and pen tip in the 3-D view."""
        coords = np.array([T.t for T in robot.fkine_all(q)])
        joints = coords[:-1]
        # The pen tip comes from fkine, like the planner, so a tool offset set in
        # robot_math is drawn as one more link instead of being ignored.
        tip = fk_position(q)
        if not np.allclose(tip, coords[-1], atol=1e-6):
            joints = coords
            coords = np.vstack([coords, tip])
        self.arm_line.set_data_3d(coords[:, 0], coords[:, 1], coords[:, 2])
        self.joint_points.set_data_3d(joints[:, 0], joints[:, 1], joints[:, 2])
        self.ee_point.set_data_3d([tip[0]], [tip[1]], [tip[2]])

        pen_down = tip[2] <= z_drawing + PEN_DOWN_TOL_M
        if pen_down:
            self.trace.append(tip)
            self._trace_dirty = True
        elif self._pen_was_down:
            self.trace.append(np.full(3, np.nan))
            self._trace_dirty = True
        self._pen_was_down = pen_down

        joints = ", ".join(f"{a:.1f}" for a in np.rad2deg(q))
        self.telemetry.setText(f"Joints ({joints})°\nPen    {fmt_mm(tip)}"
                               f"{'  · on paper' if pen_down else ''}")
        self._plot_dirty = True

    def _update_paper_plot(self, pose):
        outline = None if pose is None else pose.outline_robot()
        shown = self._shown_outline
        if outline is None and shown is None:
            return
        if outline is not None and shown is not None and np.allclose(outline, shown, atol=PLOT_EPS_M):
            return
        self._shown_outline = outline
        if outline is None:
            self.paper_line.set_data_3d([], [], [])
            self.p0_point.set_data_3d([], [], [])
        else:
            closed = np.vstack([outline, outline[:1]])
            self.paper_line.set_data_3d(closed[:, 0], closed[:, 1], np.full(5, z_drawing))
            self.p0_point.set_data_3d([outline[0, 0]], [outline[0, 1]], [z_drawing])
        self._plot_dirty = True

    def _set_top_view(self, top):
        if top:     # straight down, orthographic: XY distances read true
            self.ax.view_init(elev=90, azim=-90)
            self.ax.set_proj_type("ortho")
            self.ax.set_box_aspect((0.5, 0.5, 0.3), zoom=1.3)
        else:
            self.ax.view_init(elev=28, azim=-60)
            self.ax.set_proj_type("persp")
            self.ax.set_box_aspect((0.5, 0.5, 0.3), zoom=1.1)
        self._plot_dirty = True

    def _clear_trace(self):
        self.trace.clear()
        self._pen_was_down = False
        self._trace_dirty = True
        self._plot_dirty = True

    def _render_plot(self):
        if self._trace_dirty:     # rebuilt here, at render rate, not on every step
            self._trace_dirty = False
            t = np.array(self.trace) if self.trace else np.empty((0, 3))
            self.trace_line.set_data_3d(t[:, 0], t[:, 1], t[:, 2])
            self._plot_dirty = True
        if self._plot_dirty:
            self._plot_dirty = False
            self.canvas.draw_idle()

    # --- camera ---
    def update_camera_workspace(self):
        """Take the newest camera frame: steady the pose, refresh views and status."""
        now = time.monotonic()
        latest = self.camera_thread.latest()
        if latest is not None and latest[0] != self._frame_seq:
            self._frame_seq, frame, data = latest
            self._frame_times.append(now)
            self.tracker.update(data, now)
            self.pose, _ = self.tracker.pose(now)

            if self._busy() and self.locked_pose is not None:
                # Show the sheet the running line was planned on, not a pose the
                # arm itself may be disturbing by covering the paper.
                shown, (start, end), fits = self.locked_pose, self.locked_line, True
            else:
                shown = self.pose
                start, end = self._line_mm()
                fits, _ = self._check_line(shown, start, end)
                if not self._busy():
                    self._update_paper_plot(shown)
                    self._update_line_feedback()
            view = draw_overlay(frame, shown, start, end, fits)
            self.camera_label.set_frame(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
        elif self.pose is not None and not self.tracker.paper_visible(now):
            # No new detections for a while: stop showing a stale sheet.
            self.pose = None
            if not self._busy():
                self._update_paper_plot(None)
                self._update_line_feedback()

        self._refresh_status(now)

    def _refresh_status(self, now):
        if self._camera_error:
            self.chip_camera.set("bad", "Error")
            self.fps_label.setText("")
        elif not self._frame_times:
            self.chip_camera.set("idle", "Starting...")
        elif now - self._frame_times[-1] > 1.0:
            self.chip_camera.set("warn", "No frames")
            self.fps_label.setText("")
        else:
            span = self._frame_times[-1] - self._frame_times[0]
            fps = (len(self._frame_times) - 1) / span if span > 0 else 0.0
            self.chip_camera.set("ok", "Live")
            self.fps_label.setText(f"{fps:.0f} fps")

        if self.pose is not None:
            self.chip_paper.set("ok", f"{self.pose.width_mm:.1f} × {self.pose.height_mm:.1f} mm")
        else:
            self.chip_paper.set("bad", "Not detected")

        age = self.tracker.pin_age(now)
        if age is None:
            self.chip_pin.set("bad", "Not seen yet")
        elif age <= 0.5:
            self.chip_pin.set("ok", "Visible")
        else:
            self.chip_pin.set("warn", f"Hidden · held {age:.0f} s")

    def _on_camera_error(self, message):
        self._camera_error = message
        self.log(message, "error")
        self.camera_label.show_message(message)

    # --- Thread Execution Hooks ---
    def start_line(self):
        if self._busy():
            return
        now = time.monotonic()
        pose, _ = self.tracker.pose(now)
        start, end = self._line_mm()
        ok, message = self._check_line(pose, start, end)
        if not ok:
            self.log(f"REJECTED: {message}", "error")
            return

        age = self.tracker.pin_age(now)
        if age is not None and age > 0.5:
            self.log(f"Red pin hidden - using its position from {age:.1f} s ago.", "warn")

        (x0, y0), (x1, y1) = pose.paper_to_robot([start, end])
        self.log(f"Line ({start[0]:.1f}, {start[1]:.1f}) → ({end[0]:.1f}, {end[1]:.1f}) mm on paper "
                 f"= ({x0 * 1000:.1f}, {y0 * 1000:.1f}) → ({x1 * 1000:.1f}, {y1 * 1000:.1f}) mm "
                 f"from the red pin.")

        # Freeze the pose this line is planned with, and show exactly that.
        self.locked_pose, self.locked_line = pose, (start, end)
        self._update_paper_plot(pose)
        self._show_target(pose, start, end, True)
        self._run_worker(TrajectoryWorker("line", self.q_curr, self.serial_ctrl,
                                          line_xy=((x0, y0), (x1, y1))))

    def start_home(self):
        if self._busy():
            return
        self._run_worker(TrajectoryWorker("home", self.q_curr, self.serial_ctrl))

    def stop_motion(self):
        if self._busy():
            self.worker.requestInterruption()
            self._set_state("stopping")

    def _run_worker(self, worker):
        # The previous worker has finished (we are not busy), so dropping it is safe.
        self.worker = worker
        worker.step_signal.connect(self.update_plot_step)
        worker.log_signal.connect(self.log)
        worker.finished.connect(self.on_worker_finished)    # QThread's own: run() has returned
        self._set_state(worker.mode)
        worker.start()

    def update_plot_step(self, q_step):
        self.q_curr = q_step
        self.update_arm_geometry(q_step)

    def on_worker_finished(self):
        worker = self.worker
        self.locked_pose = self.locked_line = None
        self._set_state("idle")
        if worker is not None and worker.completed:
            self.log("Line drawn." if worker.mode == "line" else "Home reached.", "ok")
        self._update_paper_plot(self.pose)
        self._update_line_feedback()

    def closeEvent(self, event):
        self.camera_timer.stop()
        self.render_timer.stop()
        if self._busy():
            self.worker.requestInterruption()
            self.worker.wait(3000)
        self.camera_thread.requestInterruption()
        self.camera_thread.wait(3000)
        if self.serial_ctrl is not None:
            self.serial_ctrl.close()
        event.accept()


# ==========================================
# 6. Application Entry Point
# ==========================================
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = CobotControlPanel()

    # PyQt6 aborts the whole app on an exception raised inside a slot unless
    # sys.excepthook is replaced: report it in the log and keep running.
    def report(exc_type, exc, tb):
        traceback.print_exception(exc_type, exc, tb)
        try:
            window.log(f"Unexpected error: {exc!r}", "error")
        except Exception:
            pass
    sys.excepthook = report

    window.show()
    return app.exec()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("\n=== APPLICATION CRASHED ===")
        traceback.print_exc()
        input("\nPress Enter to exit...")

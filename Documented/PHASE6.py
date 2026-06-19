"""
PHASE6.py
=========
The most complete "headless" phase before the full GUI. It fuses everything:

  * YOLO + MediaPipe          -> GREEN/RED safety decision (as before).
  * RobotArm3D (yourdfpy +    -> renders the REAL SolidWorks-exported URDF arm
    pyrender + trimesh)          off-screen as a moving 3-D model.
  * compose_dashboard()       -> stitches the live camera panel and the 3-D arm
                                 panel into one polished "dashboard" window with
                                 a header status light and a footer HUD.

If the 3-D viewer can't load (missing libs / URDF), it gracefully falls back to
a placeholder canvas so the safety loop still runs.
"""

import math                                       # math: sine sweep + angle conversion for the arm.
import time                                        # time: FPS timing.

import cv2                                         # OpenCV: camera, drawing, the dashboard window.
import numpy as np                                 # NumPy: image canvases + vector math for the camera.

from ultralytics import YOLO                        # YOLOv8 object detection.

import mediapipe as mp                              # MediaPipe Hands.


CAM_INDEX = 0                                       # Default webcam.
FRAME_WIDTH = 640                                   # Capture width.
FRAME_HEIGHT = 460                                  # Capture height.
FLIP_VIEW = True                                    # Mirror image.

WINDOW_NAME = "Cobot Vision Safety Monitor"         # Dashboard window title.

YOLO_MODEL_PATH = "yolov8s.pt"                      # YOLO weights.
YOLO_CONF_THRESHOLD = 0.5                           # Min detection confidence.
YOLO_DEVICE = "cuda"                                # Run YOLO on the GPU.

MAX_HANDS = 2                                        # Track up to two hands.
MODEL_COMPLEXITY = 0                                 # Lightest hand model.
MIN_HAND_DET_CONF = 0.6                              # Hand detection gate.
MIN_HAND_TRACK_CONF = 0.5                            # Hand tracking gate.

HAND_AREA_THRESHOLD = 40_000                         # Starting "too close" area.
THRESHOLD_STEP    = 2_000                            # Adjustment per keypress.
THRESHOLD_MAX     = 150_000                          # Upper clamp.

ARM_WINDOW_NAME = "Robotic Arm"                      # (Legacy) standalone arm window name.
ARM_CANVAS_W = 640                                   # Arm render width.
ARM_CANVAS_H = 460                                   # Arm render height.
ARM_SPEED = 0.04                                     # Sweep phase increment per frame.

URDF_PATH = r"C:\Users\original\OneDrive\Desktop\GRAD VISION\URDF\urdf\allsubs10.urdf"  # 3-D arm model path.
JOINT_SWEEP_DEG = 35.0                               # Amplitude of each joint's sine sweep (degrees).

COLOR_GREEN = (0, 200, 0)                            # BGR colours.
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 215, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX                      # Shared font.


yolo_model = YOLO(YOLO_MODEL_PATH)                   # Load YOLO once.


mp_hands = mp.solutions.hands                        # Hands solution module.
mp_drawing = mp.solutions.drawing_utils              # Landmark drawing helpers.
mp_drawing_styles = mp.solutions.drawing_styles      # Default skeleton styles.

hands = mp_hands.Hands(                              # Create the hand tracker.
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    model_complexity=MODEL_COMPLEXITY,
    min_detection_confidence=MIN_HAND_DET_CONF,
    min_tracking_confidence=MIN_HAND_TRACK_CONF,
)



def compute_hand_bbox(landmarks, frame_width, frame_height):  # Pixel bbox + area for one hand.
    xs = [lm.x for lm in landmarks]                  # Normalised X of every landmark.
    ys = [lm.y for lm in landmarks]                  # Normalised Y of every landmark.

    min_x = max(0, int(min(xs) * frame_width))       # Left edge.
    max_x = min(frame_width - 1, int(max(xs) * frame_width))   # Right edge.
    min_y = max(0, int(min(ys) * frame_height))      # Top edge.
    max_y = min(frame_height - 1, int(max(ys) * frame_height)) # Bottom edge.

    box_w = max(0, max_x - min_x)                    # Width.
    box_h = max(0, max_y - min_y)                    # Height.
    area = box_w * box_h                             # Proximity proxy.

    return min_x, min_y, max_x, max_y, area          # Corners + area.


def draw_label(frame, text, x, y, bg_color, text_color=COLOR_WHITE):  # Text on a filled chip.
    (text_w, text_h), baseline = cv2.getTextSize(    # Measure text.
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    cv2.rectangle(                                   # Background chip.
        frame,
        (x, y - text_h - baseline - 4),
        (x + text_w + 6, y),
        bg_color,
        -1,
    )
    cv2.putText(                                     # Text over chip.
        frame,
        text,
        (x + 3, y - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        text_color,
        1,
        cv2.LINE_AA,
    )

def draw_threshold_indicator(frame, threshold: int) -> None:
    """Vertical gauge on the right edge."""
    h, w     = frame.shape[:2]                       # Frame size.
    bar_x    = w - 22                                # Gauge X.
    bar_top  = 70                                    # Gauge top.
    bar_bot  = h - 55                                # Gauge bottom.
    bar_len  = bar_bot - bar_top                     # Track length.
    max_area = THRESHOLD_MAX                          # Full-gauge value.

    cv2.rectangle(frame, (bar_x - 6, bar_top), (bar_x + 6, bar_bot), COLOR_BLACK,   -1)  # Track.

    ratio  = min(threshold / max_area, 1.0)          # Fill fraction.
    fill_y = int(bar_bot - ratio * bar_len)          # Fill top.
    cv2.rectangle(frame, (bar_x - 5, fill_y), (bar_x + 5, bar_bot), COLOR_YELLOW, -1)    # Fill.

    cv2.putText(frame, "THR", (bar_x - 14, bar_top - 8),  # Label.
                FONT, 0.38, COLOR_BLACK, 1, cv2.LINE_AA)


def draw_safety_banner(frame, safety_state, fps, threshold, collision_risk):  # Top banner + HUD.
    banner_h = 50                                    # Banner height.
    color = COLOR_GREEN if safety_state == "GREEN" else COLOR_RED  # Colour by state.
    cv2.rectangle(frame, (0, 0), (frame.shape[1], banner_h), color, -1)  # Banner.

    state_text = "GREEN ZONE - SAFE" if safety_state == "GREEN" else "RED ZONE - STOP"  # Banner text.
    cv2.putText(                                     # State label.
        frame,
        state_text,
        (10, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        COLOR_WHITE,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(                                     # FPS readout.
        frame,
        f"FPS: {fps:.1f}",
        (frame.shape[1] - 140, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_WHITE,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(                                     # Threshold readout.
        frame,
        f"HAND_AREA_THRESHOLD: {threshold}",
        (frame.shape[1] - 280, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_WHITE,
        1,
        cv2.LINE_AA,
    )

    if collision_risk:                               # Collision warning when relevant.
        cv2.putText(
            frame,
            "COLLISION RISK",
            (10, banner_h + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            COLOR_RED,
            2,
            cv2.LINE_AA,
        )


class RobotArm3D:
    """Loads the SolidWorks-exported URDF and renders it as a real 3D arm.

    Each actuated joint sweeps with a sine wave so the whole arm keeps moving;
    when `stopped` is passed to render() the motion is frozen in place.
    """

    def __init__(self, urdf_path: str, width: int, height: int):
        import yourdfpy                              # URDF loader (kinematics + scene graph).
        import pyrender                              # Off-screen OpenGL renderer.
        import trimesh                               # Mesh handling (used indirectly by the others).

        self._pyrender = pyrender                    # Keep a handle so render() can reuse it.
        self.width = width                           # Output image width.
        self.height = height                         # Output image height.

        self.robot = yourdfpy.URDF.load(urdf_path)   # Parse the URDF and build the kinematic model.
        self.joint_names = list(self.robot.actuated_joint_names)  # Names of joints we can move.

        # Camera framed on the model from its overall size.
        bounds = self.robot.scene.bounds          # (2, 3) min/max corners   # Axis-aligned bounding box.
        center = bounds.mean(axis=0)                 # Geometric centre of the model.
        size = float(np.linalg.norm(bounds[1] - bounds[0]))  # Diagonal length → overall scale.
        self._center = center                        # Store centre for camera aiming.
        self._radius = max(size, 1e-3)               # Store scale (avoid zero).

        self.renderer = pyrender.OffscreenRenderer(width, height)  # Create the off-screen GL renderer.

        # ---- Per-link colour table -------------------------------------
        # Colours live in the URDF (<material><color>); we read the colour
        # that actually got loaded back from each mesh, so the legend swatch
        # always matches what is drawn. Mesh nodes are named "<link>.STL" and
        # every link is the child of exactly one joint (base_link has none).
        child_to_joint = {j.child: j.name for j in self.robot.robot.joints}  # Map link → the joint above it.
        self._legend = []  # list of (label, (B, G, R)) ordered base -> tip   # Legend entries.
        scene = self.robot.scene                     # The trimesh scene holding all link meshes.
        for node_name in scene.graph.nodes_geometry: # For every mesh node in the scene...
            _, geom_name = scene.graph.get(node_name)  # Resolve the geometry name for this node.
            rgba = np.asarray(scene.geometry[geom_name].visual.main_color, int)  # Its RGBA colour (0..255).
            bgr = (int(rgba[2]), int(rgba[1]), int(rgba[0]))  # Convert RGB → BGR for OpenCV.
            link = node_name[:-4] if node_name.lower().endswith(".stl") else node_name  # Strip ".STL".
            joint = child_to_joint.get(link)         # Which joint drives this link (None for base).
            self._legend.append((f"{joint or 'base'} : {link.strip()}", bgr))  # Add a legend row.

    def _camera_pose(self):
        # Look at the model from an isometric-ish angle.
        d = self._radius * 1.1                       # Camera distance scaled to the model size.
        eye = self._center + np.array([d * 0.8, -d * 0.9, d * 0.7])  # Eye position (offset on all axes).
        forward = self._center - eye                 # Direction from eye toward the model.
        forward /= np.linalg.norm(forward)           # Normalise to unit length.
        up = np.array([0.0, 0.0, 1.0])               # World "up" is +Z.
        right = np.cross(forward, up)                # Camera's right axis = forward × up.
        right /= np.linalg.norm(right)               # Normalise.
        true_up = np.cross(right, forward)           # Re-orthogonalise the up axis.
        pose = np.eye(4)                             # 4×4 identity → we fill in the camera frame.
        pose[:3, 0] = right                          # Column 0 = right axis.
        pose[:3, 1] = true_up                        # Column 1 = up axis.
        pose[:3, 2] = -forward                       # Column 2 = backward (OpenGL looks down -Z).
        pose[:3, 3] = eye                            # Column 3 = camera position.
        return pose                                  # Return the camera-to-world transform.

    def render(self, phase: float, stopped: bool, draw_banner: bool = True):
        pyrender = self._pyrender                    # Local alias for speed/readability.

        amp = math.radians(JOINT_SWEEP_DEG)          # Sweep amplitude in radians.
        cfg = {                                      # Build a joint-name → angle dict...
            name: amp * math.sin(phase + i * 0.7)    # ...each joint phase-shifted so the arm waves.
            for i, name in enumerate(self.joint_names)
        }
        self.robot.update_cfg(cfg)                   # Apply the joint angles (forward kinematics).

        scene = pyrender.Scene(bg_color=[0.15, 0.15, 0.17, 1.0],  # New render scene with a dark bg...
                               ambient_light=[0.45, 0.45, 0.45])  # ...and soft ambient light.
        tscene = self.robot.scene                    # The posed trimesh scene.
        for node_name in tscene.graph.nodes_geometry:  # For each link mesh...
            T, geom_name = tscene.graph.get(node_name)  # Its world transform + geometry name.
            geom = tscene.geometry[geom_name]        # The mesh geometry.
            mesh = pyrender.Mesh.from_trimesh(geom, smooth=False)  # Convert to a pyrender mesh (flat shading).
            scene.add(mesh, pose=np.asarray(T))      # Add it at its computed pose.

        cam_pose = self._camera_pose()               # Compute the camera transform.
        scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4.0), pose=cam_pose)  # Add a 45° FOV camera.
        scene.add(                                   # Add a headlight that follows the camera...
            pyrender.DirectionalLight(color=np.ones(3), intensity=3.0),
            pose=cam_pose,
        )

        color, _ = self.renderer.render(scene)       # Render → RGB image (ignore the depth buffer).
        canvas = cv2.cvtColor(np.asarray(color), cv2.COLOR_RGB2BGR)  # RGB → BGR for OpenCV display.

        legend_top = 8                               # Default Y where the legend starts.
        if draw_banner:                              # Optional status banner on top of the arm view...
            arm_color = COLOR_RED if stopped else COLOR_GREEN
            status = "STOPPED - HAND IN RED ZONE" if stopped else "MOVING"
            cv2.rectangle(canvas, (0, 0), (self.width, 36), arm_color, -1)
            cv2.putText(canvas, status, (10, 26), FONT, 0.7, COLOR_WHITE, 2,
                        cv2.LINE_AA)
            legend_top = 44                          # Push the legend below the banner.

        self._draw_legend(canvas, legend_top)        # Draw the per-link colour legend.
        return canvas                                # Return the finished arm image.

    def _draw_legend(self, canvas, top):
        """Draw the per-link colour legend in the top-left corner."""
        x, y = 10, top                               # Starting position.
        sw = 14  # swatch size                        # Size of each colour square.
        for label, bgr in self._legend:             # For each (label, colour) row...
            cv2.rectangle(canvas, (x, y), (x + sw, y + sw), bgr, -1)        # Filled colour swatch.
            cv2.rectangle(canvas, (x, y), (x + sw, y + sw), COLOR_BLACK, 1) # Black outline.
            cv2.putText(canvas, label, (x + sw + 6, y + sw - 2), FONT, 0.4,  # Label text beside it.
                        COLOR_WHITE, 1, cv2.LINE_AA)
            y += sw + 6                              # Move down for the next row.

    def close(self):
        try:
            self.renderer.delete()                   # Free the GL renderer/context...
        except Exception:
            pass                                     # ...ignore errors during teardown.


def render_arm_placeholder(stopped: bool, message: str):
    """Fallback canvas shown when the URDF/3D viewer cannot be loaded."""
    canvas = np.full((ARM_CANVAS_H, ARM_CANVAS_W, 3), 40, dtype=np.uint8)  # Dark blank canvas.
    arm_color = COLOR_RED if stopped else COLOR_GREEN  # Banner colour by state.
    cv2.rectangle(canvas, (0, 0), (ARM_CANVAS_W, 36), arm_color, -1)  # Status banner.
    cv2.putText(canvas, "ARM VIEW UNAVAILABLE", (10, 26), FONT, 0.7,  # Headline.
                COLOR_WHITE, 2, cv2.LINE_AA)
    for i, line in enumerate(message.split("\n")):  # Print the error message line by line...
        cv2.putText(canvas, line, (10, 80 + i * 24), FONT, 0.5,
                    COLOR_WHITE, 1, cv2.LINE_AA)
    return canvas                                    # Return the placeholder image.


DASH_BG      = (28, 28, 34)                          # Dashboard background colour.
PANEL_BG     = (38, 38, 46)                          # Panel/header background colour.
PANEL_BORDER = (70, 70, 85)                          # Panel border colour.
PAD          = 18                                    # Padding between elements.
HEADER_H     = 66                                    # Header strip height.
FOOTER_H     = 60                                    # Footer strip height.
TITLE_H      = 28                                    # Per-panel title-bar height.


def _put(canvas, text, org, scale, color, thick=1):  # Tiny helper: anti-aliased putText with our font.
    cv2.putText(canvas, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def _place_panel(dash, img, x, y, w, h, title):
    """Draw a titled, bordered panel containing `img` resized to (w, h)."""
    cv2.rectangle(dash, (x, y), (x + w, y + TITLE_H), PANEL_BG, -1)  # Title bar background.
    _put(dash, title, (x + 10, y + 19), 0.5, COLOR_WHITE, 1)         # Title text.
    iy = y + TITLE_H                                 # Y where the image content begins.
    dash[iy:iy + h, x:x + w] = cv2.resize(img, (w, h))  # Paste the (resized) panel image into the dashboard.
    cv2.rectangle(dash, (x, y), (x + w, iy + h), PANEL_BORDER, 1)    # Border around the whole panel.


def compose_dashboard(cam_frame, arm_canvas, state, fps, threshold,
                      collision, num_hands):
    """Combine the camera feed and the 3D arm into one presentation window."""
    pw, ph = FRAME_WIDTH, FRAME_HEIGHT               # Camera panel size.
    aw, ah = ARM_CANVAS_W, ARM_CANVAS_H              # Arm panel size.
    body_h = TITLE_H + max(ph, ah)                   # Body height = tallest panel + its title bar.
    W = PAD + pw + PAD + aw + PAD                     # Total dashboard width (two panels + padding).
    H = HEADER_H + body_h + PAD + FOOTER_H            # Total dashboard height.
    dash = np.full((H, W, 3), DASH_BG, np.uint8)      # Blank dashboard canvas.

    safe = state == "GREEN"                          # Convenience boolean.
    accent = COLOR_GREEN if safe else COLOR_RED       # Accent colour follows safety state.

    # ---- header ----
    cv2.rectangle(dash, (0, 0), (W, HEADER_H), PANEL_BG, -1)         # Header background.
    cv2.rectangle(dash, (0, HEADER_H - 3), (W, HEADER_H), accent, -1)  # Coloured accent strip under header.
    _put(dash, "CSafety Rated Monitor Stop", (PAD, 43), 1.0, COLOR_WHITE, 2)  # Product title text.

    label = "SYSTEM SAFE" if safe else "EMERGENCY STOP"  # Status badge text.
    (tw, _th), _ = cv2.getTextSize(label, FONT, 0.7, 2)  # Measure it to size the badge.
    px2, px1 = W - PAD, W - PAD - (tw + 56)           # Right-aligned badge x-range.
    cv2.rectangle(dash, (px1, 15), (px2, 51), accent, -1)  # Badge background.
    cv2.circle(dash, (px1 + 20, 33), 8, COLOR_WHITE, -1)   # Status "light" dot.
    _put(dash, label, (px1 + 46, 50), 0.7, COLOR_WHITE, 2) # Badge text.

    # ---- panels ----
    y0 = HEADER_H + PAD - TITLE_H + 4                 # (Unused helper offset kept from original.)
    _place_panel(dash, cam_frame, PAD, HEADER_H + 6, pw, ph,         # Left panel: live camera.
                 "LIVE CAMERA  -  YOLO + HAND TRACKING")
    _place_panel(dash, arm_canvas, PAD + pw + PAD, HEADER_H + 6, aw, ah,  # Right panel: 3-D arm.
                 "ROBOTIC ARM  -  3D URDF MODEL")

    # ---- footer ----
    fy = H - FOOTER_H                                # Y where the footer begins.
    cv2.rectangle(dash, (0, fy), (W, H), PANEL_BG, -1)  # Footer background.
    _put(dash, f"FPS: {fps:4.1f}", (PAD, fy + 38), 0.6, COLOR_WHITE, 1)  # FPS.
    _put(dash, f"Hands detected: {num_hands}", (PAD + 150, fy + 38), 0.6,  # Hand count.
         COLOR_WHITE, 1)
    _put(dash, f"Arm: {'STOPPED' if not safe else 'MOVING'}",        # Arm state.
         (PAD + 400, fy + 38), 0.6, accent, 2)

    # horizontal sensitivity gauge
    gx1, gx2, gy = PAD + 580, PAD + 860, fy + 32      # Gauge x-start, x-end, y.
    _put(dash, f"Sensitivity threshold: {threshold}", (gx1, gy - 13), 0.45,  # Gauge label.
         COLOR_WHITE, 1)
    cv2.rectangle(dash, (gx1, gy - 7), (gx2, gy + 7), (60, 60, 70), -1)  # Gauge track.
    ratio = min(threshold / THRESHOLD_MAX, 1.0)       # Fill fraction.
    cv2.rectangle(dash, (gx1, gy - 7), (gx1 + int(ratio * (gx2 - gx1)), gy + 7),  # Gauge fill.
                  COLOR_YELLOW, -1)

    if collision:                                    # Optional collision warning in the footer.
        _put(dash, "!! COLLISION RISK !!", (W - 320, fy + 38), 0.8, COLOR_RED, 2)

    _put(dash, "Keys:  +/- adjust threshold   |   Q quit",  # Help line.
         (W - 360, H - 12), 0.45, (150, 150, 160), 1)
    return dash                                      # Return the composed dashboard image.



cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)     # Open camera (DirectShow).
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)       # Set width.
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)     # Set height.

prev_time = time.perf_counter()                      # Seed FPS timer.
last_state = None                                    # Track state changes.

threshold = HAND_AREA_THRESHOLD                      # Live threshold.
arm_phase = 0.0                                      # Arm animation phase.

# Load the 3D URDF arm once. If it fails, fall back to a message canvas.
arm_view = None                                      # Will hold the RobotArm3D instance (or stay None).
arm_error = ""                                       # Error text shown on the placeholder if load fails.
try:
    arm_view = RobotArm3D(URDF_PATH, ARM_CANVAS_W, ARM_CANVAS_H)  # Try to build the 3-D renderer.
    print(f"Loaded URDF arm with joints: {arm_view.joint_names}")  # Report success + joints.
except Exception as exc:  # missing URDF, missing libs, GL context, etc.
    arm_error = f"{type(exc).__name__}: {exc}\nCheck URDF_PATH and that\nyourdfpy + pyrender are installed."  # msg
    print("Could not load 3D arm:", arm_error)       # Log the failure (loop continues with placeholder).

while True:                                           # Main loop.
    ret, frame = cap.read()                          # Grab a frame.
    if not ret:                                      # Stop on failure.
        break

    if FLIP_VIEW:                                    # Optional mirror.
        frame = cv2.flip(frame, 1)

    # YOLO detection
    yolo_kwargs = {                                  # YOLO inference args.
        "conf": YOLO_CONF_THRESHOLD,
        "verbose": False,
    }
    if YOLO_DEVICE is not None:
        yolo_kwargs["device"] = YOLO_DEVICE

    yolo_results = yolo_model.predict(frame, **yolo_kwargs)  # Detect objects.
    objects_detected = False                         # Found anything?

    for result in yolo_results:                      # For each result...
        names = result.names                         # id → name.
        for box in result.boxes:                     # For each box...
            conf = float(box.conf[0])                # Confidence.
            if conf < YOLO_CONF_THRESHOLD:           # Skip low-confidence.
                continue
            cls_id = int(box.cls[0])                 # Class id.
            label = f"{names.get(cls_id, str(cls_id))} {conf * 100:.1f}%"  # Label.
            x1, y1, x2, y2 = box.xyxy[0].tolist()    # Corners (float).
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))  # → ints.

            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_YELLOW, 2)  # Draw box.
            draw_label(frame, label, x1, y1, COLOR_BLACK)             # Draw label.
            objects_detected = True                  # Flag.

    # MediaPipe Hands detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR → RGB.
    rgb_frame.flags.writeable = False                # Read-only → faster.
    hand_results = hands.process(rgb_frame)          # Detect/track hands.
    rgb_frame.flags.writeable = True                 # Restore.

    safety_state = "GREEN"                           # Assume safe.
    frame_h, frame_w = frame.shape[:2]               # Frame dims.
    num_hands = len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0  # Count.

    if hand_results.multi_hand_landmarks:            # If hands found...
        for hand_landmarks in hand_results.multi_hand_landmarks:  # For each hand...
            mp_drawing.draw_landmarks(               # Draw skeleton.
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            x1, y1, x2, y2, area = compute_hand_bbox(  # Measure box + area.
                hand_landmarks.landmark, frame_w, frame_h
            )
            box_color = COLOR_RED if area > threshold else COLOR_GREEN  # Colour by proximity.

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)        # Hand box.
            draw_label(frame, f"Hand area: {area}", x1, y1, box_color)    # Area label.

            if area > threshold:                     # If too close...
                safety_state = "RED"                 # ...go RED.

    collision_risk = safety_state == "RED" and objects_detected  # RED + objects = risk.

    # Decision output
    if safety_state != last_state:                   # On state change...
        if safety_state == "RED":
            print("ROBOT STOP - HUMAN TOO CLOSE")    # Log stop.
        else:
            print("ROBOT SAFE - OPERATION ALLOWED")  # Log safe.
        last_state = safety_state                    # Remember.

    now = time.perf_counter()                        # Now...
    fps = 1.0 / max(1e-6, now - prev_time)           # ...FPS...
    prev_time = now                                  # ...save.

    # Robotic arm: keep moving until the hand reaches the red zone.
    arm_stopped = safety_state == "RED"              # Freeze when unsafe.
    if not arm_stopped:                              # While safe...
        arm_phase += ARM_SPEED                       # ...advance the sweep.
    if arm_view is not None:                         # If the 3-D arm loaded...
        arm_canvas = arm_view.render(arm_phase, arm_stopped, draw_banner=False)  # ...render it.
    else:                                            # Otherwise...
        arm_canvas = render_arm_placeholder(arm_stopped, arm_error)  # ...show the placeholder.

    # Compose everything into one professional dashboard window.
    dashboard = compose_dashboard(frame, arm_canvas, safety_state, fps,  # Stitch camera + arm + HUD.
                                  threshold, collision_risk, num_hands)
    cv2.imshow(WINDOW_NAME, dashboard)               # Show the dashboard.

    # waitKeyEx returns the full keycode - arrow keys work on all OSes.
    # waitKey(1) & 0xFF only keeps the low byte, breaking arrow keys.
    #
    # Keycodes:  Windows   Linux    macOS
    #   Up       2490368   65362    63232
    #   Down     2621440   65364    63233
    # +/- keys work as a fallback on every platform.
    key = cv2.waitKeyEx(1)                           # Read a key.

    if key == ord("q"):                              # 'q' → quit.
        break
    elif key in (2490368, 65362, 63232, ord("+")):   # Up/'+' → raise threshold.
        threshold = min(threshold + THRESHOLD_STEP, THRESHOLD_MAX)
    elif key in (2621440, 65364, 63233, ord("-")):   # Down/'-' → lower threshold.
        threshold = max(threshold - THRESHOLD_STEP, THRESHOLD_STEP)



cap.release()                                        # Release camera.
cv2.destroyAllWindows()                              # Close windows.
hands.close()                                        # Free MediaPipe.
if arm_view is not None:                             # If the 3-D arm was created...
    arm_view.close()                                 # ...free its renderer too.

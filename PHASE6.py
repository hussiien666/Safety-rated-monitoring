import math
import time

import cv2
import numpy as np

from ultralytics import YOLO

import mediapipe as mp


CAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 460
FLIP_VIEW = True

WINDOW_NAME = "Cobot Vision Safety Monitor"

YOLO_MODEL_PATH = "yolov8s.pt"
YOLO_CONF_THRESHOLD = 0.5
YOLO_DEVICE = "cuda"  

MAX_HANDS = 2
MODEL_COMPLEXITY = 0
MIN_HAND_DET_CONF = 0.6
MIN_HAND_TRACK_CONF = 0.5

HAND_AREA_THRESHOLD = 40_000
THRESHOLD_STEP    = 2_000
THRESHOLD_MAX     = 150_000

ARM_WINDOW_NAME = "Robotic Arm"
ARM_CANVAS_W = 640
ARM_CANVAS_H = 460
ARM_SPEED = 0.04    

URDF_PATH = r"C:\Users\original\OneDrive\Desktop\GRAD VISION\URDF\urdf\allsubs10.urdf"
JOINT_SWEEP_DEG = 35.0 

COLOR_GREEN = (0, 200, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 215, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


yolo_model = YOLO(YOLO_MODEL_PATH)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
	static_image_mode=False,
	max_num_hands=MAX_HANDS,
	model_complexity=MODEL_COMPLEXITY,
	min_detection_confidence=MIN_HAND_DET_CONF,
	min_tracking_confidence=MIN_HAND_TRACK_CONF,
)



def compute_hand_bbox(landmarks, frame_width, frame_height):
	xs = [lm.x for lm in landmarks]
	ys = [lm.y for lm in landmarks]

	min_x = max(0, int(min(xs) * frame_width))
	max_x = min(frame_width - 1, int(max(xs) * frame_width))
	min_y = max(0, int(min(ys) * frame_height))
	max_y = min(frame_height - 1, int(max(ys) * frame_height))

	box_w = max(0, max_x - min_x)
	box_h = max(0, max_y - min_y)
	area = box_w * box_h

	return min_x, min_y, max_x, max_y, area


def draw_label(frame, text, x, y, bg_color, text_color=COLOR_WHITE):
	(text_w, text_h), baseline = cv2.getTextSize(
		text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
	)
	cv2.rectangle(
		frame,
		(x, y - text_h - baseline - 4),
		(x + text_w + 6, y),
		bg_color,
		-1,
	)
	cv2.putText(
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
    h, w     = frame.shape[:2]
    bar_x    = w - 22
    bar_top  = 70
    bar_bot  = h - 55
    bar_len  = bar_bot - bar_top
    max_area = THRESHOLD_MAX

    cv2.rectangle(frame, (bar_x - 6, bar_top), (bar_x + 6, bar_bot), COLOR_BLACK,   -1)

    ratio  = min(threshold / max_area, 1.0)
    fill_y = int(bar_bot - ratio * bar_len)
    cv2.rectangle(frame, (bar_x - 5, fill_y), (bar_x + 5, bar_bot), COLOR_YELLOW, -1)

    cv2.putText(frame, "THR", (bar_x - 14, bar_top - 8),
                FONT, 0.38, COLOR_BLACK, 1, cv2.LINE_AA)


def draw_safety_banner(frame, safety_state, fps, threshold, collision_risk):
	banner_h = 50
	color = COLOR_GREEN if safety_state == "GREEN" else COLOR_RED
	cv2.rectangle(frame, (0, 0), (frame.shape[1], banner_h), color, -1)

	state_text = "GREEN ZONE - SAFE" if safety_state == "GREEN" else "RED ZONE - STOP"
	cv2.putText(
		frame,
		state_text,
		(10, 32),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.9,
		COLOR_WHITE,
		2,
		cv2.LINE_AA,
	)

	cv2.putText(
		frame,
		f"FPS: {fps:.1f}",
		(frame.shape[1] - 140, 20),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.5,
		COLOR_WHITE,
		1,
		cv2.LINE_AA,
	)
	cv2.putText(
		frame,
		f"HAND_AREA_THRESHOLD: {threshold}",
		(frame.shape[1] - 280, 40),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.5,
		COLOR_WHITE,
		1,
		cv2.LINE_AA,
	)

	if collision_risk:
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
        import yourdfpy
        import pyrender
        import trimesh

        self._pyrender = pyrender
        self.width = width
        self.height = height

        self.robot = yourdfpy.URDF.load(urdf_path)
        self.joint_names = list(self.robot.actuated_joint_names)

        # Camera framed on the model from its overall size.
        bounds = self.robot.scene.bounds          # (2, 3) min/max corners
        center = bounds.mean(axis=0)
        size = float(np.linalg.norm(bounds[1] - bounds[0]))
        self._center = center
        self._radius = max(size, 1e-3)

        self.renderer = pyrender.OffscreenRenderer(width, height)

        # ---- Per-link colour table -------------------------------------
        # Colours live in the URDF (<material><color>); we read the colour
        # that actually got loaded back from each mesh, so the legend swatch
        # always matches what is drawn. Mesh nodes are named "<link>.STL" and
        # every link is the child of exactly one joint (base_link has none).
        child_to_joint = {j.child: j.name for j in self.robot.robot.joints}
        self._legend = []  # list of (label, (B, G, R)) ordered base -> tip
        scene = self.robot.scene
        for node_name in scene.graph.nodes_geometry:
            _, geom_name = scene.graph.get(node_name)
            rgba = np.asarray(scene.geometry[geom_name].visual.main_color, int)
            bgr = (int(rgba[2]), int(rgba[1]), int(rgba[0]))
            link = node_name[:-4] if node_name.lower().endswith(".stl") else node_name
            joint = child_to_joint.get(link)
            self._legend.append((f"{joint or 'base'} : {link.strip()}", bgr))

    def _camera_pose(self):
        # Look at the model from an isometric-ish angle.
        d = self._radius * 1.1
        eye = self._center + np.array([d * 0.8, -d * 0.9, d * 0.7])
        forward = self._center - eye
        forward /= np.linalg.norm(forward)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        true_up = np.cross(right, forward)
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = true_up
        pose[:3, 2] = -forward
        pose[:3, 3] = eye
        return pose

    def render(self, phase: float, stopped: bool, draw_banner: bool = True):
        pyrender = self._pyrender

        amp = math.radians(JOINT_SWEEP_DEG)
        cfg = {
            name: amp * math.sin(phase + i * 0.7)
            for i, name in enumerate(self.joint_names)
        }
        self.robot.update_cfg(cfg)

        scene = pyrender.Scene(bg_color=[0.15, 0.15, 0.17, 1.0],
                               ambient_light=[0.45, 0.45, 0.45])
        tscene = self.robot.scene
        for node_name in tscene.graph.nodes_geometry:
            T, geom_name = tscene.graph.get(node_name)
            geom = tscene.geometry[geom_name]
            mesh = pyrender.Mesh.from_trimesh(geom, smooth=False)
            scene.add(mesh, pose=np.asarray(T))

        cam_pose = self._camera_pose()
        scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4.0), pose=cam_pose)
        scene.add(
            pyrender.DirectionalLight(color=np.ones(3), intensity=3.0),
            pose=cam_pose,
        )

        color, _ = self.renderer.render(scene)
        canvas = cv2.cvtColor(np.asarray(color), cv2.COLOR_RGB2BGR)

        legend_top = 8
        if draw_banner:
            arm_color = COLOR_RED if stopped else COLOR_GREEN
            status = "STOPPED - HAND IN RED ZONE" if stopped else "MOVING"
            cv2.rectangle(canvas, (0, 0), (self.width, 36), arm_color, -1)
            cv2.putText(canvas, status, (10, 26), FONT, 0.7, COLOR_WHITE, 2,
                        cv2.LINE_AA)
            legend_top = 44

        self._draw_legend(canvas, legend_top)
        return canvas

    def _draw_legend(self, canvas, top):
        """Draw the per-link colour legend in the top-left corner."""
        x, y = 10, top
        sw = 14  # swatch size
        for label, bgr in self._legend:
            cv2.rectangle(canvas, (x, y), (x + sw, y + sw), bgr, -1)
            cv2.rectangle(canvas, (x, y), (x + sw, y + sw), COLOR_BLACK, 1)
            cv2.putText(canvas, label, (x + sw + 6, y + sw - 2), FONT, 0.4,
                        COLOR_WHITE, 1, cv2.LINE_AA)
            y += sw + 6

    def close(self):
        try:
            self.renderer.delete()
        except Exception:
            pass


def render_arm_placeholder(stopped: bool, message: str):
    """Fallback canvas shown when the URDF/3D viewer cannot be loaded."""
    canvas = np.full((ARM_CANVAS_H, ARM_CANVAS_W, 3), 40, dtype=np.uint8)
    arm_color = COLOR_RED if stopped else COLOR_GREEN
    cv2.rectangle(canvas, (0, 0), (ARM_CANVAS_W, 36), arm_color, -1)
    cv2.putText(canvas, "ARM VIEW UNAVAILABLE", (10, 26), FONT, 0.7,
                COLOR_WHITE, 2, cv2.LINE_AA)
    for i, line in enumerate(message.split("\n")):
        cv2.putText(canvas, line, (10, 80 + i * 24), FONT, 0.5,
                    COLOR_WHITE, 1, cv2.LINE_AA)
    return canvas


DASH_BG      = (28, 28, 34)      
PANEL_BG     = (38, 38, 46)      
PANEL_BORDER = (70, 70, 85)
PAD          = 18
HEADER_H     = 66
FOOTER_H     = 60
TITLE_H      = 28


def _put(canvas, text, org, scale, color, thick=1):
    cv2.putText(canvas, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def _place_panel(dash, img, x, y, w, h, title):
    """Draw a titled, bordered panel containing `img` resized to (w, h)."""
    cv2.rectangle(dash, (x, y), (x + w, y + TITLE_H), PANEL_BG, -1)
    _put(dash, title, (x + 10, y + 19), 0.5, COLOR_WHITE, 1)
    iy = y + TITLE_H
    dash[iy:iy + h, x:x + w] = cv2.resize(img, (w, h))
    cv2.rectangle(dash, (x, y), (x + w, iy + h), PANEL_BORDER, 1)


def compose_dashboard(cam_frame, arm_canvas, state, fps, threshold,
                      collision, num_hands):
    """Combine the camera feed and the 3D arm into one presentation window."""
    pw, ph = FRAME_WIDTH, FRAME_HEIGHT
    aw, ah = ARM_CANVAS_W, ARM_CANVAS_H
    body_h = TITLE_H + max(ph, ah)
    W = PAD + pw + PAD + aw + PAD
    H = HEADER_H + body_h + PAD + FOOTER_H
    dash = np.full((H, W, 3), DASH_BG, np.uint8)

    safe = state == "GREEN"
    accent = COLOR_GREEN if safe else COLOR_RED

    # ---- header ----
    cv2.rectangle(dash, (0, 0), (W, HEADER_H), PANEL_BG, -1)
    cv2.rectangle(dash, (0, HEADER_H - 3), (W, HEADER_H), accent, -1)
    _put(dash, "CSafety Rated Monitor Stop", (PAD, 43), 1.0, COLOR_WHITE, 2)

    label = "SYSTEM SAFE" if safe else "EMERGENCY STOP"
    (tw, _th), _ = cv2.getTextSize(label, FONT, 0.7, 2)
    px2, px1 = W - PAD, W - PAD - (tw + 56)
    cv2.rectangle(dash, (px1, 15), (px2, 51), accent, -1)
    cv2.circle(dash, (px1 + 20, 33), 8, COLOR_WHITE, -1)
    _put(dash, label, (px1 + 46, 50), 0.7, COLOR_WHITE, 2)

    # ---- panels ----
    y0 = HEADER_H + PAD - TITLE_H + 4
    _place_panel(dash, cam_frame, PAD, HEADER_H + 6, pw, ph,
                 "LIVE CAMERA  -  YOLO + HAND TRACKING")
    _place_panel(dash, arm_canvas, PAD + pw + PAD, HEADER_H + 6, aw, ah,
                 "ROBOTIC ARM  -  3D URDF MODEL")

    # ---- footer ----
    fy = H - FOOTER_H
    cv2.rectangle(dash, (0, fy), (W, H), PANEL_BG, -1)
    _put(dash, f"FPS: {fps:4.1f}", (PAD, fy + 38), 0.6, COLOR_WHITE, 1)
    _put(dash, f"Hands detected: {num_hands}", (PAD + 150, fy + 38), 0.6,
         COLOR_WHITE, 1)
    _put(dash, f"Arm: {'STOPPED' if not safe else 'MOVING'}",
         (PAD + 400, fy + 38), 0.6, accent, 2)

    # horizontal sensitivity gauge
    gx1, gx2, gy = PAD + 580, PAD + 860, fy + 32
    _put(dash, f"Sensitivity threshold: {threshold}", (gx1, gy - 13), 0.45,
         COLOR_WHITE, 1)
    cv2.rectangle(dash, (gx1, gy - 7), (gx2, gy + 7), (60, 60, 70), -1)
    ratio = min(threshold / THRESHOLD_MAX, 1.0)
    cv2.rectangle(dash, (gx1, gy - 7), (gx1 + int(ratio * (gx2 - gx1)), gy + 7),
                  COLOR_YELLOW, -1)

    if collision:
        _put(dash, "!! COLLISION RISK !!", (W - 320, fy + 38), 0.8, COLOR_RED, 2)

    _put(dash, "Keys:  +/- adjust threshold   |   Q quit",
         (W - 360, H - 12), 0.45, (150, 150, 160), 1)
    return dash



cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

prev_time = time.perf_counter()
last_state = None

threshold = HAND_AREA_THRESHOLD
arm_phase = 0.0

# Load the 3D URDF arm once. If it fails, fall back to a message canvas.
arm_view = None
arm_error = ""
try:
	arm_view = RobotArm3D(URDF_PATH, ARM_CANVAS_W, ARM_CANVAS_H)
	print(f"Loaded URDF arm with joints: {arm_view.joint_names}")
except Exception as exc:  # missing URDF, missing libs, GL context, etc.
	arm_error = f"{type(exc).__name__}: {exc}\nCheck URDF_PATH and that\nyourdfpy + pyrender are installed."
	print("Could not load 3D arm:", arm_error)

while True:
	ret, frame = cap.read()
	if not ret:
		break

	if FLIP_VIEW:
		frame = cv2.flip(frame, 1)

	# YOLO detection
	yolo_kwargs = {
		"conf": YOLO_CONF_THRESHOLD,
		"verbose": False,
	}
	if YOLO_DEVICE is not None:
		yolo_kwargs["device"] = YOLO_DEVICE

	yolo_results = yolo_model.predict(frame, **yolo_kwargs)
	objects_detected = False

	for result in yolo_results:
		names = result.names
		for box in result.boxes:
			conf = float(box.conf[0])
			if conf < YOLO_CONF_THRESHOLD:
				continue
			cls_id = int(box.cls[0])
			label = f"{names.get(cls_id, str(cls_id))} {conf * 100:.1f}%"
			x1, y1, x2, y2 = box.xyxy[0].tolist()
			x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

			cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_YELLOW, 2)
			draw_label(frame, label, x1, y1, COLOR_BLACK)
			objects_detected = True

	# MediaPipe Hands detection
	rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb_frame.flags.writeable = False
	hand_results = hands.process(rgb_frame)
	rgb_frame.flags.writeable = True

	safety_state = "GREEN"
	frame_h, frame_w = frame.shape[:2]
	num_hands = len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0

	if hand_results.multi_hand_landmarks:
		for hand_landmarks in hand_results.multi_hand_landmarks:
			mp_drawing.draw_landmarks(
				frame,
				hand_landmarks,
				mp_hands.HAND_CONNECTIONS,
				mp_drawing_styles.get_default_hand_landmarks_style(),
				mp_drawing_styles.get_default_hand_connections_style(),
			)

			x1, y1, x2, y2, area = compute_hand_bbox(
				hand_landmarks.landmark, frame_w, frame_h
			)
			box_color = COLOR_RED if area > threshold else COLOR_GREEN

			cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
			draw_label(frame, f"Hand area: {area}", x1, y1, box_color)

			if area > threshold:
				safety_state = "RED"

	collision_risk = safety_state == "RED" and objects_detected

	# Decision output
	if safety_state != last_state:
		if safety_state == "RED":
			print("ROBOT STOP - HUMAN TOO CLOSE")
		else:
			print("ROBOT SAFE - OPERATION ALLOWED")
		last_state = safety_state

	now = time.perf_counter()
	fps = 1.0 / max(1e-6, now - prev_time)
	prev_time = now

	# Robotic arm: keep moving until the hand reaches the red zone.
	arm_stopped = safety_state == "RED"
	if not arm_stopped:
		arm_phase += ARM_SPEED
	if arm_view is not None:
		arm_canvas = arm_view.render(arm_phase, arm_stopped, draw_banner=False)
	else:
		arm_canvas = render_arm_placeholder(arm_stopped, arm_error)

	# Compose everything into one professional dashboard window.
	dashboard = compose_dashboard(frame, arm_canvas, safety_state, fps,
	                              threshold, collision_risk, num_hands)
	cv2.imshow(WINDOW_NAME, dashboard)

	# waitKeyEx returns the full keycode - arrow keys work on all OSes.
	# waitKey(1) & 0xFF only keeps the low byte, breaking arrow keys.
	#
	# Keycodes:  Windows   Linux    macOS
	#   Up       2490368   65362    63232
	#   Down     2621440   65364    63233
	# +/- keys work as a fallback on every platform.
	key = cv2.waitKeyEx(1)

	if key == ord("q"):
		break
	elif key in (2490368, 65362, 63232, ord("+")):
		threshold = min(threshold + THRESHOLD_STEP, THRESHOLD_MAX)
	elif key in (2621440, 65364, 63233, ord("-")):
		threshold = max(threshold - THRESHOLD_STEP, THRESHOLD_STEP)



cap.release()
cv2.destroyAllWindows()
hands.close()
if arm_view is not None:
	arm_view.close()
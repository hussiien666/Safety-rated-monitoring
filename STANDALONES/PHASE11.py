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

# Live ruler: the span across the knuckles (index MCP -> pinky MCP) is roughly
# constant on an adult hand, so it gives a mm-per-pixel scale that re-calibrates
# itself every frame as the hand moves nearer or farther from the camera.
HAND_KNUCKLE_SPAN_MM = 80.0
INDEX_MCP_IDX = 5
PINKY_MCP_IDX = 17

# Book measuring: YOLO tells us *that* a book is there, then a contour pass
# inside its box finds the cover's true tilted rectangle so a book lying at an
# angle is measured along its own edges, not along the axis-aligned box.
BOOK_CLASS_NAMES = {"book"}
BOOK_ROI_PAD = 6                 # px of slack around the YOLO box before refining
BOOK_APPROX_EPS = 0.02           # approxPolyDP tolerance, fraction of perimeter
BOOK_MIN_ROI_FRAC = 0.30         # refined cover must fill 30% of the YOLO box
BOOK_MAX_ROI_FRAC = 0.95         # ...and must not BE the box (that is background)
BOOK_MIN_FILL = 0.70             # contour area / its minAreaRect area
SCALE_STALE_SECS = 5.0           # how long a hand-derived scale stays usable

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


def compute_mm_per_pixel(landmarks, frame_width, frame_height):
	"""Scale factor from the hand's own knuckle span. None if it is degenerate."""
	index_mcp = landmarks[INDEX_MCP_IDX]
	pinky_mcp = landmarks[PINKY_MCP_IDX]

	dx = (index_mcp.x - pinky_mcp.x) * frame_width
	dy = (index_mcp.y - pinky_mcp.y) * frame_height
	span_px = math.hypot(dx, dy)

	if span_px < 1.0:
		return None

	return HAND_KNUCKLE_SPAN_MM / span_px


def measure_book(frame, x1, y1, x2, y2):
	"""Refine a YOLO book box into the cover's true tilted rectangle.

	Returns (quad, short_px, long_px). `quad` is the 4x2 corner array in full
	frame coordinates. Falls back to the axis-aligned YOLO box when no clean
	cover contour can be found inside it (busy background, partial occlusion).
	"""
	frame_h, frame_w = frame.shape[:2]
	rx1 = max(0, x1 - BOOK_ROI_PAD)
	ry1 = max(0, y1 - BOOK_ROI_PAD)
	rx2 = min(frame_w, x2 + BOOK_ROI_PAD)
	ry2 = min(frame_h, y2 + BOOK_ROI_PAD)

	box_quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=int)
	fallback = (box_quad, float(min(x2 - x1, y2 - y1)), float(max(x2 - x1, y2 - y1)))

	roi = frame[ry1:ry2, rx1:rx2]
	if roi.size == 0:
		return fallback

	grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	grey = cv2.GaussianBlur(grey, (5, 5), 0)

	# Otsu separates the cover from whatever it is lying on, without a
	# hand-tuned threshold; the closing fills gaps that print and glare punch
	# in the cover. Both polarities are tried since a book may be lighter or
	# darker than its background.
	_, mask = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

	roi_area = float(roi.shape[0] * roi.shape[1])
	best = None

	for candidate_mask in (mask, cv2.bitwise_not(mask)):
		closed = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
		contours, _ = cv2.findContours(
			closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
		)
		for contour in contours:
			area = cv2.contourArea(contour)
			# The upper bound matters: on the inverted mask the *background*
			# forms one contour the size of the whole ROI, which would
			# otherwise win on area and hand back the padded box.
			if not (BOOK_MIN_ROI_FRAC * roi_area <= area <= BOOK_MAX_ROI_FRAC * roi_area):
				continue

			(_, _), (rect_w, rect_h), _ = cv2.minAreaRect(contour)
			if rect_w < 1 or rect_h < 1:
				continue
			if area / (rect_w * rect_h) < BOOK_MIN_FILL:
				continue

			peri = cv2.arcLength(contour, True)
			approx = cv2.approxPolyDP(contour, BOOK_APPROX_EPS * peri, True)
			if len(approx) != 4 or not cv2.isContourConvex(approx):
				continue

			if best is None or area > best[0]:
				best = (area, approx.reshape(4, 2).astype(int), rect_w, rect_h)

	if best is None:
		return fallback

	_, quad, rect_w, rect_h = best
	# ROI -> full-frame coordinates
	quad = (quad + np.array([rx1, ry1])).astype(np.int32)
	short_px, long_px = sorted((rect_w, rect_h))
	return quad, short_px, long_px


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

def lengthen_text(text: str, target_length: int) -> str:
    """Pad the text with spaces to reach the target length."""
    if len(text) >= target_length:
        return text
    padding = " " * (target_length - len(text))
    return text + padding

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


cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

prev_time = time.perf_counter()
last_state = None

threshold = HAND_AREA_THRESHOLD

# The hand ruler only exists while a hand is visible, so remember the last
# scale it gave us and keep measuring books for a few seconds after it leaves.
last_mm_per_px = None
last_scale_time = 0.0

while True:
	ret, frame = cap.read()
	if not ret:
		break

	if FLIP_VIEW:
		frame = cv2.flip(frame, 1)

	# Keep an un-annotated copy: the book refiner thresholds the image, and the
	# boxes/landmarks drawn below would otherwise show up as fake cover edges.
	clean_frame = frame.copy()

	yolo_kwargs = {
		"conf": YOLO_CONF_THRESHOLD,
		"verbose": False,
	}
	if YOLO_DEVICE is not None:
		yolo_kwargs["device"] = YOLO_DEVICE

	yolo_results = yolo_model.predict(frame, **yolo_kwargs)
	objects_detected = False
	book_boxes = []

	for result in yolo_results:
		names = result.names
		for box in result.boxes:
			conf = float(box.conf[0])
			if conf < YOLO_CONF_THRESHOLD:
				continue
			cls_id = int(box.cls[0])
			cls_name = names.get(cls_id, str(cls_id))
			label = f"{cls_name} {conf * 100:.1f}%"
			x1, y1, x2, y2 = box.xyxy[0].tolist()
			x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

			if cls_name in BOOK_CLASS_NAMES:
				# Measured below; its own outline is drawn there instead.
				book_boxes.append((x1, y1, x2, y2, conf))
			else:
				cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_YELLOW, 2)
				draw_label(frame, label, x1, y1, COLOR_BLACK)
			objects_detected = True

	rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb_frame.flags.writeable = False
	hand_results = hands.process(rgb_frame)
	rgb_frame.flags.writeable = True

	safety_state = "GREEN"
	frame_h, frame_w = frame.shape[:2]

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
			box_w = x2 - x1
			box_h = y2 - y1
			box_color = COLOR_RED if area > threshold else COLOR_GREEN

			mm_per_px = compute_mm_per_pixel(
				hand_landmarks.landmark, frame_w, frame_h
			)
			if mm_per_px is None:
				size_text = "W: -- mm  H: -- mm"
			else:
				last_mm_per_px = mm_per_px
				last_scale_time = time.perf_counter()
				size_text = (
					f"W: {box_w * mm_per_px:.0f} mm  H: {box_h * mm_per_px:.0f} mm"
				)

			cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
			draw_label(frame, f"Hand area: {area}", x1, y1, box_color)
			draw_label(
				frame,
				size_text,
				x1,
				min(frame_h - 2, y2 + 20),
				box_color,
			)

			if area > threshold:
				safety_state = "RED"

	# --- Books: outline each one and label its size in millimetres -----------
	scale_fresh = (
		last_mm_per_px is not None
		and (time.perf_counter() - last_scale_time) < SCALE_STALE_SECS
	)

	for bx1, by1, bx2, by2, bconf in book_boxes:
		quad, short_px, long_px = measure_book(clean_frame, bx1, by1, bx2, by2)
		cv2.polylines(frame, [quad], True, COLOR_YELLOW, 2)

		if scale_fresh:
			short_mm = short_px * last_mm_per_px
			long_mm = long_px * last_mm_per_px
			book_text = f"BOOK {long_mm:.0f} x {short_mm:.0f} mm"
		else:
			book_text = "BOOK -- mm (show hand to calibrate)"

		label_x = int(quad[:, 0].min())
		label_y = int(quad[:, 1].min())
		draw_label(
			frame,
			f"{book_text}  {bconf * 100:.0f}%",
			max(0, label_x),
			max(18, label_y),
			COLOR_BLACK,
			COLOR_YELLOW,
		)

	collision_risk = safety_state == "RED" and objects_detected

	if safety_state != last_state:
		if safety_state == "RED":
			print("ROBOT STOP - HUMAN TOO CLOSE")
		else:
			print("ROBOT SAFE - OPERATION ALLOWED")
		last_state = safety_state

	now = time.perf_counter()
	fps = 1.0 / max(1e-6, now - prev_time)
	prev_time = now

	draw_safety_banner(frame, safety_state, fps, threshold, collision_risk)
	draw_threshold_indicator(frame, threshold)

	cv2.imshow(WINDOW_NAME, frame)

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
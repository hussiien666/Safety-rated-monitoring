import logging
import math
import os
import time

# MediaPipe drags TensorFlow in behind it, and TensorFlow greets every run with
# a page of oneDNN and Keras-deprecation notices. None of it is an error and
# none of it is ours - it just buries the output that matters. The gags have to
# go up before the import that loads TF, because the banner is printed as it
# loads: set afterwards they are simply too late.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # TF's C++ INFO + WARNING lines
os.environ.setdefault("GLOG_minloglevel", "2")       # MediaPipe's own glog chatter
logging.getLogger("tensorflow").setLevel(logging.ERROR)   # the Keras deprecations
logging.getLogger("absl").setLevel(logging.ERROR)

# Note this only hides the oneDNN notice; oneDNN itself stays on, so the numbers
# do not change. Setting TF_ENABLE_ONEDNN_OPTS=0 instead - as the message
# suggests - would actually turn those optimisations off.

import cv2
import numpy as np

import mediapipe as mp




CAM_INDEX = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 460
FLIP_VIEW = True

WINDOW_NAME = "Cobot Vision White Surface Measurement"

MAX_HANDS = 2
MODEL_COMPLEXITY = 0
MIN_HAND_DET_CONF = 0.6
MIN_HAND_TRACK_CONF = 0.5

# The ruler is the camera mounting itself, not a hand waved at the lens. A lens
# a known distance above a flat work surface sees a known width of that surface,
# so mm-per-pixel is fixed geometry: it is there on the very first frame, with
# nothing to show the camera and nothing to wait for.
#
#     view width (mm) = 2 * height * tan(hfov / 2)
#     mm per pixel    = view width / frame width in px
#
# Two numbers have to be right for the millimetres to be right: how far the lens
# is from the surface, and how wide the lens sees. Measure the height once, from
# the lens to the table, and nudge it live with [ and ] until a known object
# reads back its true size. HFOV is a property of the camera - most webcams are
# somewhere around 55-70 degrees; the datasheet value is a good starting point.
CAMERA_HEIGHT_MM = 400.0         # lens to the work surface, looking straight down
CAMERA_HFOV_DEG = 60.0           # horizontal field of view of the lens
HEIGHT_STEP_MM = 10.0            # [ and ] nudge the working distance by this
MIN_CAMERA_HEIGHT_MM = 50.0

# Everything here assumes the surfaces lie in one plane at that height. A sheet
# lifted off the table is nearer the lens than the calibration assumes, so it
# measures large - which is exactly the trade for not needing a reference in
# frame. PHASE13 scales off a known sheet instead, and does not care about height.

# White is "bright and colourless", so it is picked in HSV: any hue, low
# saturation, high value. A plain grey threshold would also swallow bright
# yellows and skin lit by a warm lamp.
WHITE_SAT_MAX = 70               # S <= this
WHITE_VAL_MIN = 190              # V >= this, in manual mode

# A fixed brightness gate flickers indoors: the webcam's auto-exposure keeps
# breathing, so the sheet's pixels drift back and forth across the line. In
# auto mode the gate is derived from each frame's own brightness histogram
# (Otsu), so it rides the exposure instead of fighting it.
AUTO_WHITE = True
AUTO_VAL_FLOOR = 130             # never trust a gate below this
AUTO_VAL_CEIL = 245
VAL_BIAS_STEP = 5

# Hysteresis, the same idea Canny uses on edges: pixels well above the gate are
# "core" white, pixels just above it are only kept when they are connected to a
# core pixel. Borderline pixels then stop blinking in and out on their own.
CORE_VAL_MARGIN = 20
CORE_SAT_FACTOR = 0.7

SAT_STEP = 5

# Opening kills sensor speckle; the bigger closing bridges the glare streaks and
# printed text that would otherwise cut one sheet into several rival blobs.
OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))

# Temporal smoothing: a pixel has to stay white for a few frames before it
# counts, and has to stay dark for a few before it stops counting. Held on
# only while the scene is still - the same smoothing applied to a sheet being
# moved would drag the mask along a couple of frames behind the paper.
MASK_EMA = 0.45
MASK_SNAP_EMA = 0.90
MASK_MOVE_IOU = 0.90             # mask overlap below this means something moved
MASK_ON = 0.55

MIN_REGION_PX = 1_500            # area needed to *acquire* a surface
KEEP_REGION_FRAC = 0.6           # ...and only 60% of that to *keep* one
MAX_REGION_FRAC = 0.90           # >90% of the frame is a wall / blown exposure
MAX_REGIONS = 4                  # only label the biggest few, to keep it readable

# The outline is fitted as a real four-cornered quad, not a rectangle: a sheet
# held at any angle is a trapezoid once the camera's perspective is in play, and
# a min-area *rectangle* around a trapezoid stands off its corners. The looser
# tolerances are tried in turn and the first clean convex quad wins.
QUAD_EPS_STEPS = (0.01, 0.02, 0.03, 0.04, 0.05)
QUAD_AREA_TOL = 0.12             # quad must account for the blob's area, +/-12%

# Hands are tracked only so a pale hand is not measured as a white surface -
# nothing about the scale depends on them any more. Blanking the hand box out of
# the mask used to bite a notch out of a sheet being *held* - the sheet is what
# you are measuring, so instead a region is dropped only when it is mostly hand.
HAND_MASK_PAD = 18
HAND_OVERLAP_DROP = 0.60         # region this much inside a hand box -> it is the hand

# Surfaces are tracked frame to frame so #1 stays #1, the numbers settle instead
# of twitching, and one dropped frame does not blank the label.
TRACK_MATCH_DIST = 70.0          # px between centroids to count as the same sheet
TRACK_MISS_FRAMES = 8            # keep showing a surface this long after it drops
MEASURE_EMA = 0.25               # smoothing while the surface is holding still
SNAP_EMA = 0.90                  # ...and when it is actually being moved
SNAP_DIST = 10.0                 # px of corner travel that counts as "being moved"

BANNER_H = 62                    # labels are kept clear of it
MASK_TINT = 0.30                 # light enough to still read the surface under it

COLOR_GREEN = (0, 200, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 215, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX




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

	return min_x, min_y, max_x, max_y


def compute_mm_per_pixel(frame_width, height_mm, hfov_deg=CAMERA_HFOV_DEG):
	"""Scale factor from the camera's own geometry - no reference object needed.

	At a working distance of height_mm the lens spans 2*h*tan(hfov/2) millimetres
	of the work surface, and the sensor spreads that across frame_width pixels.
	The scale is therefore known before a single frame is looked at, and only
	changes if the camera is moved.
	"""
	view_width_mm = 2.0 * height_mm * math.tan(math.radians(hfov_deg) / 2.0)
	return view_width_mm / float(max(1, frame_width))


def auto_val_threshold(value_channel):
	"""Brightness gate taken from this frame's own histogram.

	Otsu splits the frame into its dark and bright populations; the sheet lives
	in the bright one. Clamped so a dim frame cannot drag the gate down onto the
	background and a blown one cannot push it past the paper.
	"""
	otsu_thr, _ = cv2.threshold(value_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return int(min(AUTO_VAL_CEIL, max(AUTO_VAL_FLOOR, otsu_thr)))


def white_mask(frame, sat_max, val_min):
	"""Binary mask of the white pixels, with hysteresis against flicker."""
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	loose = cv2.inRange(
		hsv,
		np.array([0, 0, val_min], np.uint8),
		np.array([179, sat_max, 255], np.uint8),
	)
	core = cv2.inRange(
		hsv,
		np.array([0, 0, min(255, val_min + CORE_VAL_MARGIN)], np.uint8),
		np.array([179, int(sat_max * CORE_SAT_FACTOR), 255], np.uint8),
	)

	loose = cv2.morphologyEx(loose, cv2.MORPH_OPEN, OPEN_KERNEL)
	loose = cv2.morphologyEx(loose, cv2.MORPH_CLOSE, CLOSE_KERNEL)
	core = cv2.morphologyEx(core, cv2.MORPH_OPEN, OPEN_KERNEL)

	# Keep only the loose blobs that actually contain solidly-white pixels, so a
	# borderline patch of background never becomes a surface on its own.
	count, labels = cv2.connectedComponents(loose, connectivity=8)
	if count <= 1:
		return loose

	keep = np.zeros(count, np.bool_)
	keep[labels[core > 0]] = True
	keep[0] = False
	return np.where(keep[labels], 255, 0).astype(np.uint8)


def smooth_mask(accumulator, mask):
	"""Blend this frame's mask into the running one and re-threshold it.

	How hard it smooths depends on how much the mask just moved: overlapping
	almost perfectly with the running mask means a still scene, where smoothing
	is what kills flicker, while a drop in overlap means the sheet is being
	moved and the new mask should mostly be taken as it is.
	"""
	fresh = mask > 0
	current = accumulator > MASK_ON
	union = int(np.count_nonzero(np.logical_or(current, fresh)))
	iou = int(np.count_nonzero(np.logical_and(current, fresh))) / max(1, union)

	alpha = MASK_EMA if iou >= MASK_MOVE_IOU else MASK_SNAP_EMA
	cv2.accumulateWeighted(mask.astype(np.float32) / 255.0, accumulator, alpha)
	return (accumulator > MASK_ON).astype(np.uint8) * 255


def hand_regions_mask(shape, hand_boxes):
	"""Padded hand boxes as (x1, y1, x2, y2) clipped to the frame."""
	frame_h, frame_w = shape[:2]
	boxes = []
	for x1, y1, x2, y2 in hand_boxes:
		boxes.append((
			max(0, x1 - HAND_MASK_PAD),
			max(0, y1 - HAND_MASK_PAD),
			min(frame_w, x2 + HAND_MASK_PAD),
			min(frame_h, y2 + HAND_MASK_PAD),
		))
	return boxes


def is_mostly_hand(bounds, hand_boxes):
	"""True when this blob is a hand rather than a surface it is holding.

	Judged by how much of the blob sits inside a hand box: a pale hand on its own
	is almost entirely inside one, while a held sheet only overlaps at the grip.
	"""
	x1, y1, x2, y2 = bounds
	area = float(max(1, (x2 - x1) * (y2 - y1)))

	for hx1, hy1, hx2, hy2 in hand_boxes:
		overlap_w = max(0, min(x2, hx2) - max(x1, hx1))
		overlap_h = max(0, min(y2, hy2) - max(y1, hy1))
		if (overlap_w * overlap_h) / area >= HAND_OVERLAP_DROP:
			return True
	return False


def order_corners(points):
	"""Four corners in a fixed go-around order, starting from the top-left one.

	Sorting them makes corner 0 mean the same corner from frame to frame, which
	is what lets each corner be smoothed and tracked on its own.
	"""
	pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
	centre = pts.mean(axis=0)
	order = np.argsort(np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0]))
	pts = pts[order]
	return np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)


def fit_quad(contour, area_px):
	"""The blob's true four corners, falling back to its min-area rectangle.

	approxPolyDP is tried at increasing tolerances until the outline collapses to
	a convex quadrilateral that still accounts for the blob's area - that quad
	sits on the sheet's actual corners even when perspective has skewed it into a
	trapezoid. A blob that is not four-sided (a hand, a mug) keeps the rectangle.
	"""
	peri = cv2.arcLength(contour, True)
	if peri > 0:
		for eps in QUAD_EPS_STEPS:
			approx = cv2.approxPolyDP(contour, eps * peri, True)
			if len(approx) != 4 or not cv2.isContourConvex(approx):
				continue
			quad_area = cv2.contourArea(approx)
			if quad_area > 0 and abs(quad_area - area_px) / area_px <= QUAD_AREA_TOL:
				return order_corners(approx), True

	return order_corners(cv2.boxPoints(cv2.minAreaRect(contour))), False


def quad_sides(corners):
	"""(long, short) side length in px, each averaged over its opposite pair."""
	edges = [
		float(np.linalg.norm(corners[(i + 1) % 4] - corners[i]))
		for i in range(4)
	]
	side_a = (edges[0] + edges[2]) / 2.0
	side_b = (edges[1] + edges[3]) / 2.0
	return max(side_a, side_b), min(side_a, side_b)


def find_white_regions(mask, hand_boxes=()):
	"""Biggest white blobs as (corners, area_px, is_quad), largest first."""
	frame_area = float(mask.shape[0] * mask.shape[1])
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	regions = []
	for contour in contours:
		area_px = cv2.contourArea(contour)
		if area_px < MIN_REGION_PX * KEEP_REGION_FRAC or area_px > MAX_REGION_FRAC * frame_area:
			continue

		x, y, w, h = cv2.boundingRect(contour)
		if is_mostly_hand((x, y, x + w, y + h), hand_boxes):
			continue

		corners, is_quad = fit_quad(contour, area_px)
		if min(quad_sides(corners)) < 1.0:
			continue

		regions.append((corners, area_px, is_quad))

	regions.sort(key=lambda r: r[1], reverse=True)
	return regions[:MAX_REGIONS]


def ema(old, new, alpha):
	return new if old is None else (1.0 - alpha) * old + alpha * new


def align_corners(previous, corners):
	"""Rotate the corner list onto the one it matched last frame.

	order_corners starts from the top-left corner, and on a sheet being turned
	that title passes from one physical corner to the next - without this the
	whole outline would appear to jump a quarter turn as it happened.
	"""
	best, best_cost = corners, None
	for shift in range(4):
		candidate = np.roll(corners, -shift, axis=0)
		cost = float(np.linalg.norm(candidate - previous, axis=1).sum())
		if best_cost is None or cost < best_cost:
			best, best_cost = candidate, cost
	return best


def blend_corners(previous, corners):
	"""Smooth each corner, but give way as soon as the sheet is really moving.

	Heavy smoothing is what stops a still sheet from twitching; applied while it
	is being moved it would drag the outline along behind the paper instead, so
	a corner travelling more than SNAP_DIST switches to following it.
	"""
	corners = align_corners(previous, corners)
	travel = float(np.linalg.norm(corners - previous, axis=1).mean())
	# Eased in with the travel rather than switched on at a line, so there is no
	# speed at which the outline visibly changes its mind about following.
	alpha = MEASURE_EMA + (SNAP_EMA - MEASURE_EMA) * min(1.0, travel / SNAP_DIST)
	return (1.0 - alpha) * previous + alpha * corners, alpha


class SurfaceTracker:
	"""Follows each white surface across frames so its readout can settle.

	Without this every frame is measured from scratch: blobs swap places in the
	size ordering, a single dropped frame blanks the label, and the millimetre
	figures twitch even when the sheet is lying perfectly still.
	"""

	def __init__(self):
		self.surfaces = []
		self.next_id = 1

	def update(self, regions):
		for surface in self.surfaces:
			surface["missed"] += 1

		for corners, area_px, is_quad in regions:
			centre = corners.mean(axis=0)
			match = self._nearest_free(centre[0], centre[1])

			if match is None:
				if area_px < MIN_REGION_PX:
					# Too small to acquire; only an already-tracked surface is
					# allowed to shrink this far before it is dropped.
					continue
				long_px, short_px = quad_sides(corners)
				self.surfaces.append({
					"id": self.next_id,
					"corners": corners,
					"cx": float(centre[0]), "cy": float(centre[1]),
					"long_px": long_px, "short_px": short_px,
					"area_px": area_px, "is_quad": is_quad,
					"missed": 0,
				})
				self.next_id += 1
				continue

			smoothed, alpha = blend_corners(match["corners"], corners)
			match["corners"] = smoothed
			centre = smoothed.mean(axis=0)
			match["cx"], match["cy"] = float(centre[0]), float(centre[1])
			# Sizes are read off the smoothed corners, so the numbers and the
			# outline can never disagree about where the sheet is.
			match["long_px"], match["short_px"] = quad_sides(smoothed)
			match["area_px"] = ema(match["area_px"], area_px, alpha)
			match["is_quad"] = is_quad
			match["missed"] = 0

		self.surfaces = [s for s in self.surfaces if s["missed"] <= TRACK_MISS_FRAMES]
		self.surfaces.sort(key=lambda s: s["area_px"], reverse=True)
		return self.surfaces

	def _nearest_free(self, cx, cy):
		best = None
		best_dist = TRACK_MATCH_DIST
		for surface in self.surfaces:
			if surface["missed"] == 0:
				continue        # already claimed by another blob this frame
			dist = math.hypot(surface["cx"] - cx, surface["cy"] - cy)
			if dist < best_dist:
				best, best_dist = surface, dist
		return best


def surface_quad(surface):
	return surface["corners"].astype(np.int32)


def draw_label(frame, text, x, y, bg_color, text_color=COLOR_WHITE):
	(text_w, text_h), baseline = cv2.getTextSize(text, FONT, 0.5, 1)
	cv2.rectangle(
		frame,
		(x, y - text_h - baseline - 4),
		(x + text_w + 6, y),
		bg_color,
		-1,
	)
	cv2.putText(frame, text, (x + 3, y - 4), FONT, 0.5, text_color, 1, cv2.LINE_AA)


def draw_banner(frame, white_px, total_px, region_count, mm_per_px, fps,
				sat_max, val_min, auto_white, height_mm):
	cv2.rectangle(frame, (0, 0), (frame.shape[1], BANNER_H), COLOR_BLACK, -1)

	percent = 100.0 * white_px / max(1, total_px)
	scale_text = f"auto scale {mm_per_px:.3f} mm/px @ {height_mm:.0f} mm"
	gate = f"{'auto' if auto_white else 'manual'} S<={sat_max} V>={val_min}"

	cv2.putText(frame, f"White pixels: {white_px:,} / {total_px:,}  ({percent:.2f}%)",
				(10, 20), FONT, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)
	cv2.putText(frame, f"White surfaces: {region_count}    {scale_text}",
				(10, 38), FONT, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)
	cv2.putText(frame, f"{gate}    FPS {fps:.1f}",
				(10, 56), FONT, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)


def draw_help(frame):
	cv2.putText(
		frame,
		"t: auto/manual   S: a/d   V: w/s   height: [ ]   m: mask   q: quit",
		(10, frame.shape[0] - 10),
		FONT,
		0.45,
		COLOR_YELLOW,
		1,
		cv2.LINE_AA,
	)



cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

prev_time = time.perf_counter()

sat_max = WHITE_SAT_MAX
val_min = WHITE_VAL_MIN          # used as-is in manual mode
val_bias = 0                     # nudges the automatic gate up or down
auto_white = AUTO_WHITE
show_mask = False

tracker = SurfaceTracker()
mask_accum = None

# Calibration is a standing property of the mounting, so it is worked out once
# here and only recomputed if the working distance is nudged or the camera hands
# back a different frame size than it was asked for.
camera_height_mm = CAMERA_HEIGHT_MM
mm_per_px = None
scale_frame_w = None

while True:
	ret, frame = cap.read()
	if not ret:
		break

	if FLIP_VIEW:
		frame = cv2.flip(frame, 1)

	# Keep an un-annotated copy: the white mask is built from the image, and the
	# skeletons and labels drawn below are themselves bright white pixels that
	# would otherwise be measured as surfaces.
	clean_frame = frame.copy()
	frame_h, frame_w = frame.shape[:2]

	# The camera rarely gives back exactly the size it was asked for, and the
	# scale is per-pixel, so it is derived from the frame that actually arrived.
	if mm_per_px is None or scale_frame_w != frame_w:
		mm_per_px = compute_mm_per_pixel(frame_w, camera_height_mm)
		scale_frame_w = frame_w

	rgb_frame = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)
	rgb_frame.flags.writeable = False
	hand_results = hands.process(rgb_frame)
	rgb_frame.flags.writeable = True

	# Hands are still tracked, but only so a bare hand in frame is not measured
	# as if it were a white surface. Nothing about the scale depends on them any
	# more, so the measurements are live whether or not anyone is at the bench.
	hand_boxes = []
	if hand_results.multi_hand_landmarks:
		for hand_landmarks in hand_results.multi_hand_landmarks:
			mp_drawing.draw_landmarks(
				frame,
				hand_landmarks,
				mp_hands.HAND_CONNECTIONS,
				mp_drawing_styles.get_default_hand_landmarks_style(),
				mp_drawing_styles.get_default_hand_connections_style(),
			)

			hand_boxes.append(
				compute_hand_bbox(hand_landmarks.landmark, frame_w, frame_h)
			)

	if auto_white:
		value_channel = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2HSV)[:, :, 2]
		gate = max(0, min(255, auto_val_threshold(value_channel) + val_bias))
	else:
		gate = val_min

	mask = white_mask(clean_frame, sat_max, gate)

	if mask_accum is None or mask_accum.shape != mask.shape:
		mask_accum = mask.astype(np.float32) / 255.0
	mask = smooth_mask(mask_accum, mask)

	# The hand stays in the mask - cutting it out used to bite a notch into any
	# sheet being held - but it is kept out of the reported pixel count, and
	# find_white_regions drops any blob that is mostly hand.
	hand_areas = hand_regions_mask(mask.shape, hand_boxes)
	counted = mask.copy()
	for hx1, hy1, hx2, hy2 in hand_areas:
		counted[hy1:hy2, hx1:hx2] = 0
	white_px = int(cv2.countNonZero(counted))
	total_px = frame_h * frame_w

	if show_mask:
		# Look at exactly which pixels are being counted while tuning S and V.
		view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
	else:
		view = frame
		# Tint the counted pixels so you can see what is being measured without
		# burying the surface itself under the overlay.
		view[mask > 0] = (
			(1.0 - MASK_TINT) * view[mask > 0] + MASK_TINT * np.array([0, 255, 0])
		).astype(np.uint8)

	surfaces = tracker.update(find_white_regions(mask, hand_areas))

	for i, surface in enumerate(surfaces, start=1):
		quad = surface_quad(surface)
		# A surface being coasted through a dropped frame is drawn dimmer, so a
		# stale reading never looks like a live one.
		live = surface["missed"] == 0
		outline = COLOR_GREEN if live else COLOR_YELLOW
		cv2.polylines(view, [quad], True, outline, 2 if live else 1)

		# Corner dots make it obvious at a glance whether the outline actually
		# landed on the sheet, and which corner the measurement ran between.
		for corner in quad:
			cv2.circle(view, (int(corner[0]), int(corner[1])), 4, outline, -1)
			cv2.circle(view, (int(corner[0]), int(corner[1])), 4, COLOR_BLACK, 1)

		# No four corners could be fitted, so the outline is a bounding rectangle
		# and the figures are only as good as that - say so rather than imply a
		# precision the shape does not support.
		fit_note = "" if surface["is_quad"] else " (approx)"

		# The scale is always available now, so there is no "waiting to calibrate"
		# state to fall back to - every surface is measured the frame it appears.
		long_mm = surface["long_px"] * mm_per_px
		short_mm = surface["short_px"] * mm_per_px
		# Area comes from the blob's own pixels, not from the outline, so an
		# L-shaped or rounded surface is not reported as its bounding box.
		area_mm2 = surface["area_px"] * (mm_per_px ** 2)
		size_text = (f"WHITE #{surface['id']}  {long_mm:.0f} x "
					 f"{short_mm:.0f} mm{fit_note}")
		area_text = f"area: {area_mm2:,.0f} mm^2  ({area_mm2 / 100.0:,.1f} cm^2)"

		# Keep the two label lines on screen and clear of the banner, whatever
		# corner of the frame the surface is sitting in.
		label_x = min(max(0, int(quad[:, 0].min())), max(0, frame_w - 260))
		label_y = min(max(BANNER_H + 18, int(quad[:, 1].min())), frame_h - 26)
		draw_label(view, size_text, label_x, label_y, outline, COLOR_BLACK)
		draw_label(
			view,
			area_text,
			label_x,
			min(frame_h - 2, label_y + 20),
			outline,
			COLOR_BLACK,
		)

	now = time.perf_counter()
	fps = 1.0 / max(1e-6, now - prev_time)
	prev_time = now

	draw_banner(
		view,
		white_px,
		total_px,
		len(surfaces),
		mm_per_px,
		fps,
		sat_max,
		gate,
		auto_white,
		camera_height_mm,
	)
	draw_help(view)

	cv2.imshow(WINDOW_NAME, view)

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
	elif key == ord("m"):
		show_mask = not show_mask
	elif key == ord("t"):
		auto_white = not auto_white
		val_bias = 0
	elif key == ord("d"):
		sat_max = min(255, sat_max + SAT_STEP)
	elif key == ord("a"):
		sat_max = max(5, sat_max - SAT_STEP)
	elif key == ord("w"):
		if auto_white:
			val_bias += VAL_BIAS_STEP
		else:
			val_min = min(255, val_min + VAL_BIAS_STEP)
	elif key == ord("s"):
		if auto_white:
			val_bias -= VAL_BIAS_STEP
		else:
			val_min = max(0, val_min - VAL_BIAS_STEP)
	elif key in (ord("]"), ord("[")):
		# Trim the calibration against something of known size: raising the
		# assumed working distance widens the field the frame is said to cover,
		# so every measurement grows with it, and lowering it shrinks them.
		step = HEIGHT_STEP_MM if key == ord("]") else -HEIGHT_STEP_MM
		camera_height_mm = max(MIN_CAMERA_HEIGHT_MM, camera_height_mm + step)
		mm_per_px = compute_mm_per_pixel(scale_frame_w, camera_height_mm)



cap.release()
cv2.destroyAllWindows()

import json
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
FRAME_WIDTH = 1280
FRAME_HEIGHT = 920
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
CAMERA_HEIGHT_MM = 400.0         # fallback only, before planar calibration
CAMERA_HFOV_DEG = 60.0           # fallback only, before planar calibration

# --- PLANAR WORKSPACE CALIBRATION ---
# Put ONE known rectangular sheet flat on the table, let the program detect it,
# then press C.  The reference is half an A4 sheet (A5): 210 x 148.5 mm.
# You may replace these two numbers with any accurately-known rectangle.
# For best accuracy, use a reference rectangle that covers a large part of the
# camera view. Once calibrated, ANY flat rectangular paper can be measured in mm
# without entering its size. Do not move the camera after calibration.
CALIB_REFERENCE_LONG_MM = 210.0
CALIB_REFERENCE_SHORT_MM = 148.5
CALIBRATION_FILE = os.path.join(
	os.path.dirname(os.path.abspath(__file__)),
	"paper_scale.npz",
)

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

# --- BLACK POINT: THE MARK THE ARM IS ACTUALLY AIMED AT ---
# The sheet says where the work surface is; the black mark on it says where on
# that surface the arm has to go. It is looked for only inside a tracked sheet,
# which is what makes it cheap and what keeps dark clutter on the table out.
#
# The mark is found by brightness rather than by a fixed threshold: the sheet is
# overwhelmingly white, so the median of its own pixels is the paper, and the
# mark is whatever sits far below that. A gate derived from the sheet rides the
# exposure the same way the white gate does, so it survives the webcam breathing.
DOT_DARK_FRACTION = 0.55         # dark = below this share of the sheet's brightness
DOT_GATE_FLOOR = 20
DOT_GATE_CEIL = 200
DOT_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# The interior is pulled in from the outline before it is searched, because the
# sheet's own edge shadow is darker than the paper and would read as a mark. The
# cost is that a mark within this many pixels of the edge is not seen.
DOT_EDGE_INSET_PX = 10

DOT_MIN_AREA_PX = 12             # smaller than this is sensor speckle
DOT_MAX_AREA_FRAC = 0.05         # bigger than this share of the sheet is not a mark
DOT_MIN_CIRCULARITY = 0.55       # 4*pi*A/P^2; printed text is far from round

# Same trade as the corners: smooth hard while the mark is still so the reported
# millimetres do not twitch, then give way as soon as it is genuinely moving so
# the readout does not trail the paper.
DOT_EMA = 0.35
DOT_SNAP_EMA = 0.90
DOT_SNAP_DIST = 6.0              # px of travel that counts as "being moved"
DOT_MISS_FRAMES = 8              # keep the last position this long after it drops

# Nothing is emitted until the mark has actually gone somewhere, so a still
# sheet does not flood the arm with a new target every frame.
DOT_EMIT_MM = 1.0

BANNER_H = 62                    # labels are kept clear of it
MASK_TINT = 0.30                 # light enough to still read the surface under it

COLOR_GREEN = (0, 200, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 215, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_MAGENTA = (255, 0, 255)
COLOR_CYAN = (255, 255, 0)
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



def draw_zero_point(frame, corners):
	"""Mark tracked physical corner P0 as the local zero point (0, 0).

	When a surface is first acquired, P0 starts as its top-left image vertex.
	After that, SurfaceTracker/blend_corners preserves the corner ordering from
	frame to frame, so P0 follows that same physical vertex while the paper moves
	or rotates instead of being re-selected as the current image top-left.
	"""
	tracked = np.asarray(corners, dtype=np.float32).reshape(4, 2)
	p0 = tracked[0]

	x0 = int(round(float(p0[0])))
	y0 = int(round(float(p0[1])))

	# Draw a clear origin marker.
	cv2.circle(frame, (x0, y0), 10, COLOR_RED, -1)
	cv2.circle(frame, (x0, y0), 13, COLOR_WHITE, 2)

	# Keep the text inside the image if P0 is close to the top edge.
	text_y = y0 - 14 if y0 > BANNER_H + 28 else y0 + 28
	cv2.putText(
		frame,
		"P0 ZERO (0,0)",
		(x0 + 14, text_y),
		FONT,
		0.55,
		COLOR_RED,
		2,
		cv2.LINE_AA,
	)

	return p0

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
	"""(long, short) side length, averaged over opposite pairs."""
	edges = [
		float(np.linalg.norm(corners[(i + 1) % 4] - corners[i]))
		for i in range(4)
	]
	side_a = (edges[0] + edges[2]) / 2.0
	side_b = (edges[1] + edges[3]) / 2.0
	return max(side_a, side_b), min(side_a, side_b)


def calibrate_mm_per_pixel(corners):
	"""Calibrate a fixed top-down webcam from one known half-A4 sheet.

	The reference sheet is used ONLY to learn the physical scale.  The returned
	value is millimetres per image pixel; later papers are measured from THEIR OWN
	detected pixel dimensions, so a smaller paper cannot be forced to the calibration-reference size.
	"""
	long_px, short_px = quad_sides(np.asarray(corners, dtype=np.float32))
	if long_px <= 1.0 or short_px <= 1.0:
		return None
	scale_long = CALIB_REFERENCE_LONG_MM / long_px
	scale_short = CALIB_REFERENCE_SHORT_MM / short_px
	# A top-down webcam should give nearly the same scale in both directions.
	# Averaging reduces small corner-detection noise.
	return float((scale_long + scale_short) / 2.0)


def load_workspace_calibration():
	if not os.path.exists(CALIBRATION_FILE):
		return None
	try:
		data = np.load(CALIBRATION_FILE)
		value = float(data["mm_per_px"])
		if np.isfinite(value) and value > 0:
			return value
	except Exception:
		pass
	return None


def save_workspace_calibration(mm_per_px):
	np.savez(CALIBRATION_FILE, mm_per_px=float(mm_per_px))


def measure_quad_mm(corners, calibrated_mm_per_px):
	"""Measure THIS detected paper from its own four corners."""
	long_px, short_px = quad_sides(np.asarray(corners, dtype=np.float32))
	long_mm = long_px * calibrated_mm_per_px
	short_mm = short_px * calibrated_mm_per_px
	area_mm2 = long_mm * short_mm
	return long_mm, short_mm, area_mm2

def workspace_mm(point, frame_width, mm_per_px):
	"""Where the point is on the table, in mm from the image's own fixed origin.

	The camera does not move, so an image pixel is a fixed spot on the table and
	pixels are millimetres once scaled. The preview is mirrored for the person
	watching it (FLIP_VIEW), and a mirrored X handed to an arm would send it the
	wrong way across the bench, so the flip is undone here rather than left as a
	trap for whoever wires the arm up.

	X grows to the right of the *unmirrored* view and Y grows downwards, matching
	the image. If the arm's axes differ, swap or negate them once, at the emit.
	"""
	x = (frame_width - 1 - float(point[0])) if FLIP_VIEW else float(point[0])
	return x * mm_per_px, float(point[1]) * mm_per_px


def paper_frame_mm(point, corners, mm_per_px):
	"""Where the point sits on the sheet itself, in mm from P0.

	The quad's own edges give the sheet's size in mm, so warping the quad onto
	that rectangle undoes the rotation and the perspective together: these
	coordinates describe a spot on the paper, and they stay put when the paper is
	moved or turned. That is what makes them the check on the workspace figures -
	a mark printed on the sheet keeps the same paper coordinates however far the
	sheet travels, so paper coordinates that jump mean the mark was re-acquired
	on something else, not that the sheet moved.
	"""
	quad = np.asarray(corners, dtype=np.float32).reshape(4, 2)
	side_a = (float(np.linalg.norm(quad[1] - quad[0]))
			  + float(np.linalg.norm(quad[2] - quad[3]))) / 2.0
	side_b = (float(np.linalg.norm(quad[3] - quad[0]))
			  + float(np.linalg.norm(quad[2] - quad[1]))) / 2.0
	if side_a <= 1.0 or side_b <= 1.0:
		return None

	width_mm = side_a * mm_per_px
	height_mm = side_b * mm_per_px
	target = np.array(
		[[0.0, 0.0], [width_mm, 0.0], [width_mm, height_mm], [0.0, height_mm]],
		dtype=np.float32,
	)

	try:
		transform = cv2.getPerspectiveTransform(quad, target)
	except cv2.error:
		return None

	mapped = cv2.perspectiveTransform(
		np.asarray(point, dtype=np.float32).reshape(1, 1, 2), transform
	)
	return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])


def paper_interior_mask(shape, corners, inset_px=DOT_EDGE_INSET_PX):
	"""The sheet's inside, pulled back from its own edge.

	Filled from the tracked outline rather than from the white mask, so the hole
	the mark itself punches in that mask cannot shrink the area being searched.
	"""
	interior = np.zeros(shape[:2], np.uint8)
	cv2.fillPoly(interior, [np.asarray(corners, dtype=np.float32).astype(np.int32)], (255,))
	if inset_px > 0:
		kernel = cv2.getStructuringElement(
			cv2.MORPH_ELLIPSE, (2 * inset_px + 1, 2 * inset_px + 1)
		)
		interior = cv2.erode(interior, kernel)
	return interior


def find_black_point(frame, interior):
	"""Centre of the darkest round blob on the sheet, sub-pixel, or None.

	Round is the discriminator that matters: printed text and a stray shadow are
	both dark, but neither is circular, so circularity is what separates a mark
	made on purpose from whatever else is on the page. Among the blobs that pass,
	the biggest and roundest wins - print speckle that survives the opening is
	still far smaller than a deliberate mark.
	"""
	paper_px = int(cv2.countNonZero(interior))
	if paper_px < MIN_REGION_PX * 0.25:
		return None

	value = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
	paper_value = float(np.median(value[interior > 0]))
	gate = int(min(DOT_GATE_CEIL, max(DOT_GATE_FLOOR, paper_value * DOT_DARK_FRACTION)))

	dark = np.where((value <= gate) & (interior > 0), 255, 0).astype(np.uint8)
	dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, DOT_OPEN_KERNEL)

	contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	best = None
	best_score = 0.0
	for contour in contours:
		area_px = cv2.contourArea(contour)
		if area_px < DOT_MIN_AREA_PX or area_px > DOT_MAX_AREA_FRAC * paper_px:
			continue

		perimeter = cv2.arcLength(contour, True)
		if perimeter <= 0:
			continue
		circularity = 4.0 * math.pi * area_px / (perimeter * perimeter)
		if circularity < DOT_MIN_CIRCULARITY:
			continue

		moments = cv2.moments(contour)
		if moments["m00"] <= 0:
			continue

		score = area_px * circularity
		if score > best_score:
			best_score = score
			# Moments give the centroid between pixels, which is what keeps a
			# slow drift readable instead of quantised to whole millimetres.
			best = np.array(
				[moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
				dtype=np.float32,
			)

	return best


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
	"""Match new corners to the same physical vertices tracked last frame.

	P0 is corner index 0.  We test every cyclic ordering in both directions and
	choose the ordering with the smallest total movement from the previous frame.
	This keeps P0 attached to the same physical paper corner during translation
	and rotation, provided the tracker does not lose the paper completely.
	"""
	previous = np.asarray(previous, dtype=np.float32).reshape(4, 2)
	corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)

	best, best_cost = corners.copy(), None
	for direction in (corners, corners[::-1]):
		for shift in range(4):
			candidate = np.roll(direction, -shift, axis=0)
			cost = float(np.linalg.norm(candidate - previous, axis=1).sum())
			if best_cost is None or cost < best_cost:
				best, best_cost = candidate.copy(), cost
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


class BlackPointTracker:
	"""Locks onto the black mark and measures how far it has since travelled.

	Two things are wanted from the mark and they are not the same thing: where it
	is on the table right now, which is the position the arm is sent to, and how
	far it is from where the lock was taken, which is what says the paper moved
	at all. Both are carried here so the emitted payload never has to be
	reassembled from state scattered around the loop.

	The lock is a table position, not a pixel one, so it survives the mark being
	lost for a few frames: the camera has not moved, so where the mark was is
	still where it was. It is dropped only on an explicit re-lock.
	"""

	def __init__(self):
		self.pos = None              # smoothed image position, for drawing
		self.world = None            # ...the same point on the table, in mm
		self.paper = None            # ...and where that is on the sheet, in mm
		self.missed = DOT_MISS_FRAMES + 1
		self.surface_id = None

		self.ref_world = None        # table position the lock was taken at
		self.ref_pos = None          # ...in pixels, so it can be drawn
		self.ref_paper = None
		self.dx_mm = 0.0
		self.dy_mm = 0.0
		self.moved_mm = 0.0
		self.emitted_world = None

	@property
	def live(self):
		return self.pos is not None and self.missed == 0

	def update(self, detection, surface, frame_width, mm_per_px):
		if detection is None:
			self.missed += 1
			if self.missed > DOT_MISS_FRAMES:
				# Past this the last position is stale enough to be misleading,
				# so it stops being reported. The lock stays: it is a place on
				# the table, and losing sight of the mark did not move it.
				self.pos = None
				self.world = None
				self.paper = None
			return

		point = np.asarray(detection, dtype=np.float32).reshape(2)
		if self.pos is None:
			self.pos = point
		else:
			travel = float(np.linalg.norm(point - self.pos))
			alpha = DOT_EMA + (DOT_SNAP_EMA - DOT_EMA) * min(1.0, travel / DOT_SNAP_DIST)
			self.pos = (1.0 - alpha) * self.pos + alpha * point
		self.missed = 0
		self.surface_id = surface["id"]

		self.world = workspace_mm(self.pos, frame_width, mm_per_px)
		self.paper = paper_frame_mm(self.pos, surface["corners"], mm_per_px)

		if self.ref_world is None:
			self.lock()
		else:
			self.dx_mm = self.world[0] - self.ref_world[0]
			self.dy_mm = self.world[1] - self.ref_world[1]
			self.moved_mm = math.hypot(self.dx_mm, self.dy_mm)

	def lock(self):
		"""Take the current position as the zero the movement is measured from."""
		if self.pos is None or self.world is None:
			return False
		self.ref_pos = self.pos.copy()
		self.ref_world = self.world
		self.ref_paper = self.paper
		self.dx_mm = self.dy_mm = self.moved_mm = 0.0
		self.emitted_world = None
		return True

	def pending_move(self):
		"""The move worth telling the arm about, or None while it is still.

		Measured against what was last emitted rather than against the lock, so a
		sheet nudged across the bench in steps reports each step, and a sheet
		sitting still reports nothing however far from the lock it ended up.
		"""
		if not self.live or self.world is None:
			return None
		if self.emitted_world is None:
			return self.world
		step = math.hypot(
			self.world[0] - self.emitted_world[0],
			self.world[1] - self.emitted_world[1],
		)
		return self.world if step >= DOT_EMIT_MM else None


def emit_black_point(tracker, calibrated):
	"""Hand the arm the mark's new placement - one JSON line per real move.

	Printed rather than sent: the link to the arm is yours to make, and this is
	the point to make it. Replace the print with the write to your serial port or
	socket and the payload is already in the shape the arm needs - absolute table
	position to drive to, and the displacement since the last one it was given.

	x_mm/y_mm are on the table from the image origin; paper_x_mm/paper_y_mm are on
	the sheet from P0. A mark printed on the sheet holds its paper coordinates
	however far the sheet is pushed, so those two disagreeing is the signal that
	the lock jumped to a different mark and the move should not be trusted.
	"""
	moved_to = tracker.pending_move()
	if moved_to is None:
		return None

	payload = {
		"t": round(time.time(), 3),
		"surface": tracker.surface_id,
		"x_mm": round(moved_to[0], 2),
		"y_mm": round(moved_to[1], 2),
		"dx_mm": round(tracker.dx_mm, 2),
		"dy_mm": round(tracker.dy_mm, 2),
		"moved_mm": round(tracker.moved_mm, 2),
		"paper_x_mm": round(tracker.paper[0], 2) if tracker.paper else None,
		"paper_y_mm": round(tracker.paper[1], 2) if tracker.paper else None,
		# Uncalibrated millimetres come from the height/HFOV guess and are only
		# as good as that guess, so the arm is told which it is being given.
		"calibrated": bool(calibrated),
	}
	print(json.dumps(payload), flush=True)
	tracker.emitted_world = moved_to
	return payload


def draw_black_point(view, tracker, frame_h):
	"""The mark, the lock it is measured from, and the travel between them."""
	x = int(round(float(tracker.pos[0])))
	y = int(round(float(tracker.pos[1])))

	if tracker.ref_pos is not None:
		rx = int(round(float(tracker.ref_pos[0])))
		ry = int(round(float(tracker.ref_pos[1])))
		cv2.circle(view, (rx, ry), 7, COLOR_CYAN, 1)
		cv2.drawMarker(view, (rx, ry), COLOR_CYAN, cv2.MARKER_TILTED_CROSS, 10, 1)
		# Drawn only once the travel is long enough to have a direction worth
		# showing; below that the arrowhead is just noise on top of the mark.
		if math.hypot(x - rx, y - ry) > 6.0:
			cv2.arrowedLine(view, (rx, ry), (x, y), COLOR_CYAN, 2, cv2.LINE_AA, tipLength=0.15)

	cv2.circle(view, (x, y), 10, COLOR_MAGENTA, 2)
	cv2.drawMarker(view, (x, y), COLOR_MAGENTA, cv2.MARKER_CROSS, 16, 1)

	label_x = min(max(0, x + 16), max(0, view.shape[1] - 250))
	label_y = min(max(BANNER_H + 18, y - 10), frame_h - 46)
	draw_label(
		view,
		f"PT  x {tracker.world[0]:.1f}  y {tracker.world[1]:.1f} mm",
		label_x,
		label_y,
		COLOR_MAGENTA,
		COLOR_BLACK,
	)
	draw_label(
		view,
		f"moved {tracker.moved_mm:.1f} mm  (dx {tracker.dx_mm:+.1f}  dy {tracker.dy_mm:+.1f})",
		label_x,
		min(frame_h - 24, label_y + 20),
		COLOR_MAGENTA,
		COLOR_BLACK,
	)
	if tracker.paper is not None:
		draw_label(
			view,
			f"on sheet from P0: {tracker.paper[0]:.1f}, {tracker.paper[1]:.1f} mm",
			label_x,
			min(frame_h - 2, label_y + 40),
			COLOR_MAGENTA,
			COLOR_BLACK,
		)


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
				sat_max, val_gate, auto_white, height_mm, calibrated_scale,
				val_bias=0):
	cv2.rectangle(frame, (0, 0), (frame.shape[1], BANNER_H), COLOR_BLACK, -1)

	percent = 100.0 * white_px / max(1, total_px)
	if calibrated_scale is not None:
		scale_text = f"CALIBRATED: {calibrated_scale:.4f} mm/px"
	else:
		scale_text = f"UNCALIBRATED estimate {mm_per_px:.3f} mm/px @ {height_mm:.0f} mm"

	if auto_white:
		# In auto mode the gate shown is Otsu's own figure plus the nudge, and
		# Otsu re-derives it from every frame - so the gate drifts on its own and
		# a few presses of w/s can be swallowed whole. The bias is printed on its
		# own for that reason: it is the part the keys actually control, and
		# without it a press that landed is indistinguishable from one that did
		# not.
		gate = f"auto S<={sat_max} V>={val_gate} (otsu {val_bias:+d})"
	else:
		gate = f"manual S<={sat_max} V>={val_gate}"

	cv2.putText(frame, f"White pixels: {white_px:,} / {total_px:,}  ({percent:.2f}%)",
				(10, 20), FONT, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)
	cv2.putText(frame, f"White surfaces: {region_count}    {scale_text}",
				(10, 38), FONT, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)
	cv2.putText(frame, f"{gate}    FPS {fps:.1f}",
				(10, 56), FONT, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)


def draw_help(frame):
	# Every key the loop answers to is listed. The S and V gates were tunable all
	# along but appeared on no line of the display, which is indistinguishable
	# from their not being tunable at all.
	cv2.putText(
		frame,
		"a/d: S gate   w/s: V gate   t: auto/manual V",
		(10, frame.shape[0] - 30),
		FONT,
		0.45,
		COLOR_YELLOW,
		1,
		cv2.LINE_AA,
	)
	cv2.putText(
		frame,
		"c: calibrate   x: clear calib   r: re-lock mark   m: mask   q: quit",
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
point_tracker = BlackPointTracker()
mask_accum = None

# Calibration is a standing property of the mounting, so it is worked out once
# here and only recomputed if the working distance is nudged or the camera hands
# back a different frame size than it was asked for.
camera_height_mm = CAMERA_HEIGHT_MM
mm_per_px = None
scale_frame_w = None
calibrated_mm_per_px = load_workspace_calibration()
calibration_message = "Loaded saved calibration" if calibrated_mm_per_px is not None else "Press C with the known reference sheet visible"
calibration_message_until = time.perf_counter() + 4.0

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

		# P0 starts as top-left only on acquisition; tracking then preserves that physical vertex.
		# It is visualized only for a live four-corner surface.
		if live:
			zero_point = draw_zero_point(view, surface["corners"])

		# Corner dots make it obvious at a glance whether the outline actually
		# landed on the sheet, and which corner the measurement ran between.
		for corner in quad:
			cv2.circle(view, (int(corner[0]), int(corner[1])), 4, outline, -1)
			cv2.circle(view, (int(corner[0]), int(corner[1])), 4, COLOR_BLACK, 1)

		# No four corners could be fitted, so the outline is a bounding rectangle
		# and the figures are only as good as that - say so rather than imply a
		# precision the shape does not support.
		fit_note = "" if surface["is_quad"] else " (approx)"

		# Once calibration exists, measure the CURRENT paper's own detected pixel
		# dimensions and multiply by the learned mm/pixel scale. The half-A4 reference is
		# not reused as the measurement; it only teaches the scale once.
		if calibrated_mm_per_px is not None:
			long_mm, short_mm, area_mm2 = measure_quad_mm(surface["corners"], calibrated_mm_per_px)
			mode_note = ""
		else:
			# Before calibration, retain the original height/HFOV estimate so the
			# program still gives a rough reading rather than no reading at all.
			long_mm = surface["long_px"] * mm_per_px
			short_mm = surface["short_px"] * mm_per_px
			area_mm2 = surface["area_px"] * (mm_per_px ** 2)
			mode_note = " (EST)"

		size_text = (f"WHITE #{surface['id']}  {long_mm:.0f} x "
					 f"{short_mm:.0f} mm{fit_note}{mode_note}")
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

	# The mark is looked for on one sheet - the biggest live one, which is the
	# sheet the bench is working on. Searching every tracked surface would only
	# raise the chance of locking onto a shadow on a sheet nobody cares about.
	target = next((s for s in surfaces if s["missed"] == 0), None)
	point_scale = calibrated_mm_per_px if calibrated_mm_per_px is not None else mm_per_px

	if target is None:
		point_tracker.update(None, None, frame_w, point_scale)
	else:
		interior = paper_interior_mask(mask.shape, target["corners"])
		# Read from the clean frame: the outline, the corner dots and the labels
		# drawn above are all real pixels, and the dark ones among them sit
		# exactly where a mark would be looked for.
		point_tracker.update(
			find_black_point(clean_frame, interior), target, frame_w, point_scale
		)

	if point_tracker.live:
		draw_black_point(view, point_tracker, frame_h)
		emit_black_point(point_tracker, calibrated_mm_per_px is not None)
	elif target is not None:
		# Kept above the two help lines rather than on top of them.
		draw_label(view, "no black mark on the sheet", 10, frame_h - 50,
				   COLOR_MAGENTA, COLOR_BLACK)

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
		calibrated_mm_per_px,
		val_bias,
	)
	draw_help(view)

	if time.perf_counter() < calibration_message_until:
		draw_label(
			view,
			calibration_message,
			10,
			BANNER_H + 22,
			COLOR_YELLOW,
			COLOR_BLACK,
		)

	cv2.imshow(WINDOW_NAME, view)

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
	elif key == ord("m"):
		show_mask = not show_mask
	elif key == ord("r"):
		# Re-zero the travel here. The lock is taken automatically the first time
		# the mark is seen, so this is for after the sheet has been placed where
		# the run is actually meant to start from.
		if point_tracker.lock():
			calibration_message = "Black mark re-locked - travel measured from here"
		else:
			calibration_message = "Nothing to lock onto: no black mark visible"
		calibration_message_until = time.perf_counter() + 3.0
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
	elif key == ord("c"):
		# IMPORTANT: C is ONLY for one-time calibration with a known half-A4 reference.
		# Once calibrated, do NOT recalibrate on every paper. Otherwise every sheet
		# would be forced to the reference size (210 x 148.5 mm).
		if calibrated_mm_per_px is not None:
			calibration_message = "Already calibrated - press X first only if the camera moved"
		else:
			live_surfaces = [s for s in surfaces if s["missed"] == 0 and s["is_quad"]]
			if not live_surfaces:
				calibration_message = "Calibration failed: show ONE clear half-A4 reference sheet (210 x 148.5 mm)"
			else:
				reference = max(live_surfaces, key=lambda s: s["area_px"])
				calibrated_mm_per_px = calibrate_mm_per_pixel(reference["corners"])
				if calibrated_mm_per_px is None:
					calibration_message = "Calibration failed: could not measure half-A4 corners"
					calibration_message_until = time.perf_counter() + 5.0
					continue
				save_workspace_calibration(calibrated_mm_per_px)
				# Forget the calibration-sheet geometry immediately. The next detected sheet must be
				# measured from its own current corners, not blended with the reference.
				tracker = SurfaceTracker()
				# The lock was taken in the old estimated millimetres, so it is
				# not a distance in the new ones. Dropped rather than converted:
				# the reference sheet is about to be lifted off anyway.
				point_tracker = BlackPointTracker()
				mask_accum = None
				calibration_message = (
					f"Calibration locked: half A4 = {CALIB_REFERENCE_LONG_MM:.0f} x "
					f"{CALIB_REFERENCE_SHORT_MM:.0f} mm. Now remove it and measure other papers."
				)
		calibration_message_until = time.perf_counter() + 5.0
	elif key == ord("x"):
		calibrated_mm_per_px = None
		point_tracker = BlackPointTracker()     # same reason as on calibrate
		try:
			os.remove(CALIBRATION_FILE)
		except FileNotFoundError:
			pass
		calibration_message = "Calibration cleared - press C with the reference sheet visible"
		calibration_message_until = time.perf_counter() + 4.0



cap.release()
cv2.destroyAllWindows()

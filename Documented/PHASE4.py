"""
PHASE4.py
=========
Same vision-based safety logic as PHASE3, but instead of driving a real Arduino
over serial it adds a SIMULATED 2-link robotic arm in a second OpenCV window.

  * YOLO + MediaPipe -> decide GREEN (safe) / RED (stop) exactly as before.
  * render_arm()      -> draws an animated arm that sweeps continuously while
                         safe and FREEZES when a hand enters the red zone.

This lets you demo the whole system on a laptop with no hardware attached.
"""

import math                                       # math: trig (sin/cos) for the arm's joint motion.
import time                                        # time: FPS timing.

import cv2                                         # OpenCV: camera, drawing, windows.
import numpy as np                                 # NumPy: build the blank arm canvas as an array.

from ultralytics import YOLO                        # YOLOv8 object detection.

import mediapipe as mp                              # MediaPipe Hands.




# =========================
# Configuration constants
# =========================
CAM_INDEX = 0                                       # Default webcam.
FRAME_WIDTH = 640                                   # Capture width.
FRAME_HEIGHT = 460                                  # Capture height.
FLIP_VIEW = True                                    # Mirror image (selfie view).

WINDOW_NAME = "Cobot Vision Safety Monitor"         # Main camera window title.

YOLO_MODEL_PATH = "yolov8s.pt"                      # YOLO weights.
YOLO_CONF_THRESHOLD = 0.5                           # Min detection confidence.
YOLO_DEVICE = "cuda"  # Use None for auto, or "cpu" / "0" for GPU index   # Inference device.

MAX_HANDS = 2                                        # Track up to two hands.
MODEL_COMPLEXITY = 0                                 # Lightest hand model.
MIN_HAND_DET_CONF = 0.6                              # Hand detection gate.
MIN_HAND_TRACK_CONF = 0.5                            # Hand tracking gate.

HAND_AREA_THRESHOLD = 40_000                         # Starting "too close" area.
THRESHOLD_STEP    = 2_000                            # Adjustment per keypress.
THRESHOLD_MAX     = 150_000                          # Upper clamp.

# Robotic arm visualization (second window)
ARM_WINDOW_NAME = "Robotic Arm"                      # Title of the simulated-arm window.
ARM_CANVAS_W = 640                                   # Arm canvas width.
ARM_CANVAS_H = 460                                   # Arm canvas height.
ARM_BASE = (ARM_CANVAS_W // 2, ARM_CANVAS_H - 60)    # Pixel position of the arm's base (bottom-centre).
ARM_L1 = 150          # length of first link (px)    # First link (shoulder→elbow) length.
ARM_L2 = 120          # length of second link (px)   # Second link (elbow→hand) length.
ARM_SPEED = 0.04      # animation speed (radians per frame)  # How fast the sweep advances each frame.

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
    static_image_mode=False,                         # Video mode (tracking between frames).
    max_num_hands=MAX_HANDS,
    model_complexity=MODEL_COMPLEXITY,
    min_detection_confidence=MIN_HAND_DET_CONF,
    min_tracking_confidence=MIN_HAND_TRACK_CONF,
)



def compute_hand_bbox(landmarks, frame_width, frame_height):  # Pixel bbox + area for one hand.
    xs = [lm.x for lm in landmarks]                  # Normalised X of every landmark.
    ys = [lm.y for lm in landmarks]                  # Normalised Y of every landmark.

    min_x = max(0, int(min(xs) * frame_width))       # Left edge (clamped >= 0).
    max_x = min(frame_width - 1, int(max(xs) * frame_width))   # Right edge (clamped in-frame).
    min_y = max(0, int(min(ys) * frame_height))      # Top edge.
    max_y = min(frame_height - 1, int(max(ys) * frame_height)) # Bottom edge.

    box_w = max(0, max_x - min_x)                    # Width.
    box_h = max(0, max_y - min_y)                    # Height.
    area = box_w * box_h                             # Proximity proxy.

    return min_x, min_y, max_x, max_y, area          # Corners + area.


def draw_label(frame, text, x, y, bg_color, text_color=COLOR_WHITE):  # Text on a filled chip.
    (text_w, text_h), baseline = cv2.getTextSize(    # Measure text to size the chip.
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    cv2.rectangle(                                   # Filled background chip.
        frame,
        (x, y - text_h - baseline - 4),
        (x + text_w + 6, y),
        bg_color,
        -1,
    )
    cv2.putText(                                     # Text over the chip.
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
    bar_x    = w - 22                                # Gauge X near right edge.
    bar_top  = 70                                    # Gauge top.
    bar_bot  = h - 55                                # Gauge bottom.
    bar_len  = bar_bot - bar_top                     # Track length.
    max_area = THRESHOLD_MAX                          # Value at full gauge.

    cv2.rectangle(frame, (bar_x - 6, bar_top), (bar_x + 6, bar_bot), COLOR_BLACK,   -1)  # Track bg.

    ratio  = min(threshold / max_area, 1.0)          # Fill fraction.
    fill_y = int(bar_bot - ratio * bar_len)          # Fill top (grows up).
    cv2.rectangle(frame, (bar_x - 5, fill_y), (bar_x + 5, bar_bot), COLOR_YELLOW, -1)    # Yellow fill.

    cv2.putText(frame, "THR", (bar_x - 14, bar_top - 8),  # Gauge label.
                FONT, 0.38, COLOR_BLACK, 1, cv2.LINE_AA)


def draw_safety_banner(frame, safety_state, fps, threshold, collision_risk):  # Top banner + HUD.
    banner_h = 50                                    # Banner height.
    color = COLOR_GREEN if safety_state == "GREEN" else COLOR_RED  # Colour by state.
    cv2.rectangle(frame, (0, 0), (frame.shape[1], banner_h), color, -1)  # Banner fill.

    state_text = "GREEN ZONE - SAFE" if safety_state == "GREEN" else "RED ZONE - STOP"  # Banner text.
    cv2.putText(                                     # Big state label.
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

    if collision_risk:                               # If both a close hand AND objects are present...
        cv2.putText(                                 # ...draw the collision warning.
            frame,
            "COLLISION RISK",
            (10, banner_h + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            COLOR_RED,
            2,
            cv2.LINE_AA,
        )


def render_arm(phase: float, stopped: bool):
    """Draw a 2-link robotic arm reaching toward a red target zone.

    `phase` drives a continuous sweeping motion. When `stopped` is True the
    arm is frozen (the hand has reached the red zone) and the banner shows STOP.
    """
    canvas = np.full((ARM_CANVAS_H, ARM_CANVAS_W, 3), 40, dtype=np.uint8)  # Dark-gray blank image (value 40).

    # Red target zone (top-right) the arm reaches toward.
    zone_tl = (ARM_CANVAS_W - 170, 40)               # Top-left corner of the target zone.
    zone_br = (ARM_CANVAS_W - 40, 170)               # Bottom-right corner of the target zone.
    zone_color = COLOR_RED if stopped else (40, 40, 120)  # Bright red when stopped, dim red otherwise.
    cv2.rectangle(canvas, zone_tl, zone_br, zone_color, -1)  # Filled zone.
    cv2.rectangle(canvas, zone_tl, zone_br, COLOR_RED, 2)    # Red outline.
    cv2.putText(canvas, "RED ZONE", (zone_tl[0] - 4, zone_tl[1] - 8),  # Zone label.
                FONT, 0.5, COLOR_RED, 2, cv2.LINE_AA)

    # Joint angles oscillate so the arm continuously moves while safe.
    base_angle = -math.pi / 2          # point upward   # -90° → arm points straight up by default.
    a1 = base_angle + 0.6 * math.sin(phase)          # Shoulder angle sways ±0.6 rad with the phase.
    a2 = -0.8 + 0.7 * math.sin(phase * 1.3)          # Elbow angle sways at a different rate (organic motion).

    bx, by = ARM_BASE                                # Base pixel position.
    jx = int(bx + ARM_L1 * math.cos(a1))             # Elbow X = base + link1 along angle a1.
    jy = int(by + ARM_L1 * math.sin(a1))             # Elbow Y.
    ex = int(jx + ARM_L2 * math.cos(a1 + a2))        # Hand X = elbow + link2 along combined angle.
    ey = int(jy + ARM_L2 * math.sin(a1 + a2))        # Hand Y.

    arm_color = COLOR_RED if stopped else COLOR_GREEN  # Arm turns red when stopped.

    # Base
    cv2.circle(canvas, (bx, by), 14, COLOR_WHITE, -1)   # Draw the base joint.
    # Link 1
    cv2.line(canvas, (bx, by), (jx, jy), arm_color, 10) # Draw the upper arm (thick line).
    # Joint
    cv2.circle(canvas, (jx, jy), 9, COLOR_WHITE, -1)    # Draw the elbow joint.
    # Link 2
    cv2.line(canvas, (jx, jy), (ex, ey), arm_color, 8)  # Draw the forearm.
    # End effector / hand
    cv2.circle(canvas, (ex, ey), 11, COLOR_YELLOW, -1)  # Draw the gripper (yellow fill)...
    cv2.circle(canvas, (ex, ey), 11, arm_color, 2)      # ...with a coloured outline.

    status = "STOPPED - HAND IN RED ZONE" if stopped else "MOVING"  # Status text.
    cv2.rectangle(canvas, (0, 0), (ARM_CANVAS_W, 36), arm_color, -1)  # Status banner.
    cv2.putText(canvas, status, (10, 26), FONT, 0.7, COLOR_WHITE, 2, cv2.LINE_AA)  # Status label.

    return canvas                                    # Hand the finished arm image back to the caller.



cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)     # Open camera (DirectShow backend on Windows).
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)       # Set width.
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)     # Set height.

prev_time = time.perf_counter()                      # Seed FPS timer.
last_state = None                                    # Track state changes for logging.

threshold = HAND_AREA_THRESHOLD                      # Live threshold.
arm_phase = 0.0                                      # Animation phase accumulator for the arm.

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

    yolo_results = yolo_model.predict(frame, **yolo_kwargs)  # Run object detection.
    objects_detected = False                         # Did YOLO find anything?

    for result in yolo_results:                      # For each result...
        names = result.names                         # id → name table.
        for box in result.boxes:                     # For each detected box...
            conf = float(box.conf[0])                # Confidence.
            if conf < YOLO_CONF_THRESHOLD:           # Skip low-confidence.
                continue
            cls_id = int(box.cls[0])                 # Class id.
            label = f"{names.get(cls_id, str(cls_id))} {conf * 100:.1f}%"  # "name conf%".
            x1, y1, x2, y2 = box.xyxy[0].tolist()    # Corners as floats.
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))  # → ints.

            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_YELLOW, 2)  # Draw object box.
            draw_label(frame, label, x1, y1, COLOR_BLACK)             # Draw its label.
            objects_detected = True                  # Flag detection.

    # MediaPipe Hands detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR → RGB for MediaPipe.
    rgb_frame.flags.writeable = False                # Read-only → faster.
    hand_results = hands.process(rgb_frame)          # Detect/track hands.
    rgb_frame.flags.writeable = True                 # Restore.

    safety_state = "GREEN"                           # Assume safe.
    frame_h, frame_w = frame.shape[:2]               # Frame dims.

    if hand_results.multi_hand_landmarks:            # If hands found...
        for hand_landmarks in hand_results.multi_hand_landmarks:  # For each...
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

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)        # Draw hand box.
            draw_label(frame, f"Hand area: {area}", x1, y1, box_color)    # Show area.

            if area > threshold:                     # If too close...
                safety_state = "RED"                 # ...go RED.

    collision_risk = safety_state == "RED" and objects_detected  # RED + objects = collision risk.

    # Decision output
    if safety_state != last_state:                   # On state change only...
        if safety_state == "RED":
            print("ROBOT STOP - HUMAN TOO CLOSE")    # Log stop.
        else:
            print("ROBOT SAFE - OPERATION ALLOWED")  # Log safe.
        last_state = safety_state                    # Remember new state.

    now = time.perf_counter()                        # Now...
    fps = 1.0 / max(1e-6, now - prev_time)           # ...FPS...
    prev_time = now                                  # ...save.

    draw_safety_banner(frame, safety_state, fps, threshold, collision_risk)  # Banner/HUD.
    draw_threshold_indicator(frame, threshold)       # Gauge.

    cv2.imshow(WINDOW_NAME, frame)                   # Show camera window.

    # Robotic arm window: keep moving until the hand reaches the red zone.
    arm_stopped = safety_state == "RED"              # Freeze the arm when unsafe.
    if not arm_stopped:                              # While safe...
        arm_phase += ARM_SPEED                       # ...advance the animation phase.
    cv2.imshow(ARM_WINDOW_NAME, render_arm(arm_phase, arm_stopped))  # Render + show the arm.

    key = cv2.waitKeyEx(1)                           # Read a key.

    if key == ord("q"):                              # 'q' → quit.
        break
    elif key in (2490368, 65362, 63232, ord("+")):   # Up/'+' → raise threshold.
        threshold = min(threshold + THRESHOLD_STEP, THRESHOLD_MAX)
    elif key in (2621440, 65364, 63233, ord("-")):   # Down/'-' → lower threshold.
        threshold = max(threshold - THRESHOLD_STEP, THRESHOLD_STEP)



cap.release()                                        # Release camera.
cv2.destroyAllWindows()                              # Close windows.
hands.close()                                        # Free MediaPipe resources.

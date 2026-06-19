"""
PHASE3.py
=========
Combines the two earlier phases into one safety loop AND adds hardware control:

  * YOLOv8  -> detects objects/people in the scene.
  * MediaPipe Hands -> measures how close a hand is (via bounding-box area).
  * pySerial -> sends 'G' (go/safe) or 'R' (red/stop) to an Arduino over COM4,
                so a real robot/LED reacts to the safety state.

If a hand crosses the area threshold the state becomes RED, a STOP command is
sent to the microcontroller, and (if YOLO also sees objects) a COLLISION RISK
warning is shown.
"""

import cv2                                        # OpenCV: camera, drawing, windows.

from ultralytics import YOLO                       # YOLOv8 object-detection model.

import mediapipe as mp                             # MediaPipe Hands for hand landmarks.


import serial                                      # pySerial: talk to the Arduino over a COM port.

import time                                        # Timing (startup delay + FPS).

ser = serial.Serial("COM4",9600)                   # Open serial port COM4 at 9600 baud (the Arduino link).

time.sleep(2)                                       # Wait 2 s for the Arduino to reset after the port opens.

# =========================
# Configuration constants
# =========================
CAM_INDEX = 0                                       # Default webcam.
FRAME_WIDTH = 640                                   # Requested capture width.
FRAME_HEIGHT = 460                                  # Requested capture height.
FLIP_VIEW = True                                    # Mirror the image (selfie view).

WINDOW_NAME = "Cobot Vision Safety Monitor"         # Display window title.

YOLO_MODEL_PATH = "yolov8s.pt"                      # Pre-trained YOLOv8-small weights.
YOLO_CONF_THRESHOLD = 0.5                           # Ignore detections under 50% confidence.
YOLO_DEVICE = "cuda"  # Use None for auto, or "cpu" / "0" for GPU index   # Run YOLO on the GPU.

MAX_HANDS = 2                                        # Track up to two hands.
MODEL_COMPLEXITY = 0                                 # Lightest hand model (fastest).
MIN_HAND_DET_CONF = 0.6                              # Hand detection confidence gate.
MIN_HAND_TRACK_CONF = 0.5                            # Hand tracking confidence gate.

HAND_AREA_THRESHOLD = 40_000                         # Starting "too close" area (px²).
THRESHOLD_STEP    = 2_000                            # Per-keypress adjustment.
THRESHOLD_MAX     = 150_000                          # Upper clamp for the threshold.

COLOR_GREEN = (0, 200, 0)                            # BGR colours used across overlays.
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 215, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX                      # Shared font.


yolo_model = YOLO(YOLO_MODEL_PATH)                   # Load the YOLO model once at startup.




mp_hands = mp.solutions.hands                        # MediaPipe Hands solution module.
mp_drawing = mp.solutions.drawing_utils              # Landmark drawing helpers.
mp_drawing_styles = mp.solutions.drawing_styles      # Default colour styles for hand skeletons.

hands = mp_hands.Hands(                              # Instantiate the hand tracker with our settings.
    static_image_mode=False,                         # False = video mode (uses tracking between frames).
    max_num_hands=MAX_HANDS,
    model_complexity=MODEL_COMPLEXITY,
    min_detection_confidence=MIN_HAND_DET_CONF,
    min_tracking_confidence=MIN_HAND_TRACK_CONF,
)



def compute_hand_bbox(landmarks, frame_width, frame_height):  # Pixel bounding box + area for one hand.
    xs = [lm.x for lm in landmarks]                  # Normalised (0..1) X of every landmark.
    ys = [lm.y for lm in landmarks]                  # Normalised (0..1) Y of every landmark.

    min_x = max(0, int(min(xs) * frame_width))       # Left edge in px, clamped to >= 0.
    max_x = min(frame_width - 1, int(max(xs) * frame_width))   # Right edge, clamped inside the frame.
    min_y = max(0, int(min(ys) * frame_height))      # Top edge, clamped to >= 0.
    max_y = min(frame_height - 1, int(max(ys) * frame_height)) # Bottom edge, clamped inside the frame.

    box_w = max(0, max_x - min_x)                    # Box width (never negative).
    box_h = max(0, max_y - min_y)                    # Box height (never negative).
    area = box_w * box_h                             # Area = the proximity proxy.

    return min_x, min_y, max_x, max_y, area          # Return corners + area to the caller.


def draw_label(frame, text, x, y, bg_color, text_color=COLOR_WHITE):  # Draw text on a filled chip.
    (text_w, text_h), baseline = cv2.getTextSize(    # Measure the text so the chip fits it...
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    cv2.rectangle(                                   # Draw the filled background rectangle (the chip)...
        frame,
        (x, y - text_h - baseline - 4),              # ...top-left, sized from the measured text...
        (x + text_w + 6, y),                         # ...bottom-right.
        bg_color,
        -1,                                          # -1 = filled.
    )
    cv2.putText(                                     # Draw the text on top of the chip.
        frame,
        text,
        (x + 3, y - 4),                              # Small padding inside the chip.
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        text_color,
        1,
        cv2.LINE_AA,                                 # Anti-aliased text (smoother).
    )

def draw_threshold_indicator(frame, threshold: int) -> None:
    """Vertical gauge on the right edge."""
    h, w     = frame.shape[:2]                       # Frame size.
    bar_x    = w - 22                                # Gauge X near the right edge.
    bar_top  = 70                                    # Gauge top Y.
    bar_bot  = h - 55                                # Gauge bottom Y.
    bar_len  = bar_bot - bar_top                     # Gauge track length.
    max_area = THRESHOLD_MAX                          # Value at a full gauge.

    cv2.rectangle(frame, (bar_x - 6, bar_top), (bar_x + 6, bar_bot), COLOR_BLACK,   -1)  # Track background.

    ratio  = min(threshold / max_area, 1.0)          # Fill fraction (max 100%).
    fill_y = int(bar_bot - ratio * bar_len)          # Top of the fill (grows upward).
    cv2.rectangle(frame, (bar_x - 5, fill_y), (bar_x + 5, bar_bot), COLOR_YELLOW, -1)    # Yellow fill.

    cv2.putText(frame, "THR", (bar_x - 14, bar_top - 8),  # Gauge label.
                FONT, 0.38, COLOR_BLACK, 1, cv2.LINE_AA)


def draw_safety_banner(frame, safety_state, fps, threshold, collision_risk):  # Top status banner + HUD text.
    banner_h = 50                                    # Banner height.
    color = COLOR_GREEN if safety_state == "GREEN" else COLOR_RED  # Colour follows the state.
    cv2.rectangle(frame, (0, 0), (frame.shape[1], banner_h), color, -1)  # Solid banner across the top.

    state_text = "GREEN ZONE - SAFE" if safety_state == "GREEN" else "RED ZONE - STOP"  # Banner text.
    cv2.putText(                                     # Draw the big state text.
        frame,
        state_text,
        (10, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        COLOR_WHITE,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(                                     # FPS readout (top-right).
        frame,
        f"FPS: {fps:.1f}",
        (frame.shape[1] - 140, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_WHITE,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(                                     # Current threshold readout (top-right, second line).
        frame,
        f"HAND_AREA_THRESHOLD: {threshold}",
        (frame.shape[1] - 280, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_WHITE,
        1,
        cv2.LINE_AA,
    )

    if collision_risk:                               # Only when a hand is close AND YOLO sees objects...
        cv2.putText(                                 # ...draw a prominent collision warning under the banner.
            frame,
            "COLLISION RISK",
            (10, banner_h + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            COLOR_RED,
            2,
            cv2.LINE_AA,
        )



cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)     # Open camera using DirectShow backend (Windows).
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)       # Request capture width.
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)     # Request capture height.

prev_time = time.perf_counter()                      # Seed FPS timer.
last_state = None                                    # Remember last safety state (to detect changes).
last_serial_state = None                             # (Reserved) last state sent over serial.

cmd = input("G or R: ")                              # Manual prompt at startup (initial command from the user).

threshold = HAND_AREA_THRESHOLD                      # Live, adjustable threshold.

while True:                                           # Main loop.
    ret, frame = cap.read()                          # Grab a frame.
    if not ret:                                      # Stop if capture failed.
        break

    if FLIP_VIEW:                                    # Optionally mirror the frame.
        frame = cv2.flip(frame, 1)

    # YOLO detection
    yolo_kwargs = {                                  # Build keyword args for YOLO inference.
        "conf": YOLO_CONF_THRESHOLD,                 # Confidence threshold.
        "verbose": False,                            # Suppress console spam.
    }
    if YOLO_DEVICE is not None:                      # If a device was specified...
        yolo_kwargs["device"] = YOLO_DEVICE          # ...pass it (e.g. "cuda").

    yolo_results = yolo_model.predict(frame, **yolo_kwargs)  # Run object detection on the frame.
    objects_detected = False                         # Track whether YOLO found anything this frame.

    for result in yolo_results:                      # Iterate YOLO result objects (usually one).
        names = result.names                         # id -> class-name lookup table.
        for box in result.boxes:                     # Each detected box.
            conf = float(box.conf[0])                # Detection confidence.
            if conf < YOLO_CONF_THRESHOLD:           # Extra safety: skip low-confidence boxes.
                continue
            cls_id = int(box.cls[0])                 # Class id.
            label = f"{names.get(cls_id, str(cls_id))} {conf * 100:.1f}%"  # "name conf%" label.
            x1, y1, x2, y2 = box.xyxy[0].tolist()    # Box corners as floats.
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))  # Convert to ints for drawing.

            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_YELLOW, 2)  # Draw the object box (yellow).
            draw_label(frame, label, x1, y1, COLOR_BLACK)             # Draw its label.
            objects_detected = True                  # Mark that we saw at least one object.


    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe.
    rgb_frame.flags.writeable = False                # Read-only → faster MediaPipe processing.
    hand_results = hands.process(rgb_frame)          # Detect/track hands.
    rgb_frame.flags.writeable = True                 # Restore writability.

    safety_state = "GREEN"                           # Assume safe each frame until proven otherwise.
    frame_h, frame_w = frame.shape[:2]               # Frame dimensions.
    hand_too_close = False                           # Whether any hand is over threshold.

    if hand_results.multi_hand_landmarks:            # If any hands were found...
        for hand_landmarks in hand_results.multi_hand_landmarks:  # For each hand...
            mp_drawing.draw_landmarks(               # Draw its skeleton with default styles.
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            x1, y1, x2, y2, area = compute_hand_bbox(  # Measure the hand's box + area.
                hand_landmarks.landmark, frame_w, frame_h
            )
            if area > threshold:                     # If the hand is too close...
                box_color = COLOR_RED                # ...draw it red...
                hand_too_close = True                # ...and flag it.
            else:
                box_color = COLOR_GREEN              # Otherwise green (safe).

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)        # Draw the hand box.
            draw_label(frame, f"Hand area: {area}", x1, y1, box_color)    # Show its measured area.

            if area > threshold:                     # Any close hand...
                safety_state = "RED"                 # ...forces the overall state to RED.

    collision_risk = safety_state == "RED" and objects_detected  # RED + objects present = collision risk.

    if safety_state != last_state:                   # Only act when the state actually CHANGES (debounce).

         if safety_state == "RED":                   # On transition into RED...
             print("ROBOT STOP - HUMAN TOO CLOSE")   # ...log it...
             ser.write(b"R")                         # ...and send 'R' (stop) to the Arduino.

         else:                                       # On transition into GREEN...
             print("ROBOT SAFE - OPERATION ALLOWED") # ...log it...
             ser.write(b"G")                         # ...and send 'G' (go) to the Arduino.

         last_state = safety_state                   # Remember the new state.

    now = time.perf_counter()                        # Timestamp now...
    fps = 1.0 / max(1e-6, now - prev_time)           # ...compute FPS (guard divide-by-zero).
    prev_time = now                                  # Save for next frame.

    draw_safety_banner(frame, safety_state, fps, threshold, collision_risk)  # Top banner + HUD.
    draw_threshold_indicator(frame, threshold)       # Right-edge gauge.

    cv2.imshow(WINDOW_NAME, frame)                   # Show the frame.

    # waitKeyEx returns the full keycode - arrow keys work on all OSes.
    # waitKey(1) & 0xFF only keeps the low byte, breaking arrow keys.
    #
    # Keycodes:  Windows   Linux    macOS
    #   Up       2490368   65362    63232
    #   Down     2621440   65364    63233
    # +/- keys work as a fallback on every platform.
    key = cv2.waitKeyEx(1)                           # Read a key (full code).

    if key == ord("q"):                              # 'q' → quit.
        break
    elif key in (2490368, 65362, 63232, ord("+")):   # Up/'+' → raise threshold...
        threshold = min(threshold + THRESHOLD_STEP, THRESHOLD_MAX)
    elif key in (2621440, 65364, 63233, ord("-")):   # Down/'-' → lower threshold...
        threshold = max(threshold - THRESHOLD_STEP, THRESHOLD_STEP)



cap.release()                                        # Release the camera.
cv2.destroyAllWindows()                              # Close OpenCV windows.
hands.close()                                        # Free the MediaPipe Hands resources.

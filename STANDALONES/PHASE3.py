import cv2

from ultralytics import YOLO

import mediapipe as mp


import serial

import time

ser = serial.Serial("COM4",9600)

time.sleep(2)

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



cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

prev_time = time.perf_counter()
last_state = None
last_serial_state = None

cmd = input("G or R: ")

threshold = HAND_AREA_THRESHOLD

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if FLIP_VIEW:
        frame = cv2.flip(frame, 1)

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

    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    hand_results = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True

    safety_state = "GREEN"
    frame_h, frame_w = frame.shape[:2]
    hand_too_close = False

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
            if area > threshold:
                box_color = COLOR_RED
                hand_too_close = True
            else:
                box_color = COLOR_GREEN

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            draw_label(frame, f"Hand area: {area}", x1, y1, box_color)

            if area > threshold:
                safety_state = "RED"

    collision_risk = safety_state == "RED" and objects_detected

    if safety_state != last_state:

         if safety_state == "RED":
             print("ROBOT STOP - HUMAN TOO CLOSE")
             ser.write(b"R")

         else:
             print("ROBOT SAFE - OPERATION ALLOWED")
             ser.write(b"G")

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

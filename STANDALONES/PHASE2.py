"""
Cobot Safety Monitor
--------------------
Real-time hand-proximity safety system using MediaPipe Hands + OpenCV.
Estimates distance via apparent hand-box area and switches between
GREEN (safe) and RED (stop) zones accordingly.

Usage:
    python cobot_safety_monitor.py

Controls:
    q          - quit
    Up/Down    - raise / lower area threshold
    +/-        - same as Up/Down (universal fallback)
"""

import cv2
import mediapipe as mp
import time


CAMERA_INDEX      = 0
WINDOW_NAME       = "Cobot Safety Monitor"

DEFAULT_THRESHOLD = 40_000
THRESHOLD_STEP    = 2_000
THRESHOLD_MAX     = 150_000

MAX_HANDS         = 2
DETECTION_CONF    = 0.7
TRACKING_CONF     = 0.6

CLR_GREEN   = (0,   210,  80)
CLR_RED     = (0,    40, 230)
CLR_WHITE   = (255, 255, 255)
CLR_BLACK   = (0,     0,   0)
CLR_YELLOW  = (0,   200, 255)
CLR_GRAY    = (160, 160, 160)

FONT = cv2.FONT_HERSHEY_SIMPLEX


mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

LANDMARK_SPEC   = mp_drawing.DrawingSpec(color=CLR_WHITE,  thickness=2, circle_radius=3)
CONNECTION_SPEC = mp_drawing.DrawingSpec(color=CLR_YELLOW, thickness=2)



def compute_bounding_box(hand_landmarks, frame_w: int, frame_h: int) -> dict:
    """Return pixel bounding-box metrics for one hand."""
    xs = [lm.x * frame_w for lm in hand_landmarks.landmark]
    ys = [lm.y * frame_h for lm in hand_landmarks.landmark]

    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    width        = x_max - x_min
    height       = y_max - y_min

    return dict(x_min=x_min, y_min=y_min,
                x_max=x_max, y_max=y_max,
                width=width, height=height, area=width * height)


def draw_bounding_box(frame, bbox: dict, colour: tuple, hand_idx: int) -> None:
    """Draw bounding rectangle, glow, and area label for one hand."""
    padding = 12
    x1 = max(bbox["x_min"] - padding, 0)
    y1 = max(bbox["y_min"] - padding, 0)
    x2 = min(bbox["x_max"] + padding, frame.shape[1])
    y2 = min(bbox["y_max"] + padding, frame.shape[0])

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), colour, 3)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    label        = f"Hand {hand_idx + 1}"
    (tw, th), _  = cv2.getTextSize(label, FONT, 0.55, 1)
    lx, ly       = x1, max(y1 - 10, th + 4)
    cv2.rectangle(frame, (lx - 4, ly - th - 4), (lx + tw + 4, ly + 4), colour, -1)
    cv2.putText(frame, label, (lx, ly), FONT, 0.55, CLR_BLACK, 1, cv2.LINE_AA)


def draw_safety_banner(frame, is_red: bool) -> None:
    """Full-width translucent banner at the top."""
    h, w      = frame.shape[:2]
    bar_h     = 40
    colour    = CLR_RED if is_red else CLR_GREEN
    state_txt = "STOP -- TOO CLOSE" if is_red else "SAFE -- GREEN ZONE"

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), colour, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    (tw, th), _ = cv2.getTextSize(state_txt, FONT, 1.0, 2)
    tx = (w - tw) // 2
    ty = (bar_h + th) // 2
    cv2.putText(frame, state_txt, (tx, ty), FONT, 1.0, CLR_WHITE, 2, cv2.LINE_AA)


def draw_hud(frame, fps: float, threshold: int, hand_count: int) -> None:
    """Bottom HUD: FPS, threshold, hand count."""
    h, w  = frame.shape[:2]
    bar_h = 44
    y0    = h - bar_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), CLR_BLACK, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    y_text = h - 14
    cv2.putText(frame, f"FPS: {fps:.0f}",
                (12, y_text), FONT, 0.65, CLR_WHITE, 1, cv2.LINE_AA)

    cv2.putText(frame, f"Threshold: {threshold:,}",
                (120, y_text), FONT, 0.52, CLR_GRAY, 1, cv2.LINE_AA)

    hand_txt    = f"Hands: {hand_count}"
    (tw, _), _  = cv2.getTextSize(hand_txt, FONT, 0.60, 1)
    cv2.putText(frame, hand_txt,
                (w - tw - 12, y_text), FONT, 0.60, CLR_YELLOW, 1, cv2.LINE_AA)


def draw_threshold_indicator(frame, threshold: int) -> None:
    """Vertical gauge on the right edge."""
    h, w     = frame.shape[:2]
    bar_x    = w - 22
    bar_top  = 70
    bar_bot  = h - 55
    bar_len  = bar_bot - bar_top
    max_area = THRESHOLD_MAX

    cv2.rectangle(frame, (bar_x - 6, bar_top), (bar_x + 6, bar_bot), CLR_GRAY,   -1)

    ratio  = min(threshold / max_area, 1.0)
    fill_y = int(bar_bot - ratio * bar_len)
    cv2.rectangle(frame, (bar_x - 5, fill_y), (bar_x + 5, bar_bot), CLR_YELLOW, -1)

    cv2.putText(frame, "THR", (bar_x - 14, bar_top - 8),
                FONT, 0.38, CLR_GRAY, 1, cv2.LINE_AA)



def main() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    threshold = DEFAULT_THRESHOLD
    prev_time = time.perf_counter()

    with mp_hands.Hands(
        model_complexity         = 0,
        max_num_hands            = MAX_HANDS,
        min_detection_confidence = DETECTION_CONF,
        min_tracking_confidence  = TRACKING_CONF,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame - exiting.")
                break

            frame    = cv2.flip(frame, 1)
            h, w     = frame.shape[:2]

            frame.flags.writeable = False
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            frame.flags.writeable = True

            is_red     = False
            hand_count = 0

            if result.multi_hand_landmarks:
                hand_count = len(result.multi_hand_landmarks)

                for idx, hand_lm in enumerate(result.multi_hand_landmarks):

                    mp_drawing.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        LANDMARK_SPEC, CONNECTION_SPEC,
                    )

                    bbox   = compute_bounding_box(hand_lm, w, h)
                    is_hot = bbox["area"] > threshold
                    if is_hot:
                        is_red = True

                    draw_bounding_box(frame, bbox, CLR_RED if is_hot else CLR_GREEN, idx)

            draw_safety_banner(frame, is_red)

            now       = time.perf_counter()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            draw_hud(frame, fps, threshold, hand_count)
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
    print("Safety monitor stopped.")


if __name__ == "__main__":
    main()
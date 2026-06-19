from ultralytics import YOLO
import cv2
import time
import torch

CONF_THRESHOLD  = 0.70
MODEL_PATH      = "yolov8s.pt"
CAMERA_INDEX    = 0
WINDOW_NAME     = "YOLO Detection"

BOX_COLOR   = (0, 255, 0)
DOT_COLOR   = (0, 0, 255)
FPS_COLOR   = (255, 0, 0)
FONT        = cv2.FONT_HERSHEY_SIMPLEX

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

model = YOLO(MODEL_PATH).to(device)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

def draw_detection(frame, box, label: str, conf: float) -> None:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
    cv2.circle(frame, (cx, cy), 5, DOT_COLOR, -1)
    cv2.putText(frame, f"{label} {conf*100:.1f}%",
                (x1, y1 - 10), FONT, 0.7, BOX_COLOR, 2)
    cv2.putText(frame, f"({cx},{cy})",
                (cx + 10, cy), FONT, 0.6, DOT_COLOR, 2)


def draw_fps(frame, fps: float) -> None:
    cv2.putText(frame, f"FPS: {int(fps)}",
                (20, 40), FONT, 1, FPS_COLOR, 2)


prev_time = time.perf_counter()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame — exiting.")
            break


        results = model(frame, conf=CONF_THRESHOLD, verbose=False)

        for box in results[0].boxes:
            cls   = int(box.cls[0])
            label = model.names[cls]
            conf  = float(box.conf[0])
            draw_detection(frame, box, label, conf)

        now       = time.perf_counter()
        fps       = 1.0 / (now - prev_time)
        prev_time = now
        draw_fps(frame, fps)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")
"""
PHASE1.py
=========
First development milestone of the project: a minimal real-time object detector.

It opens the webcam, runs the YOLOv8 model on every frame, draws a box + centre
dot + label for each detected object, shows the live FPS, and quits on 'q'.
This is the foundation the later phases (safety zones, faces, GUI) build on.
"""

from ultralytics import YOLO                    # YOLO: the object-detection model class.
import cv2                                       # OpenCV: webcam capture, drawing, and window display.
import time                                      # time: used to measure how long each frame takes (for FPS).
import torch                                     # PyTorch: lets us check for a CUDA-capable GPU.

# ── Config ────────────────────────────────────────────────────────────────────
CONF_THRESHOLD  = 0.70                           # Ignore detections below 70% confidence (reduces false positives).
MODEL_PATH      = "yolov8s.pt"                   # Path to the pre-trained YOLOv8-small weights file.
CAMERA_INDEX    = 0                              # Which camera to open (0 = default/built-in webcam).
WINDOW_NAME     = "YOLO Detection"               # Title shown on the OpenCV display window.

# Colours (BGR)  ── OpenCV uses Blue-Green-Red order, NOT RGB.
BOX_COLOR   = (0, 255, 0)                         # Green — bounding box + label text.
DOT_COLOR   = (0, 0, 255)                         # Red — centre dot + coordinate text.
FPS_COLOR   = (255, 0, 0)                         # Blue — FPS counter.
FONT        = cv2.FONT_HERSHEY_SIMPLEX            # Built-in OpenCV font used for all on-screen text.

# ── Setup ─────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use the GPU if one is available, else fall back to CPU.
print(f"Running on: {device}")                   # Tell the user which device will run inference.

model = YOLO(MODEL_PATH).to(device)              # Load the YOLO weights and move the model onto the chosen device.

cap = cv2.VideoCapture(CAMERA_INDEX)             # Open a video stream from the selected camera.
if not cap.isOpened():                           # If the camera could not be opened...
    raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")  # ...abort with a clear error.

# ── Helpers ───────────────────────────────────────────────────────────────────
def draw_detection(frame, box, label: str, conf: float) -> None:  # Draw one detection onto the frame.
    x1, y1, x2, y2 = map(int, box.xyxy[0])       # Unpack the box corners (top-left & bottom-right) as ints.
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2       # Compute the centre point of the box (integer division).

    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)        # Draw the bounding rectangle (thickness 2).
    cv2.circle(frame, (cx, cy), 5, DOT_COLOR, -1)                # Draw a filled (-1) dot at the centre.
    cv2.putText(frame, f"{label} {conf*100:.1f}%",              # Write "<label> <confidence%>"...
                (x1, y1 - 10), FONT, 0.7, BOX_COLOR, 2)          # ...just above the box's top-left corner.
    cv2.putText(frame, f"({cx},{cy})",                          # Write the centre pixel coordinates...
                (cx + 10, cy), FONT, 0.6, DOT_COLOR, 2)          # ...next to the centre dot.


def draw_fps(frame, fps: float) -> None:          # Draw the frames-per-second counter.
    cv2.putText(frame, f"FPS: {int(fps)}",        # Render "FPS: <whole number>"...
                (20, 40), FONT, 1, FPS_COLOR, 2)  # ...near the top-left corner of the frame.


prev_time = time.perf_counter()                   # Timestamp of the previous frame (high-resolution clock).

try:                                              # Wrap the main loop so the camera is always released on exit.
    while True:                                   # Process frames forever (until 'q' or a read failure).
        ret, frame = cap.read()                   # Grab one frame; ret=False if the grab failed.
        if not ret:                               # If no frame was returned...
            print("Failed to grab frame — exiting.")  # ...report it and...
            break                                 # ...leave the loop.


        results = model(frame, conf=CONF_THRESHOLD, verbose=False)  # Run detection (silent, with our threshold).

        for box in results[0].boxes:              # Iterate over every detected box in this frame.
            cls   = int(box.cls[0])               # Numeric class id of the detection.
            label = model.names[cls]              # Convert the class id to a human-readable name (e.g. "person").
            conf  = float(box.conf[0])            # Detection confidence as a float in 0..1.
            draw_detection(frame, box, label, conf)  # Draw this detection onto the frame.

        # FPS  ── frames per second = 1 / (time taken for this frame).
        now       = time.perf_counter()           # Current timestamp.
        fps       = 1.0 / (now - prev_time)       # Instantaneous FPS from the gap since the last frame.
        prev_time = now                           # Remember this timestamp for the next iteration.
        draw_fps(frame, fps)                      # Overlay the FPS value.

        cv2.imshow(WINDOW_NAME, frame)            # Display the annotated frame in the window.
        if cv2.waitKey(1) & 0xFF == ord("q"):     # Wait 1 ms for a key; if it's 'q'...
            break                                 # ...exit the loop.

finally:                                          # Always runs, even on error/break:
    cap.release()                                 # Release the camera so other apps can use it.
    cv2.destroyAllWindows()                       # Close any OpenCV windows we opened.
    print("Done.")                                # Final confirmation message.

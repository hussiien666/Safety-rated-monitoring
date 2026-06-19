"""
Cobot Safety Monitor  (PHASE2.py)
--------------------
Real-time hand-proximity safety system using MediaPipe Hands + OpenCV.
Estimates distance via apparent hand-box area and switches between
GREEN (safe) and RED (stop) zones accordingly.

Idea: a hand that is closer to the camera appears LARGER, so its bounding-box
area grows. When that area crosses a tunable threshold we treat the hand as
"too close" and raise a STOP state.

Usage:
    python cobot_safety_monitor.py

Controls:
    q          - quit
    Up/Down    - raise / lower area threshold
    +/-        - same as Up/Down (universal fallback)
"""

import cv2                                        # OpenCV: camera capture, drawing, window/key handling.
import mediapipe as mp                            # MediaPipe: Google's ML pipeline; here for hand landmarks.
import time                                       # time: high-resolution timing for the FPS counter.

# -- Configuration --------------------------------------------------------------

CAMERA_INDEX      = 0                             # Default webcam.
WINDOW_NAME       = "Cobot Safety Monitor"        # Title of the display window.

DEFAULT_THRESHOLD = 40_000                        # Starting area threshold in px² (underscore = digit separator).
THRESHOLD_STEP    = 2_000                         # How much each key press changes the threshold.
THRESHOLD_MAX     = 150_000     # upper clamp     # Threshold can never be raised above this.

MAX_HANDS         = 2                             # Track at most two hands at once.
DETECTION_CONF    = 0.7                           # Min confidence to first DETECT a hand.
TRACKING_CONF     = 0.6                           # Min confidence to keep TRACKING a detected hand.

# BGR colours  ── remember OpenCV is Blue-Green-Red, not RGB.
CLR_GREEN   = (0,   210,  80)                      # "Safe" colour.
CLR_RED     = (0,    40, 230)                      # "Stop / too close" colour.
CLR_WHITE   = (255, 255, 255)                      # Text / landmarks.
CLR_BLACK   = (0,     0,   0)                      # Label text / dark backgrounds.
CLR_YELLOW  = (0,   200, 255)                      # Accents (hand connections, gauges).
CLR_GRAY    = (160, 160, 160)                      # Muted text / inactive gauge.

FONT = cv2.FONT_HERSHEY_SIMPLEX                    # Single font reused for all overlays.

# -- MediaPipe setup ------------------------------------------------------------

mp_hands   = mp.solutions.hands                    # Sub-module exposing the Hands solution + connection map.
mp_drawing = mp.solutions.drawing_utils            # Helper utilities for drawing landmarks/connections.

# DrawingSpec bundles colour + thickness + dot radius for the landmark renderer.
LANDMARK_SPEC   = mp_drawing.DrawingSpec(color=CLR_WHITE,  thickness=2, circle_radius=3)  # Joints (dots).
CONNECTION_SPEC = mp_drawing.DrawingSpec(color=CLR_YELLOW, thickness=2)                   # Bones (lines).


# -- Helper functions -----------------------------------------------------------

def compute_bounding_box(hand_landmarks, frame_w: int, frame_h: int) -> dict:
    """Return pixel bounding-box metrics for one hand."""
    # MediaPipe gives landmark coords normalised to 0..1; multiply by frame size → pixels.
    xs = [lm.x * frame_w for lm in hand_landmarks.landmark]   # All landmark X positions in pixels.
    ys = [lm.y * frame_h for lm in hand_landmarks.landmark]   # All landmark Y positions in pixels.

    x_min, x_max = int(min(xs)), int(max(xs))     # Left & right extents of the hand.
    y_min, y_max = int(min(ys)), int(max(ys))     # Top & bottom extents of the hand.
    width        = x_max - x_min                  # Box width in pixels.
    height       = y_max - y_min                  # Box height in pixels.

    return dict(x_min=x_min, y_min=y_min,         # Pack everything the callers need into a dict...
                x_max=x_max, y_max=y_max,
                width=width, height=height, area=width * height)  # area = the proximity proxy.


def draw_bounding_box(frame, bbox: dict, colour: tuple, hand_idx: int) -> None:
    """Draw bounding rectangle, glow, and area label for one hand."""
    padding = 12                                  # Expand the box slightly so it doesn't clip the fingertips.
    x1 = max(bbox["x_min"] - padding, 0)          # Clamp left edge to >= 0 (stay inside the frame).
    y1 = max(bbox["y_min"] - padding, 0)          # Clamp top edge to >= 0.
    x2 = min(bbox["x_max"] + padding, frame.shape[1])  # Clamp right edge to <= frame width.
    y2 = min(bbox["y_max"] + padding, frame.shape[0])  # Clamp bottom edge to <= frame height.

    overlay = frame.copy()                        # Work on a copy so we can blend a translucent "glow".
    cv2.rectangle(overlay, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), colour, 3)  # Thick outer rectangle on the copy.
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)  # Blend copy(40%)+original(60%) → soft glow effect.

    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)  # Crisp inner rectangle drawn directly on the frame.

    label        = f"Hand {hand_idx + 1}"         # Human-friendly 1-based hand label.
    (tw, th), _  = cv2.getTextSize(label, FONT, 0.55, 1)  # Measure label so we can size its background chip.
    lx, ly       = x1, max(y1 - 10, th + 4)       # Place label above the box, but keep it on-screen.
    cv2.rectangle(frame, (lx - 4, ly - th - 4), (lx + tw + 4, ly + 4), colour, -1)  # Filled chip behind text.
    cv2.putText(frame, label, (lx, ly), FONT, 0.55, CLR_BLACK, 1, cv2.LINE_AA)      # Draw the label text.


def draw_safety_banner(frame, is_red: bool) -> None:
    """Full-width translucent banner at the top."""
    h, w      = frame.shape[:2]                   # Frame height & width.
    bar_h     = 40                                # Banner height in pixels.
    colour    = CLR_RED if is_red else CLR_GREEN  # Red when unsafe, green when safe.
    state_txt = "STOP -- TOO CLOSE" if is_red else "SAFE -- GREEN ZONE"  # Matching message.

    overlay = frame.copy()                        # Copy for translucent fill.
    cv2.rectangle(overlay, (0, 0), (w, bar_h), colour, -1)  # Solid coloured bar across the top of the copy.
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)   # Blend → semi-transparent banner.

    (tw, th), _ = cv2.getTextSize(state_txt, FONT, 1.0, 2)  # Measure the status text...
    tx = (w - tw) // 2                            # ...to centre it horizontally...
    ty = (bar_h + th) // 2                        # ...and vertically within the banner.
    cv2.putText(frame, state_txt, (tx, ty), FONT, 1.0, CLR_WHITE, 2, cv2.LINE_AA)  # Draw status text.


def draw_hud(frame, fps: float, threshold: int, hand_count: int) -> None:
    """Bottom HUD: FPS, threshold, hand count."""
    h, w  = frame.shape[:2]                       # Frame dimensions.
    bar_h = 44                                    # HUD strip height.
    y0    = h - bar_h                             # Y where the HUD strip starts.

    overlay = frame.copy()                        # Copy for translucent dark strip.
    cv2.rectangle(overlay, (0, y0), (w, h), CLR_BLACK, -1)  # Black bar along the bottom of the copy.
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)   # Blend → dim translucent HUD background.

    y_text = h - 14                               # Baseline Y for all HUD text.
    cv2.putText(frame, f"FPS: {fps:.0f}",         # Current frame rate (no decimals)...
                (12, y_text), FONT, 0.65, CLR_WHITE, 1, cv2.LINE_AA)  # ...on the left.

    cv2.putText(frame, f"Threshold: {threshold:,}",  # Current area threshold ( , = thousands separator)...
                (120, y_text), FONT, 0.52, CLR_GRAY, 1, cv2.LINE_AA)  # ...in the middle.

    hand_txt    = f"Hands: {hand_count}"          # How many hands are currently tracked.
    (tw, _), _  = cv2.getTextSize(hand_txt, FONT, 0.60, 1)  # Measure it so we can right-align it.
    cv2.putText(frame, hand_txt,
                (w - tw - 12, y_text), FONT, 0.60, CLR_YELLOW, 1, cv2.LINE_AA)  # Draw on the right.


def draw_threshold_indicator(frame, threshold: int) -> None:
    """Vertical gauge on the right edge."""
    h, w     = frame.shape[:2]                    # Frame dimensions.
    bar_x    = w - 22                             # X position of the gauge (near the right edge).
    bar_top  = 70                                 # Top Y of the gauge track.
    bar_bot  = h - 55                             # Bottom Y of the gauge track.
    bar_len  = bar_bot - bar_top                  # Total track length in pixels.
    max_area = THRESHOLD_MAX                       # The value that corresponds to a full gauge.

    cv2.rectangle(frame, (bar_x - 6, bar_top), (bar_x + 6, bar_bot), CLR_GRAY,   -1)  # Gauge background track.

    ratio  = min(threshold / max_area, 1.0)       # Fraction of the gauge to fill (capped at 100%).
    fill_y = int(bar_bot - ratio * bar_len)       # Y where the yellow fill should start (grows upward).
    cv2.rectangle(frame, (bar_x - 5, fill_y), (bar_x + 5, bar_bot), CLR_YELLOW, -1)   # The yellow fill.

    cv2.putText(frame, "THR", (bar_x - 14, bar_top - 8),  # "THR" label above the gauge.
                FONT, 0.38, CLR_GRAY, 1, cv2.LINE_AA)


# -- Main -----------------------------------------------------------------------

def main() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)          # Open the webcam.
    if not cap.isOpened():                        # Abort early if it failed to open.
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)        # Request a 640x480 capture (smaller = faster).
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    threshold = DEFAULT_THRESHOLD                  # Live, user-adjustable threshold (starts at default).
    prev_time = time.perf_counter()                # Seed the FPS timer.

    with mp_hands.Hands(                           # Create the hand tracker; 'with' auto-closes it on exit.
        model_complexity         = 0,              # 0 = lightest/fastest model variant.
        max_num_hands            = MAX_HANDS,      # Cap on simultaneously tracked hands.
        min_detection_confidence = DETECTION_CONF, # Detection gate.
        min_tracking_confidence  = TRACKING_CONF,  # Tracking gate.
    ) as hands:

        while True:                                # Main per-frame loop.
            ret, frame = cap.read()                # Grab a frame (ret=False on failure).
            if not ret:
                print("Failed to grab frame - exiting.")
                break

            frame    = cv2.flip(frame, 1)          # Mirror horizontally → natural "selfie" view.
            h, w     = frame.shape[:2]             # Current frame size.

            frame.flags.writeable = False          # Mark read-only → MediaPipe can process without a copy (faster).
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe expects RGB, OpenCV gives BGR.
            result = hands.process(rgb)            # Run hand detection/tracking on the RGB frame.
            frame.flags.writeable = True           # Make writable again so we can draw on it.

            is_red     = False                     # Assume safe until a hand proves otherwise.
            hand_count = 0                         # Number of hands found this frame.

            if result.multi_hand_landmarks:        # If at least one hand was detected...
                hand_count = len(result.multi_hand_landmarks)  # Count them.

                for idx, hand_lm in enumerate(result.multi_hand_landmarks):  # For each detected hand...

                    mp_drawing.draw_landmarks(     # Draw the skeleton (dots + connecting lines).
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        LANDMARK_SPEC, CONNECTION_SPEC,
                    )

                    bbox   = compute_bounding_box(hand_lm, w, h)  # Measure this hand's box + area.
                    is_hot = bbox["area"] > threshold             # "Hot" = too close (area over threshold).
                    if is_hot:                                    # If any hand is too close...
                        is_red = True                             # ...the whole frame goes into STOP state.

                    draw_bounding_box(frame, bbox, CLR_RED if is_hot else CLR_GREEN, idx)  # Box per hand.

            draw_safety_banner(frame, is_red)      # Top banner reflects the overall safe/stop state.

            now       = time.perf_counter()        # Timestamp now...
            fps       = 1.0 / max(now - prev_time, 1e-6)  # ...compute FPS (guard against divide-by-zero).
            prev_time = now                        # Save for next frame.

            draw_hud(frame, fps, threshold, hand_count)   # Bottom HUD.
            draw_threshold_indicator(frame, threshold)    # Right-edge threshold gauge.

            cv2.imshow(WINDOW_NAME, frame)         # Show the finished frame.

            # waitKeyEx returns the full keycode - arrow keys work on all OSes.
            # waitKey(1) & 0xFF only keeps the low byte, breaking arrow keys.
            #
            # Keycodes:  Windows   Linux    macOS
            #   Up       2490368   65362    63232
            #   Down     2621440   65364    63233
            # +/- keys work as a fallback on every platform.
            key = cv2.waitKeyEx(1)                 # Read a key (full code), waiting up to 1 ms.

            if key == ord("q"):                    # 'q' → quit.
                break
            elif key in (2490368, 65362, 63232, ord("+")):  # Up/'+' → raise threshold (less sensitive)...
                threshold = min(threshold + THRESHOLD_STEP, THRESHOLD_MAX)  # ...clamped to the max.
            elif key in (2621440, 65364, 63233, ord("-")):  # Down/'-' → lower threshold (more sensitive)...
                threshold = max(threshold - THRESHOLD_STEP, THRESHOLD_STEP)  # ...clamped to one step minimum.

    cap.release()                                  # Release the camera.
    cv2.destroyAllWindows()                        # Close OpenCV windows.
    print("Safety monitor stopped.")               # Goodbye message.


if __name__ == "__main__":                         # Only run main() when executed directly (not when imported).
    main()

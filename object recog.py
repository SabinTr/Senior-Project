# pip install ultralytics opencv-python

import os
import time
import cv2

# --- Pi resource management -------------------------------------------------
# Limit CPU threads used by the underlying inference engine so this doesn't
# starve other processes (voice recognition, etc.) running on the same Pi.
os.environ.setdefault("OMP_NUM_THREADS", "3")
try:
    import torch
    torch.set_num_threads(3)  # Pi 5 has 4 cores; leave 1 free for the OS/other scripts
except ImportError:
    pass
# -----------------------------------------------------------------------------

from ultralytics import YOLO

# --- Settings ----------------------------------------------------------------
CAMERA_INDEX = 0          # change if you have more than one camera device
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
FRAME_SKIP = 1            # run YOLO every (FRAME_SKIP + 1)th frame; 1 = every other frame
CONFIDENCE_THRESHOLD = 0.4
SHOW_FPS = True
# -----------------------------------------------------------------------------

model = YOLO("yolov8n.pt")  # downloads automatically the first time

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise IOError(f"Cannot open camera at index {CAMERA_INDEX}")

# Ask the camera to capture at a lower resolution directly, rather than
# capturing full-size and downscaling in software - saves CPU either way.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

frame_count = 0
last_annotated_frame = None
fps_timer = time.time()
fps_counter = 0
current_fps = 0.0

print("Running. Press 'q' in the video window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera, stopping.")
        break

    frame_count += 1

    # Only run the (expensive) YOLO inference every Nth frame. On the
    # skipped frames, keep showing the last annotated result so the video
    # doesn't look choppy, it just doesn't update the boxes every frame.
    if frame_count % (FRAME_SKIP + 1) == 0 or last_annotated_frame is None:
        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
        last_annotated_frame = results[0].plot()

    display_frame = last_annotated_frame

    if SHOW_FPS:
        fps_counter += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_timer = time.time()
        cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Object Detection - press 'q' to quit", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
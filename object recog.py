# pip install ultralytics opencv-python requests

import os
import time
import cv2
import requests

# --- Pi resource management -------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "3")
try:
    import torch
    torch.set_num_threads(3)
except ImportError:
    pass
# -----------------------------------------------------------------------------

from ultralytics import YOLO

# --- Settings you can change --------------------------------------------------
TARGET_OBJECT = "bottle"   # <-- set this to whatever you want it to search for
                           # must match a YOLO class name, e.g.:
                           # person, cat, dog, bottle, chair, cup, laptop, etc.

TELEGRAM_TOKEN = "YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"

CAMERA_INDEX = 0
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
FRAME_SKIP = 1
CONFIDENCE_THRESHOLD = 0.6  # raised from 0.5 - fewer shaky, low-confidence guesses
SEARCH_TIMEOUT = 60         # give up after this many seconds if never found
SHOW_VIDEO_WINDOW = True    # set False if you don't need to watch it live

# How many detection checks IN A ROW must see the object before it's
# trusted as real, instead of acting on a single flickery frame.
CONSECUTIVE_FRAMES_REQUIRED = 4
# -----------------------------------------------------------------------------


def send_photo_telegram(path, caption=None):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    with open(path, "rb") as photo:
        files = {"photo": photo}
        data = {"chat_id": TELEGRAM_CHAT_ID}
        if caption:
            data["caption"] = caption
        response = requests.post(url, files=files, data=data, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Telegram API error: {response.text}")


def main():
    if TELEGRAM_TOKEN == "YOUR_BOT_TOKEN_HERE" or TELEGRAM_CHAT_ID == "YOUR_CHAT_ID_HERE":
        raise SystemExit("Set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID at the top of this file first.")

    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise IOError(f"Cannot open camera at index {CAMERA_INDEX}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    print(f"Searching for '{TARGET_OBJECT}'... (timeout: {SEARCH_TIMEOUT}s)")
    start_time = time.time()
    frame_count = 0
    found = False
    consecutive_hits = 0

    try:
        while time.time() - start_time < SEARCH_TIMEOUT:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera, stopping.")
                break

            frame_count += 1
            if frame_count % (FRAME_SKIP + 1) != 0:
                continue

            results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
            annotated = results[0].plot()

            detected_names = {model.names[int(box.cls[0])] for box in results[0].boxes}

            if SHOW_VIDEO_WINDOW:
                cv2.imshow("Searching - press 'q' to cancel", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Cancelled.")
                    break

            if TARGET_OBJECT in detected_names:
                consecutive_hits += 1
                print(f"  ...saw '{TARGET_OBJECT}' ({consecutive_hits}/{CONSECUTIVE_FRAMES_REQUIRED} confirmations)")
            else:
                consecutive_hits = 0  # any miss resets the streak - must be seen repeatedly in a row

            if consecutive_hits >= CONSECUTIVE_FRAMES_REQUIRED:
                photo_path = "found_object.jpg"
                cv2.imwrite(photo_path, annotated)
                print(f"Confirmed '{TARGET_OBJECT}'! Sending photo to Telegram...")
                send_photo_telegram(photo_path, caption=f"Found: {TARGET_OBJECT}")
                os.remove(photo_path)
                print("Sent. Stopping.")
                found = True
                break

        if not found and time.time() - start_time >= SEARCH_TIMEOUT:
            print(f"Gave up after {SEARCH_TIMEOUT}s, '{TARGET_OBJECT}' was not seen.")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
# pip install ultralytics opencv-python

from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # downloads automatically the first time

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    cv2.imshow("Object Detection - press 'q' to quit", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

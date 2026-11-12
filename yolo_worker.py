"""YOLO in its OWN program, started by roam_and_orbit.py (don't run it yourself).

Why: with YOLO inside roam_and_orbit.py the lanes came out crooked, and with YOLO switched
off they were perfect (user, 2026-11-13). Inside the same program YOLO slowed the steering:
importing ultralytics makes OpenCV single-threaded (cv2.setNumThreads(0)) for the tile
steering too, and YOLO's Python work competes with the steering loop. As a separate program
it can't do either, and it runs at low priority so the steering always goes first.

Talks to roam_and_orbit.py through pipes: gets a camera picture, sends back what it sees as
(name, confidence, x1, y1, x2, y2) with the box as fractions of the picture (0..1).
Arguments: model path, image size, confidence, CPU threads."""
import os
import pickle
import signal
import sys

# The pipe back to roam_and_orbit.py is the real stdout. Anything a library prints goes to
# the terminal's error output instead, so it can't mix into the answers.
answers = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)
sys.stdout = sys.stderr
questions = sys.stdin.buffer

# Ctrl+C / SSH dropping reach this program too: roam_and_orbit.py handles them (stops the
# motors). This one just ends when roam_and_orbit.py closes the pipe.
signal.signal(signal.SIGINT, signal.SIG_IGN)
if hasattr(signal, "SIGHUP"):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

model_path, imgsz, conf, threads = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
os.environ["OMP_NUM_THREADS"] = str(threads)
try:
    os.nice(10)                  # lower priority than the driving program
except (AttributeError, OSError):
    pass

import numpy as np
import torch
from ultralytics import YOLO

model = YOLO(model_path)
model(np.zeros((240, 320, 3), np.uint8), imgsz=imgsz, verbose=False)   # slow first check, now
torch.set_num_threads(threads)   # ultralytics' first check set it to (all cores - 1)
pickle.dump("ready", answers)
answers.flush()

while True:
    try:
        frame = pickle.load(questions)
    except EOFError:
        break                    # roam_and_orbit.py has ended
    result = model(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
    boxes = [(model.names[int(c)], float(p), *(float(v) for v in xyxyn))
             for c, p, xyxyn in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxyn)]
    pickle.dump(boxes, answers)
    answers.flush()

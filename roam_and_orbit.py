"""
roam_and_orbit.py  -  movement only (no object recognition yet)

The robot sweeps the room in lanes (like mowing a lawn). It uses the
ultrasonic sensor to notice that SOMETHING is in front of it, without
knowing what it is:

  - Something narrow (an object on the floor) -> drive around it
    ORBIT_LAPS times, always facing it, so the camera can later see it
    from every side. Then go around it and continue the lane.
  - Something wide (a wall / big furniture) -> end of lane: U-turn into
    the next lane and sweep back.
  - When the U-turn finds a wall where the next lane should be, the
    whole room is done.

Narrow vs wide: the robot turns a little left and right. If the sensor
sees past the thing on either side, it's a narrow object; if it is still
blocked on both sides, it's a wall.

The sensor only looks forward, so the robot never moves sideways blind:
it turns to face where it wants to go, checks, then drives. (The orbit
itself is the one exception - keep the area around objects clear.)

Start position: a corner, facing along a wall, with that wall on the
robot's RIGHT (room on its left). U-turns go left, right, left, ...

--------------------------------------------------------------------
BEFORE RUNNING
--------------------------------------------------------------------
1) Stop Hiwonder's background service (it may hold the motors/camera):
   sudo systemctl stop <service name>
2) Calibrate first, with the wheels OFF the ground:
   python3 roam_and_orbit.py --test
   Then on the floor, measure and fill in the CALIBRATION values below.
3) Run:
   python3 roam_and_orbit.py
Ctrl+C stops the motors and quits.

Limits: there is no map, so lanes drift a bit over time, and the same
object can be orbited again if a later lane passes close to it. Objects
lower than the ultrasonic sensor's beam are not detected.
"""

import os
import sys
import math
import time
import signal
import cv2
import numpy as np

# --- Robot-specific part (Hiwonder SDK) -------------------------------------
# MasterPi SDK lives in ~/MasterPi/masterpi_sdk/common_sdk/common/
# (mecanum.py opens the motor board as soon as it's imported)
sys.path.append("/home/pi/MasterPi/masterpi_sdk/common_sdk")
sys.path.append("/home/pi/MasterPi/masterpi_sdk/kinematics_sdk")
from common import mecanum as mecanum_lib
from common.sonar import Sonar
# ---------------------------------------------------------------------------

# --- Speeds -------------------------------------------------------------------
# The SDK says mm/s, but the value goes straight to set_motor_duty: it's motor
# power in % (0-100). Too low and the wheels don't turn at all.
DRIVE_SPEED = 40          # forward / sideways power, keep it low while testing
ROTATE_RATE = 0.4         # turning speed (x126 = turning power in %)
TURN_RIGHT_SIGN = 1       # checked in mecanum.py: positive angular_rate = clockwise
# The board has no gyroscope and the motors no speed sensors, so the robot can't
# measure its heading; this constant turn cancels its natural curve while driving.
# Curves LEFT while driving forward -> make it positive (try 0.03, 0.06, ...).
# Curves RIGHT -> make it negative.
STEER_TRIM = -0.01       # 0 curved right; -0.03 a lot left; -0.02 slightly left

# --- CALIBRATION: measure these on the floor after --test --------------------
FORWARD_CM_PER_SEC = 15   # distance driven forward in 1 s at DRIVE_SPEED
STRAFE_CM_PER_SEC = 12    # distance slid sideways in 1 s at DRIVE_SPEED
# The robot needs a moment to get going before it really turns. From real results
# (2026-10-20, robot-bridge stopped): 360 deg took 2.5 s; a 0.9 s turn gave ~67 deg
# (U-turn ended ~135 deg instead of 180, seen in a photo) -> ~0.53 s start-up, then ~183 deg/s.
# U-turn not enough -> raise TURN_START_DELAY a little; too much -> lower it.
TURN_DEG_PER_SEC = 183    # turning speed once it's moving
TURN_START_DELAY = 0.53   # extra seconds every turn needs to get going

# --- Behaviour ----------------------------------------------------------------
# Sized for the user's small room (2 m x 1.6 m); for big rooms use 50 / 25 / 40 / 40 / 25
DETECT_CM = 40            # something closer than this = "something in front"
ORBIT_CM = 20             # distance to keep from an object while circling it
OBJECT_RADIUS_CM = 5      # rough guess of an object's size (for the circle)
ORBIT_LAPS = 1            # how many times to go around each object
PROBE_DEG = 35            # how far to turn left/right to test narrow vs wide
CLEAR_MARGIN_CM = 30      # "sees past it" = reading this much farther than before
LOST_CM = 80              # while circling, farther than this = object lost
LANE_WIDTH_CM = 30        # distance between lanes
SIDESTEP_CM = 30          # how far to move past an object after circling it
SAFE_CM = 20              # never drive closer than this to anything in front
MAX_LANES = 30            # safety limit; normally the sweep ends at the far side wall
# The ultrasonic sensor can't reliably tell a wall from an object (a flat wall seen at
# an angle reflects the sound away, so a corner looked like "an object" and the robot
# slid right into the wall). So for now everything in front is treated as a wall;
# objects will be found with the camera later. True = old object check + orbit.
DETECT_OBJECTS = False
MAX_RUN_SECONDS = 15 * 60 # safety limit for the whole run

# The sensor only looks forward, so the robot never slides sideways into the
# unknown: to go left/right it first turns to face that way and checks.

# --- Staying straight with the floor tiles (camera) ---------------------------
# Tile lines running ACROSS the robot's path look horizontal in the picture when
# the robot is straight; if they tilt, the robot steers back. If the camera sees
# no lines (other floor), it falls back to STEER_TRIM only.
USE_TILE_LINES = True
CAMERA_INDEX = 0
# The camera AND the ultrasonic sensor sit on the arm. If the arm points down, the
# sensor sees the floor ("wall" with nothing there) and the camera sees no tile joints.
# So every run starts by putting the arm in a fixed pose: Hiwonder's obstacle-avoidance
# pose (0, 6, 18) looks straight ahead. Must stay level: the sensor's beam is a ~30 deg
# cone, and at -10 deg its lower edge hit the floor ~35-40 cm ahead -> a "wall" every
# 1.6 s (the robot stayed inside one tile).
ARM_POSITION = (0, 6, 18)
ARM_PITCH = 0
LINE_SIGN = 1             # check with --lines: turn the robot clockwise -> angle must go UP
LINE_MAX_TILT = 35        # only lines within this many degrees of horizontal count
JOINT_CONTRAST = 20       # how much darker than the floor around it a joint must be
LINE_IGNORE_TOP = 0.1     # ignore this top part of the picture (joints across the path are
                          # often near the top, so don't cut more)
LINE_MIN_TOTAL_PX = 150   # need at least this much line length to trust the angle
STEER_GAIN = 0.005        # steering per degree of tilt while driving (higher = wobbles)
STEER_MAX = 0.1           # max steering correction while driving
CAMERA_TILT_FACTOR = 0.8  # picture tilt ~ 0.6-1 x the real angle (test pictures); used for turns
# The U-turns learn TURN_START_DELAY from the tiles and remember it here between runs
# (delete the file to start over, e.g. on a different floor or with a new battery).
LEARNED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "learned_turn.txt")
HEADING_WINDOW_S = 1.5    # steer by the median of the tile readings from this many seconds
                          # (while driving ~half the pictures have no lines, single ones jump)
# No straightening on the spot: every turn needs ~0.5 s to get going and that varies,
# so a small "5 deg" correction turn can become 30 deg. In simulation, steering while
# driving brought turn errors of up to 25 deg down to ~4 deg within 4 s.


class TileHeading:
    """Measures how crooked the robot is from the tile lines in the camera picture."""

    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError("camera could not be opened")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.last_frame = None
        self.last_lines = []
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        self.dark_kernel = np.ones((15, 15), np.uint8)

    def angle(self):
        """Tilt of the across-the-path tile lines in degrees (0 = straight),
        positive = robot turned clockwise. None if no lines are visible."""
        for _ in range(4):                     # drop old buffered frames (a saved picture
            self.cap.grab()                    # once showed the closet the robot had left)
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.last_frame = frame
        h = frame.shape[0]
        top = int(h * LINE_IGNORE_TOP)
        floor = frame[top:, :]                   # skip the very top (far away / furniture)
        # Tile joints are THIN DARK lines on a bright floor. A black-hat filter keeps only
        # thin dark lines; plain edge detection also picked up the marble pattern everywhere
        # (checked on the robot's own pictures, 2026-10-21).
        gray = cv2.cvtColor(floor, cv2.COLOR_BGR2GRAY)
        dark_lines = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.kernel)
        # Joints are medium grey; things that are nearly black (the Ethernet cable, shadows
        # under furniture) are masked out, otherwise they hide the joints.
        very_dark = cv2.dilate((gray < 90).astype(np.uint8), self.dark_kernel) > 0
        mask = ((dark_lines > JOINT_CONTRAST) & ~very_dark).astype(np.uint8) * 255
        lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=40,
                                minLineLength=80, maxLineGap=10)
        self.last_lines = []
        if lines is None:
            return None
        angles, weights = [], []
        w = frame.shape[1]
        for x1, y1, x2, y2 in lines.reshape(-1, 4):   # OpenCV 4 and 5 use different shapes
            a = math.degrees(math.atan2(y2 - y1, x2 - x1))
            a = (a + 90) % 180 - 90            # -90..90, direction doesn't matter
            # the wide-angle lens bends lines near the edges: only use pieces from the middle
            mid_x = (x1 + x2) / 2
            if abs(a) <= LINE_MAX_TILT and w * 0.15 <= mid_x <= w * 0.85:
                angles.append(a)
                weights.append(math.hypot(x2 - x1, y2 - y1))
                self.last_lines.append((x1, y1 + top, x2, y2 + top))
        if sum(weights) < LINE_MIN_TOTAL_PX:
            return None
        # weighted median: robust against a few stray lines (cable, bottle edges)
        order = np.argsort(angles)
        cum = np.cumsum(np.array(weights)[order])
        median = np.array(angles)[order][np.searchsorted(cum, cum[-1] / 2)]
        # Robot turned clockwise -> lines across the path rise to the right in the
        # picture (negative image angle), so flip it: positive = clockwise.
        return -float(median) * LINE_SIGN

    def release(self):
        self.cap.release()


class Robot:
    """Small wrapper so the rest of the code doesn't depend on the SDK."""

    def __init__(self):
        self.chassis = mecanum_lib.MecanumChassis()
        self.sonar = Sonar()
        try:   # the sensor's purple LEDs sit next to the camera and tint its picture
            self.sonar.setRGBMode(0)
            self.sonar.setPixelColor(0, (0, 0, 0))
            self.sonar.setPixelColor(1, (0, 0, 0))
        except Exception as e:
            print(f"(Couldn't switch off the sensor LEDs: {e})")
        self.sensor_failures = 0
        self.last_cm = 500
        self.set_arm_pose()
        self.tiles = None
        self.heading = None          # smoothed tile angle used for steering
        self.readings = []           # recent (time, angle) tile readings
        self.no_lines_count = 0      # pictures saved when no tile lines were found
        if USE_TILE_LINES:
            try:
                self.tiles = TileHeading()
            except RuntimeError as e:
                print(f"(No camera: {e} - driving without tile lines)")

    def set_arm_pose(self):
        """Point the arm (camera + distance sensor) forward, slightly down."""
        try:
            from kinematics.arm_move_ik import ArmIK
            ak = ArmIK()
            ak.board = mecanum_lib.board        # same motor board the wheels use
            result = ak.setPitchRangeMoving(ARM_POSITION, ARM_PITCH, -90, 90, 1500)
            if not result:
                print(f"(Arm can't reach {ARM_POSITION} at {ARM_PITCH} deg - arm not moved)")
                return
            time.sleep(1.6)                     # wait for the 1.5 s arm move
            print("Arm set: camera and sensor looking forward.")
        except Exception as e:
            print(f"(Couldn't move the arm: {e} - check that it points forward!)")

    def steer(self):
        """Steering correction from the tile lines (0 if none are visible).
        While driving, about half the pictures show no lines (motion blur) and single
        readings jump around, so: average them, skip wild jumps, and keep the last
        good value for a short moment."""
        if self.tiles is None:
            return 0.0
        a = self.tiles.angle()
        now = time.time()
        if a is not None:
            self.readings.append((now, a))
        # median of the readings from the last HEADING_WINDOW_S seconds:
        # one wild value (-15 among -5s) can't take over
        self.readings = [(t, v) for t, v in self.readings if now - t <= HEADING_WINDOW_S]
        if not self.readings:
            self.heading = None
            return 0.0
        self.heading = float(np.median([v for _, v in self.readings]))
        return max(-STEER_MAX, min(STEER_MAX, -STEER_GAIN * self.heading))

    def move(self, right=0.0, forward=0.0, turn_right=0.0):
        """right/forward in speed units, turn_right in ROTATE_RATE units."""
        speed = math.hypot(right, forward)
        direction = math.degrees(math.atan2(forward, right)) % 360  # 90 = forward
        turn_right += STEER_TRIM * forward / DRIVE_SPEED  # cancel the natural curve
        self.chassis.set_velocity(speed, direction, turn_right * TURN_RIGHT_SIGN)

    def stop(self):
        self.chassis.set_velocity(0, 90, 0)

    def distance_cm(self):
        """Median of 3 readings, in cm (sensor gives mm, max 5000, 99999 = read error).
        Stops everything if the sensor keeps failing, instead of driving blind."""
        readings = []
        for _ in range(3):
            mm = self.sonar.getDistance()
            if mm != 99999:
                readings.append(mm / 10)
            time.sleep(0.02)
        if not readings:
            self.sensor_failures += 1
            if self.sensor_failures >= 3:
                raise RuntimeError("Distance sensor is not responding - stopping.")
            return 0  # treat as "something very close" until it answers again
        self.sensor_failures = 0
        cm = sorted(readings)[len(readings) // 2]
        # Closer than ~5 cm the sensor loses the echo and reports 5000 mm ("far").
        # If it was just very close, it's still very close - don't drive into it.
        if cm >= 500 and self.last_cm < 15:
            cm = self.last_cm
        self.last_cm = cm
        return cm


# ----------------------------- Basic moves ----------------------------------
def drive_for(robot, seconds, right=0.0, forward=0.0, turn_right=0.0):
    # Re-send the command every 0.1 s instead of once, so nothing else can
    # cut the move short (robot-bridge's watchdog stopped the motors after 300 ms).
    end = time.time() + seconds
    while time.time() < end:
        robot.move(right, forward, turn_right)
        time.sleep(min(0.1, max(0.0, end - time.time())))
    robot.stop()
    time.sleep(0.2)


def turn(robot, degrees):
    """Positive = turn right, negative = turn left."""
    rate = ROTATE_RATE if degrees > 0 else -ROTATE_RATE
    drive_for(robot, TURN_START_DELAY + abs(degrees) / TURN_DEG_PER_SEC, turn_right=rate)


# ----------------------------- Behaviours -----------------------------------
def drive_forward_cm(robot, cm):
    """Drive forward, watching the sensor. Returns False if something blocked the way."""
    end = time.time() + cm / FORWARD_CM_PER_SEC
    while time.time() < end:
        if robot.distance_cm() < SAFE_CM:
            robot.stop()
            return False
        robot.move(forward=DRIVE_SPEED, turn_right=robot.steer())
    robot.stop()
    time.sleep(0.2)
    return True


def is_narrow_object(robot, distance):
    """Turn a little left and right; if the sensor sees past it on either side, it's narrow.
    (A wall still reads about the same distance when the robot turns.)"""
    turn(robot, -PROBE_DEG)
    left = robot.distance_cm()
    turn(robot, 2 * PROBE_DEG)
    right = robot.distance_cm()
    turn(robot, -PROBE_DEG)  # face the middle again
    print(f"  probe: centre {distance:.0f} cm, left {left:.0f} cm, right {right:.0f} cm")
    return left > distance + CLEAR_MARGIN_CM or right > distance + CLEAR_MARGIN_CM


def move_sideways(robot, cm, to_right):
    """Turn to face the side, check it's clear, drive there, turn back.
    Returns False (and doesn't move) if that side is blocked."""
    angle = 90 if to_right else -90
    turn(robot, angle)
    if robot.distance_cm() < cm + SAFE_CM:
        turn(robot, -angle)
        return False
    ok = drive_forward_cm(robot, cm)
    turn(robot, -angle)
    return ok


def next_lane(robot, to_right):
    """U-turn into the next lane: turn 90, drive one lane width, turn 90 again.
    Returns False if a wall is there, i.e. the whole room is done."""
    angle = 90 if to_right else -90
    # Each 90 deg turn is corrected by what the tiles show before it ("tilt" = how far
    # the robot is from square with the tiles, positive = clockwise). The correction
    # is folded into the big turn, so it never needs an unreliable tiny turn.
    tilt = measure_tilt(robot) or 0.0
    turn(robot, angle - tilt / CAMERA_TILT_FACTOR)
    learn_from_turn(robot, angle)
    if robot.distance_cm() < LANE_WIDTH_CM + SAFE_CM:
        print("  side wall reached: room finished")
        return False
    drive_forward_cm(robot, LANE_WIDTH_CM)
    tilt = measure_tilt(robot) or 0.0
    turn(robot, angle - tilt / CAMERA_TILT_FACTOR)
    learn_from_turn(robot, angle)
    return True


def measure_tilt(robot):
    """Stand still and read the tile angle from a few pictures (median).
    None if the camera can't see tile lines."""
    if robot.tiles is None:
        return None
    time.sleep(0.3)                       # let the robot settle (no motion blur)
    vals = [v for v in (robot.tiles.angle() for _ in range(5)) if v is not None]
    if len(vals) >= 2:
        return float(np.median(vals))
    # Save what the camera saw, to find out why there were no lines
    if robot.tiles.last_frame is not None:
        robot.no_lines_count += 1
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"no_lines_{robot.no_lines_count}.jpg")
        cv2.imwrite(path, robot.tiles.last_frame)
        print(f"    (saved the camera picture as {os.path.basename(path)})")
    return None


def learn_from_turn(robot, angle):
    """After a turn the tiles show how far off it was: adjust TURN_START_DELAY so the
    next turns come out closer. Positive tilt after a LEFT turn = it turned too little."""
    global TURN_START_DELAY
    if robot.tiles is None:
        return
    side = "right" if angle > 0 else "left"
    tilt = measure_tilt(robot)
    if tilt is None:
        print(f"  turn {side}: camera can't see the tiles - not checked")
        return
    error_deg = tilt / CAMERA_TILT_FACTOR * (1 if angle > 0 else -1)   # + = turned too far
    if abs(error_deg) < 3:
        print(f"  turn {side}: good (off by {error_deg:+.0f} deg)")
        return
    print(f"  turn {side}: off by {error_deg:+.0f} deg "
          f"({'too far' if error_deg > 0 else 'not enough'})")
    if abs(error_deg) > 35:
        print("    (too far off to trust the camera - not learning from this one)")
        return
    # half a step at a time, so one bad reading can't wreck it
    TURN_START_DELAY -= 0.5 * error_deg / TURN_DEG_PER_SEC
    TURN_START_DELAY = max(0.2, min(0.9, TURN_START_DELAY))
    print(f"    -> TURN_START_DELAY now {TURN_START_DELAY:.2f} s")
    try:
        with open(LEARNED_FILE, "w") as f:
            f.write(f"{TURN_START_DELAY:.3f}\n")
    except OSError:
        pass


def load_learned_turns():
    """Start from what the previous run learned, if there is one."""
    global TURN_START_DELAY
    try:
        with open(LEARNED_FILE) as f:
            TURN_START_DELAY = max(0.2, min(0.9, float(f.read().strip())))
        print(f"Using learned TURN_START_DELAY = {TURN_START_DELAY:.2f} s")
    except (OSError, ValueError):
        pass


def orbit(robot):
    """Circle the object in front, always facing it, ORBIT_LAPS times."""
    radius = ORBIT_CM + OBJECT_RADIUS_CM
    deg_per_sec = math.degrees(STRAFE_CM_PER_SEC / radius)
    turn_rate = ROTATE_RATE * deg_per_sec / TURN_DEG_PER_SEC
    duration = ORBIT_LAPS * 360 / deg_per_sec
    print(f"  orbiting {ORBIT_LAPS} lap(s), about {duration:.0f} s")

    moved = 0.0          # seconds actually spent circling (searching doesn't count)
    last = time.time()
    while moved < duration:
        d = robot.distance_cm()
        now = time.time()
        if d > LOST_CM:
            robot.stop()
            print("  object out of view, looking for it...")
            if not reacquire(robot):
                print("  lost the object, stopping the orbit")
                break
            last = time.time()
            continue
        moved += now - last
        last = now
        # Slide right while turning right keeps us on a circle facing the object;
        # the forward part pulls us back to ORBIT_CM if we drift in or out.
        forward = max(-DRIVE_SPEED, min(DRIVE_SPEED, 1.5 * (d - ORBIT_CM)))
        robot.move(right=DRIVE_SPEED, forward=forward, turn_right=turn_rate)
    robot.stop()
    time.sleep(0.2)


def approach(robot):
    """Drive forward until we're ORBIT_CM from the thing in front (gives up after 5 s).
    Returns None if the 'thing' disappears - it was a false reading."""
    end = time.time() + 5
    while time.time() < end:
        d = robot.distance_cm()
        if d > DETECT_CM + 20:
            robot.stop()
            print(f"  false alarm, nothing there (now reads {d:.0f} cm)")
            return None
        if d <= ORBIT_CM + 8:  # stop early: the robot keeps rolling a bit
            break
        robot.move(forward=DRIVE_SPEED, turn_right=robot.steer())
    robot.stop()
    time.sleep(0.3)

    # Rolled too close? Back up the way we came (that path is known to be clear).
    end = time.time() + 2
    d = robot.distance_cm()
    while d < ORBIT_CM - 3 and time.time() < end:
        robot.move(forward=-DRIVE_SPEED)
        d = robot.distance_cm()
    robot.stop()
    time.sleep(0.3)
    return robot.distance_cm()


def reacquire(robot):
    """Object slipped out of the sensor's view: look left, then right, in small steps.
    While circling to the right the object usually drifts to the LEFT, so try that first."""
    for step in [-10, -10, -10, -10, 50, 10, 10, 10]:   # left to -40 deg, then right to +40 deg
        turn(robot, step)
        if robot.distance_cm() <= LOST_CM:
            return True
    turn(robot, -40)  # face the original direction again
    return False


def sweep_room(robot):
    """Start in a corner with the WALL on the robot's right (room on its left).
    At each wall: U-turn LEFT, then RIGHT, then LEFT... Each U-turn = 90 deg, one lane
    width forward, 90 deg, so the robot comes back one lane over (a 180 deg spin on the
    spot would just drive back along the same line)."""
    start = time.time()
    objects_seen = 0
    next_turn_right = False  # first U-turn is to the LEFT
    if robot.tiles is not None:
        a = robot.tiles.angle()
        print("Camera: no tile lines visible - driving without them." if a is None
              else f"Camera sees the tile lines (tilt {a:+.1f} deg).")

    for lane in range(1, MAX_LANES + 1):
        print(f"Lane {lane}")
        lane_start = time.time()
        while True:
            if time.time() - start > MAX_RUN_SECONDS:
                print("Time limit reached.")
                return objects_seen

            # Only react if two readings in a row say something is close
            d1 = robot.distance_cm()
            d2 = robot.distance_cm() if d1 <= DETECT_CM else d1
            if d1 > DETECT_CM or d2 > DETECT_CM:
                robot.move(forward=DRIVE_SPEED, turn_right=robot.steer())
                continue

            robot.stop()
            print(f"  sensor: something at {d1:.0f} / {d2:.0f} cm "
                  f"(after {time.time() - lane_start:.1f} s in this lane)")
            d = approach(robot)
            if d is None:
                continue
            if not DETECT_OBJECTS or not is_narrow_object(robot, d):
                print(f"  wall: end of lane (stopped {d:.0f} cm from it)")
                break

            objects_seen += 1
            print(f"Object #{objects_seen} found")
            orbit(robot)
            # Go around it: prefer the side the next lanes are on, else the other side
            if not (move_sideways(robot, SIDESTEP_CM, next_turn_right) or
                    move_sideways(robot, SIDESTEP_CM, not next_turn_right)):
                print("  both sides blocked: end of lane")
                break

        if not next_lane(robot, next_turn_right):
            break
        next_turn_right = not next_turn_right

    return objects_seen


def spin_test(robot):
    """Turn on the spot; press Enter after exactly one full turn -> real TURN_DEG_PER_SEC."""
    input("Robot on the floor, nothing around it. Press Enter to start spinning,\n"
          "then press Enter AGAIN when it has turned exactly once (back to the start)...")
    import threading
    pressed = threading.Event()
    threading.Thread(target=lambda: (input(), pressed.set()), daemon=True).start()
    start = time.time()
    try:
        while not pressed.is_set():   # keep re-sending the turn until Enter
            robot.move(turn_right=ROTATE_RATE)
            time.sleep(0.1)
    finally:
        robot.stop()
    seconds = time.time() - start
    print(f"One turn took {seconds:.1f} s  ->  TURN_DEG_PER_SEC = "
          f"{360 / max(0.1, seconds - TURN_START_DELAY):.0f} (with TURN_START_DELAY = {TURN_START_DELAY})")


def straight_test(robot):
    """Drive forward for 4 s WITH the tile-line steering, printing what the camera sees."""
    input("Put the robot next to a tile joint, pointing along it, 1.5 m clear in front.\n"
          "Press Enter to drive for 4 s...")
    end = time.time() + 4.0
    while time.time() < end:
        steer = robot.steer()
        h = robot.heading
        print("  no lines" if h is None else f"  averaged angle {h:+5.1f} deg  steer {steer:+.3f}")
        robot.move(forward=DRIVE_SPEED, turn_right=steer)
    robot.stop()
    print("Did it follow the tile joint, or end up at an angle to it (which way)?")


def lines_test(robot):
    """Robot doesn't move. Prints the tile angle; in VNC also shows the picture with
    the lines it uses. Turn the robot CLOCKWISE by hand: the angle must go UP
    (positive). If it goes DOWN, set LINE_SIGN = -1."""
    if robot.tiles is None:
        print("No camera - can't test.")
        return
    show = bool(os.environ.get("DISPLAY"))
    snapshot = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lines.jpg")
    print("Ctrl+C to stop." + ("" if show else " (No VNC desktop: numbers only.)"))
    print(f"The latest picture with the lines in red is saved as {snapshot}")
    while True:
        a = robot.tiles.angle()
        print("  no tile lines found" if a is None else f"  angle {a:+5.1f} deg")
        if robot.tiles.last_frame is not None:
            img = robot.tiles.last_frame.copy()
            for x1, y1, x2, y2 in robot.tiles.last_lines:
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite(snapshot, img)
            if show:
                cv2.imshow("tile lines (q to quit)", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        time.sleep(0.2)


def calibration_test(robot):
    print("Wheels should be OFF the ground (or lots of space around).")
    for label, kwargs in [("forward", {"forward": DRIVE_SPEED}),
                          ("slide right", {"right": DRIVE_SPEED}),
                          ("turn right", {"turn_right": ROTATE_RATE})]:
        input(f"Press Enter to {label} for 1 s...")
        drive_for(robot, 1.0, **kwargs)
    print("Distance sensor (Ctrl+C to stop):")
    while True:
        print(f"  {robot.distance_cm():.0f} cm")
        time.sleep(0.3)


def stop_on_signal(signum, frame):
    # SSH dropping (SIGHUP) or `kill` (SIGTERM) -> behave like Ctrl+C so the motors stop
    raise KeyboardInterrupt


def main():
    # The other project's robot-bridge service (starts at boot) fights over the motors
    import subprocess
    state = subprocess.run(["systemctl", "is-active", "robot-bridge"],
                           capture_output=True, text=True).stdout.strip()
    if state == "active":
        raise SystemExit("robot-bridge is running and will stop the motors.\n"
                         "Run:  sudo systemctl stop robot-bridge   then start this again.")

    signal.signal(signal.SIGHUP, stop_on_signal)
    signal.signal(signal.SIGTERM, stop_on_signal)
    robot = Robot()
    try:
        if "--test" in sys.argv:
            calibration_test(robot)
        elif "--straight" in sys.argv:
            straight_test(robot)
        elif "--spin" in sys.argv:
            spin_test(robot)
        elif "--lines" in sys.argv:
            lines_test(robot)
        else:
            load_learned_turns()
            count = sweep_room(robot)
            print(f"Done. Circled {count} object(s).")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        robot.stop()
        if robot.tiles is not None:
            robot.tiles.release()


if __name__ == "__main__":
    main()

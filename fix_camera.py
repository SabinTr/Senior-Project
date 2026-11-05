"""Puts the arm (camera) in the forward-looking pose that roam_and_orbit.py uses, then lets
YOU tilt the camera until it points the right way:
    u = tilt up a little,  d = tilt down a little,  o = ok, done.
Runs once and exits. Nothing else moves. Run it ON THE ROBOT, from ~/sabin:
    python3 fix_camera.py
The servos keep holding the pose after this ends, as long as the robot stays switched on."""
import subprocess
import time

import roam_and_orbit as r       # same pose numbers (ARM_POSITION, ARM_PITCH) and motor board

CAMERA_SERVO = 3                 # the servo at the end of the arm that tilts the camera
STEP_PULSE = 25                  # one u/d press (servo pulses, 500-2500)
# Never READ servo positions here: reading needs the board's receive mode, and without it
# pwm_servo_read_position waits for an answer forever (the first version hung silently).


def camera_pulse(servos):
    """Pulse the arm move just sent to the camera servo (incl. the arm's calibration offset)."""
    import kinematics.arm_move_ik as arm_ik
    pulse = servos.get("servo3") if isinstance(servos, dict) else servos[0]
    try:
        pulse += int(arm_ik.deviation_data[str(CAMERA_SERVO)])
    except Exception:
        pass                     # no offset file: the plain pulse is what was sent
    return int(pulse)


def main():
    print("Setting the camera arm...", flush=True)
    state = subprocess.run(["systemctl", "is-active", "robot-bridge"],
                           capture_output=True, text=True).stdout.strip()
    if state == "active":
        raise SystemExit("robot-bridge is running and holds the arm's board.\n"
                         "Run:  sudo systemctl stop robot-bridge   then start this again.")
    from kinematics.arm_move_ik import ArmIK
    board = r.mecanum_lib.board
    ak = ArmIK()
    ak.board = board
    result = ak.setPitchRangeMoving(r.ARM_POSITION, r.ARM_PITCH, -90, 90, 1500)
    if not result:
        raise SystemExit(f"The arm can't reach {r.ARM_POSITION} at {r.ARM_PITCH} deg - not moved.")
    time.sleep(1.6)                  # wait for the 1.5 s move to finish
    # result = (servo pulses, pitch it could actually reach, time)
    print(f"Normal pose set (pitch {result[1]} deg). Servo pulses: {result[0]}")

    pulse = camera_pulse(result[0])
    start = pulse
    while True:
        answer = input(f"[camera servo {pulse}] u = tilt up, d = tilt down, o = ok: ").strip().lower()
        if answer in ("u", "d"):
            pulse += STEP_PULSE if answer == "u" else -STEP_PULSE
            pulse = max(500, min(2500, pulse))
            board.pwm_servo_set_position(0.3, [[CAMERA_SERVO, pulse]])
            time.sleep(0.4)
        elif answer == "o":
            break
    print(f"\nRESULT: camera servo {CAMERA_SERVO} = {pulse} (normal pose gave {start}). "
          "Tell Claude this number.")
    print("If u/d didn't move the camera at all: the arm has no power (battery?) or something "
          "blocks it. If u tilted it DOWN, that's fine - just use d to go up.")


if __name__ == "__main__":
    main()

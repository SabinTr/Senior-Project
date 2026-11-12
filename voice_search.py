"""Voice-controlled room search = the two working programs together. Run it ON THE ROBOT,
from ~/sabin (next to "Telegram voice robot.py", roam_and_orbit.py and yolov8n.pt):
    sudo systemctl stop robot-bridge
    python3 voice_search.py

"Telegram voice robot.py" does the Telegram part exactly as before: waits for your voice
note, checks that it's your voice, works out the object ("find a bottle"). Then, instead of
looking with the camera standing still (watch_for_object), it starts the PUSHED robot program,
unchanged, exactly as if you typed
    python3 roam_and_orbit.py --find bottle
It runs in its own process with nothing else loaded next to it, so the driving, the tile
steering and the U-turns are exactly the pushed code. When it finds the object it saves
object_found.jpg, and that photo is sent to Telegram.
Put the robot at its start position (straight along the tiles, wall on its right) before
each voice note. Ctrl+C stops the robot and this program."""
import importlib.util
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT_PROGRAM = os.path.join(HERE, "roam_and_orbit.py")
PHOTO = os.path.join(HERE, "object_found.jpg")   # roam_and_orbit.py saves it here

# the file name has spaces, so it can't be a normal "import"
_spec = importlib.util.spec_from_file_location(
    "telegram_voice_robot", os.path.join(HERE, "Telegram voice robot.py"))
telegram = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(telegram)


def search_room(model, target):
    """Used instead of the Telegram script's watch_for_object(): runs the pushed robot program."""
    if os.path.exists(PHOTO):
        os.remove(PHOTO)                          # so an old photo can't count as "found"
    robot = subprocess.Popen([sys.executable, "-u", ROBOT_PROGRAM, "--find", target], cwd=HERE)
    try:
        robot.wait()
    except KeyboardInterrupt:
        robot.wait()        # Ctrl+C reached the robot program too: let it stop the motors first
        raise
    if os.path.exists(PHOTO):
        telegram.tg_send_photo(PHOTO, caption=f"Found: {target}")
        return True
    if robot.returncode != 0:
        telegram.tg_send_message("The robot program stopped with an error - see the terminal.")
        return False
    telegram.tg_send_message(f"I searched the room but didn't find a {target}.")
    return False


def main():
    state = subprocess.run(["systemctl", "is-active", "robot-bridge"],
                           capture_output=True, text=True).stdout.strip()
    if state == "active":
        raise SystemExit("robot-bridge is running and will stop the motors.\n"
                         "Run:  sudo systemctl stop robot-bridge   then start this again.")
    telegram.watch_for_object = search_room   # handle_voice_note() calls it after "find X"
    telegram.main()


if __name__ == "__main__":
    main()

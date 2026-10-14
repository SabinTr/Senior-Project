"""
telegram_voice_robot.py

You send a VOICE NOTE to your Telegram bot (e.g. "find a bottle").
The Pi downloads it, checks that it's YOUR voice (using your voiceprint),
transcribes it, and if it's you, searches through the camera for that
object. When the object is confirmed it sends you ONE photo, then goes
back to waiting for the next voice note.

- Voice notes from anyone else, or from a different Telegram chat, are
  ignored (the bot tells you the voice wasn't recognized).
- Typed text messages are not accepted as commands.

Speaker verification uses SpeechBrain's ECAPA model. It is already in the
robot's shared Hugging Face cache (~/.cache/huggingface), so nothing is
downloaded; pretrained_models/ in this folder only holds small links to it.

--------------------------------------------------------------------
SETUP (on the Pi)
--------------------------------------------------------------------
sudo apt install ffmpeg flac -y
pip3 install --no-cache-dir SpeechRecognition --break-system-packages
(speechbrain, torch, ultralytics, opencv, numpy, scipy are already installed)

--------------------------------------------------------------------
USAGE
--------------------------------------------------------------------
Put your voice recordings (.wav) in the 'voice_samples' folder next to this
script, then run:
   python3 "Telegram voice robot.py"
On the first run (or whenever you add/change recordings) it builds
'my_voiceprint.npy' from them automatically. To force a rebuild:
   python3 "Telegram voice robot.py" --enroll
Press Ctrl+C to quit.
"""

import os
import re
import sys
import time
import subprocess
import numpy as np
import cv2
import requests
import speech_recognition as sr
from scipy.io import wavfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Pi resource limiting ---------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "3")
import torch
torch.set_num_threads(3)
# ---------------------------------------------------------------------------

from speechbrain.inference.speaker import EncoderClassifier
from ultralytics import YOLO

# --- Fill these in ----------------------------------------------------------
TELEGRAM_TOKEN = "YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"
# ---------------------------------------------------------------------------

VOICE_SAMPLES_DIR = os.path.join(SCRIPT_DIR, "voice_samples")
VOICEPRINT_FILE = os.path.join(SCRIPT_DIR, "my_voiceprint.npy")
SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
SPEAKER_MODEL_DIR = os.path.join(SCRIPT_DIR, "pretrained_models", "spkrec-ecapa-voxceleb")
EMBEDDING_SIZE = 192          # ECAPA voiceprint length
# ECAPA scores are lower than resemblyzer's: same person is usually 0.4-0.8,
# other people usually below 0.3. Watch the printed scores and tune this.
SIMILARITY_THRESHOLD = 0.40
TOP_K_MATCHES = 3
# Long recordings are cut into pieces about as long as a voice note,
# so each piece becomes its own voiceprint sample.
SAMPLE_CHUNK_SECONDS = 5
MIN_CHUNK_SECONDS = 2
SILENCE_RATIO = 0.3           # chunks quieter than 30% of the recording's average are skipped

CAMERA_INDEX = 0
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
FRAME_SKIP = 1
DETECTION_CONFIDENCE = 0.5
CONSECUTIVE_FRAMES_REQUIRED = 4
SEARCH_TIMEOUT = 120
SHOW_VIDEO_WINDOW = True  # needs a display (VNC desktop) if True

API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
FILE_API = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}"

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Things people say that don't match the YOLO name exactly
SYNONYMS = {
    "phone": "cell phone", "mobile": "cell phone", "mobile phone": "cell phone",
    "television": "tv", "sofa": "couch", "plant": "potted plant",
    "table": "dining table", "ball": "sports ball", "glass": "wine glass",
    "fridge": "refrigerator", "computer": "laptop", "bike": "bicycle",
    "motorbike": "motorcycle", "plane": "airplane", "teddy": "teddy bear",
}


# ----------------------------- Telegram helpers -----------------------------
def tg_send_message(text):
    requests.post(f"{API}/sendMessage",
                  data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=30)


def tg_send_photo(path, caption=None):
    with open(path, "rb") as photo:
        data = {"chat_id": TELEGRAM_CHAT_ID}
        if caption:
            data["caption"] = caption
        r = requests.post(f"{API}/sendPhoto", files={"photo": photo},
                          data=data, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Telegram API error: {r.text}")


def tg_get_updates(offset):
    r = requests.get(f"{API}/getUpdates",
                     params={"timeout": 30, "offset": offset}, timeout=40)
    r.raise_for_status()
    return r.json().get("result", [])


def tg_download_voice(file_id, dest_path):
    info = requests.get(f"{API}/getFile", params={"file_id": file_id}, timeout=30).json()
    file_path = info["result"]["file_path"]
    data = requests.get(f"{FILE_API}/{file_path}", timeout=60)
    data.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(data.content)


# ----------------------------- Voice helpers --------------------------------
def sample_files():
    if not os.path.isdir(VOICE_SAMPLES_DIR):
        return []
    return sorted(os.path.join(VOICE_SAMPLES_DIR, f) for f in os.listdir(VOICE_SAMPLES_DIR)
                  if f.lower().endswith(".wav"))


def voiceprint_is_stale():
    """True if my_voiceprint.npy is missing, from resemblyzer, or older than a recording."""
    if not os.path.exists(VOICEPRINT_FILE):
        return True
    vp = np.load(VOICEPRINT_FILE)
    if vp.ndim != 2 or vp.shape[1] != EMBEDDING_SIZE:
        return True
    built = os.path.getmtime(VOICEPRINT_FILE)
    return any(os.path.getmtime(f) > built for f in sample_files())


def build_voiceprints(encoder):
    """Turn every recording in voice_samples/ into voiceprint samples."""
    files = sample_files()
    if not files:
        raise SystemExit(f"No .wav recordings found in '{VOICE_SAMPLES_DIR}'.")

    print(f"Building voiceprint from {len(files)} recording(s)...")
    chunk_len = SAMPLE_CHUNK_SECONDS * 16000
    min_len = MIN_CHUNK_SECONDS * 16000
    samples = []
    for path in files:
        tmp = "sample_16k.wav"
        try:
            convert_to_wav(path, tmp)  # recordings are 48 kHz; the model needs 16 kHz
            audio = load_wav(tmp)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        avg_rms = np.sqrt(np.mean(audio ** 2))
        count = 0
        for start in range(0, len(audio), chunk_len):
            chunk = audio[start:start + chunk_len]
            if len(chunk) < min_len or np.sqrt(np.mean(chunk ** 2)) < SILENCE_RATIO * avg_rms:
                continue
            samples.append(embed_audio(encoder, chunk))
            count += 1
        print(f"  {os.path.basename(path)}: {count} sample(s)")

    vp = np.stack(samples)
    np.save(VOICEPRINT_FILE, vp)
    print(f"Saved {len(vp)} voiceprint samples to {VOICEPRINT_FILE}")
    return vp


def load_voiceprints(encoder, force_rebuild=False):
    if force_rebuild or voiceprint_is_stale():
        return build_voiceprints(encoder)
    return np.load(VOICEPRINT_FILE)


def load_speaker_encoder():
    print("Loading speaker model...")
    return EncoderClassifier.from_hparams(source=SPEAKER_MODEL, savedir=SPEAKER_MODEL_DIR,
                                          run_opts={"device": "cpu"})


def load_wav(wav_path):
    """16 kHz mono 16-bit wav -> float audio in [-1, 1]."""
    _, audio = wavfile.read(wav_path)
    return audio.astype(np.float32) / 32768.0


def embed_audio(encoder, audio):
    """Audio -> 192-number voiceprint."""
    signal = torch.from_numpy(audio).unsqueeze(0)
    with torch.no_grad():
        emb = encoder.encode_batch(signal)
    return emb.squeeze().cpu().numpy()


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def best_match_score(embedding, voiceprints, top_k=TOP_K_MATCHES):
    scores = sorted((cosine_similarity(embedding, vp) for vp in voiceprints), reverse=True)
    return float(np.mean(scores[:min(top_k, len(scores))]))


def convert_to_wav(src_path, wav_path):
    """Telegram voice notes are .ogg/opus. Convert to 16 kHz mono wav."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src_path,
         "-ar", "16000", "-ac", "1", wav_path],
        check=True,
    )


def extract_target_object(text):
    """'find a bottle' / 'look for the cell phone' / 'search for me a cup' -> class name."""
    text = text.lower().strip().rstrip(".!?")
    m = re.search(r"(?:find|look for|search for)\s+(?:me\s+)?(?:the|a|an)?\s*(.+)", text)
    if not m:
        return None
    phrase = m.group(1).strip()
    if phrase in COCO_CLASSES:
        return phrase
    if phrase in SYNONYMS:
        return SYNONYMS[phrase]
    for cls in COCO_CLASSES:
        if cls in phrase:
            return cls
    for word, cls in SYNONYMS.items():
        if word in phrase:
            return cls
    return None


# ----------------------------- Camera search --------------------------------
def watch_for_object(model, target_class):
    """Returns True if the object was confirmed and a photo was sent."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        tg_send_message("I couldn't open the camera.")
        return False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    print(f"Watching for '{target_class}'... (timeout: {SEARCH_TIMEOUT}s)")
    start = time.time()
    frame_count = 0
    hits = 0
    found = False

    try:
        while time.time() - start < SEARCH_TIMEOUT:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from the camera.")
                break

            frame_count += 1
            if frame_count % (FRAME_SKIP + 1) != 0:
                continue

            results = model(frame, verbose=False, conf=DETECTION_CONFIDENCE)
            annotated = results[0].plot()
            names = {model.names[int(b.cls[0])] for b in results[0].boxes}

            if SHOW_VIDEO_WINDOW:
                cv2.imshow("Searching - press 'q' to cancel", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if target_class in names:
                hits += 1
                print(f"  ...saw '{target_class}' ({hits}/{CONSECUTIVE_FRAMES_REQUIRED})")
            else:
                hits = 0

            if hits >= CONSECUTIVE_FRAMES_REQUIRED:
                cv2.imwrite("found_object.jpg", annotated)
                tg_send_photo("found_object.jpg", caption=f"Found: {target_class}")
                os.remove("found_object.jpg")
                found = True
                break
    finally:
        cap.release()
        if SHOW_VIDEO_WINDOW:
            cv2.destroyAllWindows()

    if not found:
        tg_send_message(f"I looked for {SEARCH_TIMEOUT}s but didn't see a {target_class}.")
    return found


# ----------------------------- Message handling -----------------------------
def handle_voice_note(message, encoder, voiceprints, model):
    file_id = message["voice"]["file_id"]
    src, wav = "incoming_voice.oga", "incoming_voice.wav"
    try:
        tg_download_voice(file_id, src)
        convert_to_wav(src, wav)

        embedding = embed_audio(encoder, load_wav(wav))
        score = best_match_score(embedding, voiceprints)
        print(f"Voice match score: {score:.2f}")

        if score < SIMILARITY_THRESHOLD:
            tg_send_message(f"Voice not recognized (match {score:.2f}). Ignoring.")
            return

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav) as source:
            audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            tg_send_message(f"Recognized your voice (match {score:.2f}) but couldn't understand the words.")
            return
        except sr.RequestError as e:
            tg_send_message(f"Speech recognition service error: {e}")
            return

        print(f"You said: {text}")
        target = extract_target_object(text)
        if not target:
            tg_send_message(f'I heard "{text}" but no object I know. Try "find a bottle".')
            return

        tg_send_message(f"Heard you (match {score:.2f}). Searching for: {target}")
        watch_for_object(model, target)
    finally:
        for p in (src, wav):
            if os.path.exists(p):
                os.remove(p)


def skip_old_updates():
    """Returns an offset that skips messages sent while the script wasn't running."""
    pending = tg_get_updates(offset=-1)
    return pending[-1]["update_id"] + 1 if pending else None


def main():
    if TELEGRAM_TOKEN == "YOUR_BOT_TOKEN_HERE" or TELEGRAM_CHAT_ID == "YOUR_CHAT_ID_HERE":
        raise SystemExit("Set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID at the top of this file first.")

    encoder = load_speaker_encoder()
    voiceprints = load_voiceprints(encoder, force_rebuild="--enroll" in sys.argv)
    print(f"Loaded {len(voiceprints)} enrolled voice sample(s).")
    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")

    offset = skip_old_updates()

    tg_send_message("Robot is online. Send me a voice note, like: find a bottle.")
    print("Waiting for voice notes... (Ctrl+C to quit)")

    try:
        while True:
            try:
                updates = tg_get_updates(offset)
            except requests.RequestException as e:
                print(f"(Network error, retrying: {e})")
                time.sleep(3)
                continue

            for update in updates:
                offset = update["update_id"] + 1
                message = update.get("message")
                if not message:
                    continue
                if str(message["chat"]["id"]) != str(TELEGRAM_CHAT_ID):
                    continue  # ignore everyone except you

                if "voice" in message:
                    try:
                        handle_voice_note(message, encoder, voiceprints, model)
                    except Exception as e:
                        print(f"Error handling voice note: {e}")
                        tg_send_message(f"Something went wrong: {e}")
                else:
                    tg_send_message("Please send a voice note.")
    except KeyboardInterrupt:
        print("\nStopping. Goodbye!")


if __name__ == "__main__":
    main()
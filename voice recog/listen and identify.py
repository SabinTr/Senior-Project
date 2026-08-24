"""
listen_and_identify.py

Listens to your microphone continuously. Every time it hears someone
speak, it checks the voice against your saved voiceprint (built by
build_profile.py). If it's YOU speaking, it transcribes what you said
to text. If it's someone else, it's ignored (no transcription).

--------------------------------------------------------------------
SETUP
--------------------------------------------------------------------
pip install resemblyzer SpeechRecognition pyaudio numpy scipy

IMPORTANT: Run build_profile.py first to create 'my_voiceprint.npy'.

--------------------------------------------------------------------
USAGE
--------------------------------------------------------------------
python listen_and_identify.py

Press Ctrl+C to quit.
"""

import os
import io
import numpy as np
import speech_recognition as sr
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.io import wavfile

VOICEPRINT_FILE = "my_voiceprint.npy"

# Cosine similarity ranges 0-1. Typical good threshold: 0.65-0.80.
# Lower it if it's rejecting your real voice too often.
# Raise it if it's accepting other people's voices as you.
# NOTE: since we now compare against each sample individually and take
# the best matches, scores tend to run higher than with a single
# averaged voiceprint, so you may need to re-tune this.
SIMILARITY_THRESHOLD = 0.65

# When comparing to multiple enrolled samples, how many of the top
# matches to average together (instead of just using the single best
# match, which can be noisy / a fluke).
TOP_K_MATCHES = 3

# How long (in seconds) of silence before a phrase is considered "done"
PAUSE_THRESHOLD = 2.0


def load_voiceprints():
    if not os.path.exists(VOICEPRINT_FILE):
        raise FileNotFoundError(
            f"'{VOICEPRINT_FILE}' not found. Run build_profile.py first "
            "to create your voiceprint."
        )
    voiceprints = np.load(VOICEPRINT_FILE)
    # Backward compatibility: if an old single-vector voiceprint is
    # loaded, wrap it so it still works as a 1-sample array.
    if voiceprints.ndim == 1:
        voiceprints = voiceprints[np.newaxis, :]
    return voiceprints


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def best_match_score(embedding, voiceprints, top_k=TOP_K_MATCHES):
    """Compare embedding against every enrolled sample, return the
    average of the top-k highest similarity scores."""
    scores = [cosine_similarity(embedding, vp) for vp in voiceprints]
    scores.sort(reverse=True)
    top_scores = scores[:min(top_k, len(scores))]
    return float(np.mean(top_scores))


def audio_to_embedding(audio_data, encoder):
    """Convert a speech_recognition AudioData object into a resemblyzer embedding."""
    wav_bytes = audio_data.get_wav_data()
    sample_rate, wav_np = wavfile.read(io.BytesIO(wav_bytes))
    wav_np = wav_np.astype(np.float32) / np.iinfo(np.int16).max
    wav_processed = preprocess_wav(wav_np, source_sr=sample_rate)
    return encoder.embed_utterance(wav_processed)


def listen_and_identify():
    my_voiceprints = load_voiceprints()
    print(f"Loaded {len(my_voiceprints)} enrolled sample(s).\n")
    encoder = VoiceEncoder()
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = PAUSE_THRESHOLD

    with sr.Microphone(sample_rate=16000) as source:
        print("Calibrating for ambient noise... stay quiet for a moment.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Ready. Start speaking (Ctrl+C to stop).\n")

        while True:
            try:
                print("Listening...")
                audio = recognizer.listen(source, phrase_time_limit=15)

                embedding = audio_to_embedding(audio, encoder)
                similarity = best_match_score(embedding, my_voiceprints)

                if similarity >= SIMILARITY_THRESHOLD:
                    print(f"This is YOU speaking. (match: {similarity:.2f})")
                    print("Transcribing...")
                    try:
                        text = recognizer.recognize_google(audio)
                        print(f"You said: {text}\n")
                    except sr.UnknownValueError:
                        print("(Couldn't understand the words)\n")
                    except sr.RequestError as e:
                        print(f"(Could not reach speech recognition service: {e})\n")
                else:
                    print(f"This is NOT you speaking. (match: {similarity:.2f}) - ignored\n")

            except KeyboardInterrupt:
                print("\nStopping. Goodbye!")
                break


if __name__ == "__main__":
    listen_and_identify()
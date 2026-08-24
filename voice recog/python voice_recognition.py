import os
import numpy as np
import speech_recognition as sr
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.io import wavfile
import io
 
VOICEPRINT_FILE = "my_voiceprint.npy"
 
# How similar a voice needs to be to count as "you".
# Cosine similarity ranges 0-1. Typical good threshold: 0.75-0.80
# Lower it if it's rejecting your real voice too often.
# Raise it if it's accepting other people's voices.
SIMILARITY_THRESHOLD = 0.60
 
# Set to True to print a message even when speech is ignored
# (useful for tuning the threshold above)
VERBOSE = True
 
 
def load_voiceprint():
    if not os.path.exists(VOICEPRINT_FILE):
        raise FileNotFoundError(
            f"'{VOICEPRINT_FILE}' not found. Run enroll_voice.py first "
            "to create your voiceprint."
        )
    return np.load(VOICEPRINT_FILE)
 
 
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
 
 
def audio_to_embedding(audio_data, encoder):
    """Convert a speech_recognition AudioData object into a resemblyzer embedding."""
    wav_bytes = audio_data.get_wav_data()
    sample_rate, wav_np = wavfile.read(io.BytesIO(wav_bytes))
    wav_np = wav_np.astype(np.float32) / np.iinfo(np.int16).max
    wav_processed = preprocess_wav(wav_np, source_sr=sample_rate)
    return encoder.embed_utterance(wav_processed)
 
 
def listen_and_transcribe():
    my_voiceprint = load_voiceprint()
    encoder = VoiceEncoder()
    recognizer = sr.Recognizer()
 
    with sr.Microphone(sample_rate=16000) as source:
        print("Calibrating for ambient noise... stay quiet for a moment.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Ready. Start speaking (Ctrl+C to stop).\n")
 
        while True:
            try:
                print("Listening...")
                audio = recognizer.listen(source, phrase_time_limit=10)
 
                # Check who's speaking
                embedding = audio_to_embedding(audio, encoder)
                similarity = cosine_similarity(embedding, my_voiceprint)
 
                if similarity >= SIMILARITY_THRESHOLD:
                    print("Transcribing...")
                    try:
                        text = recognizer.recognize_google(audio)
                        print(f"You said: {text}  (match: {similarity:.2f})\n")
                    except sr.UnknownValueError:
                        print("(Your voice detected, but couldn't understand words)\n")
                    except sr.RequestError as e:
                        print(f"(Could not reach speech recognition service: {e})\n")
                else:
                    if VERBOSE:
                        print(f"(Ignored - not your voice, match: {similarity:.2f})\n")
 
            except KeyboardInterrupt:
                print("\nStopping. Goodbye!")
                break
 
 
if __name__ == "__main__":
    listen_and_transcribe()
 
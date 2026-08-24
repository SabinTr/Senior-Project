"""
build_profile.py

Builds YOUR voiceprint from a folder of pre-recorded .wav samples of
your voice (instead of recording live through the mic).

--------------------------------------------------------------------
SETUP
--------------------------------------------------------------------
pip install resemblyzer numpy

--------------------------------------------------------------------
PREPARE YOUR SAMPLES
--------------------------------------------------------------------
1. Create a folder called "voice_samples" next to this script.
2. Put multiple .wav recordings of YOUR voice in it (e.g. sample1.wav,
   sample2.wav, sample3.wav...). More samples = a more robust profile.
   - Ideally: 3-10 clips, each 5-20 seconds, clear speech, minimal
     background noise, varied tone/sentences if possible.
   - Any sample rate/mono or stereo is fine, they'll be auto-converted.

--------------------------------------------------------------------
USAGE
--------------------------------------------------------------------
python build_profile.py

This reads every .wav file in "voice_samples", computes an embedding
for each, averages them into a single voiceprint, and saves it to
"my_voiceprint.npy".
"""

import os
import glob
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

SAMPLES_DIR = "voice_samples"
VOICEPRINT_FILE = "my_voiceprint.npy"

"""
--------------------------------------------------------------------
SAMPLE SCRIPTS TO READ ALOUD WHEN RECORDING voice_samples/*.wav
--------------------------------------------------------------------
Record a few different ones (not just the same line repeated) so the
voiceprint captures variety in pitch, speed, and sounds. Speak at your
normal volume/pace, in the same room/mic setup you'll actually use.

Sample 1 (neutral, everyday speech):
    "Hi, my name is Nancy. I'm recording this sample to help my
    computer recognize my voice. Today the weather is pretty nice
    outside, and I'm just talking normally like I would in a regular
    conversation."

Sample 2 (numbers and technical words - good phonetic variety):
    "One, two, three, four, five, six, seven, eight, nine, ten. I have
    three meetings today, one at nine, one at noon, and one at four
    thirty. My phone number ends in seven eight two one."

Sample 3 (questions and varied intonation):
    "How are you doing today? Did you get a chance to check the email
    I sent? I was wondering if we could meet sometime this week to go
    over the project details."

Sample 4 (slightly longer, storytelling tone):
    "Yesterday I went for a walk in the park and saw a few dogs
    playing near the pond. It was a really relaxing afternoon, and I
    ended up staying out much longer than I planned. On the way back,
    I stopped to get some coffee."

Sample 5 (short commands - useful if you'll use this for voice
commands later):
    "Turn on the lights. Play some music. What's the weather like
    today? Set a timer for ten minutes. Open my calendar."

Tips:
- Don't perform or over-enunciate, just talk normally.
- Try one sample close to the mic and one a bit farther away, to
  capture natural variation.
- Save each as a separate .wav file in the voice_samples/ folder
  (e.g. sample1.wav, sample2.wav, ...).
--------------------------------------------------------------------
"""


def build_profile():
    wav_paths = sorted(glob.glob(os.path.join(SAMPLES_DIR, "*.wav")))

    if not wav_paths:
        raise FileNotFoundError(
            f"No .wav files found in '{SAMPLES_DIR}/'. Add some recordings "
            "of your voice there first."
        )

    print(f"Found {len(wav_paths)} sample(s):")
    for p in wav_paths:
        print(f"  - {p}")
    print()

    encoder = VoiceEncoder()
    embeddings = []

    for path in wav_paths:
        print(f"Processing {path}...")
        wav = preprocess_wav(path)
        embedding = encoder.embed_utterance(wav)
        embeddings.append(embedding)

    # Save ALL individual embeddings (not averaged). Comparing new audio
    # against each sample separately (instead of one blurred-together
    # average) gives noticeably better separation between "you" and
    # "not you" scores.
    voiceprints = np.stack(embeddings, axis=0)  # shape: (num_samples, 256)

    np.save(VOICEPRINT_FILE, voiceprints)
    print(f"\nVoiceprint profile built from {len(wav_paths)} sample(s).")
    print(f"Saved to '{VOICEPRINT_FILE}' (shape: {voiceprints.shape}).")


if __name__ == "__main__":
    build_profile()
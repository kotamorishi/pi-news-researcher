#!/usr/bin/env python3
"""Register a speaker voice profile for identification."""

import sys
import json
import numpy as np
import subprocess
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav

PROFILES_DIR = Path("/home/kota/hailo-apps/speaker_profiles")
PROFILES_DIR.mkdir(exist_ok=True)

def record_audio(duration=10):
    """Record audio from USB mic and return as numpy array."""
    wav_path = "/tmp/speaker_register.wav"
    print(f"Recording {duration} seconds... Speak now!")
    subprocess.run([
        "arecord", "-D", "plughw:2,0", "-f", "S16_LE",
        "-r", "16000", "-c", "1", "-d", str(duration), wav_path
    ], check=True)
    print("Recording complete.")
    return wav_path

def main():
    if len(sys.argv) < 2:
        print("Usage: python register_speaker.py <name>")
        print("Example: python register_speaker.py kota")
        sys.exit(1)

    name = sys.argv[1]
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    wav_path = record_audio(duration)

    print("Computing voice embedding...")
    encoder = VoiceEncoder()
    wav = preprocess_wav(wav_path)
    embedding = encoder.embed_utterance(wav)

    profile_path = PROFILES_DIR / f"{name}.npy"
    np.save(profile_path, embedding)
    print(f"Saved voice profile: {profile_path}")
    print(f"Embedding shape: {embedding.shape}")

if __name__ == "__main__":
    main()

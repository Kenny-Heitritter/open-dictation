"""Training data collection for Whisper fine-tuning.

Saves audio + transcript pairs from each dictation session.
The user can later review and correct transcripts, then fine-tune
a personal Whisper model that adapts to their voice.

Data layout in ~/.dictation-training/:
    samples/
        {id}.wav          - 16kHz mono WAV (Whisper's native format)
        {id}.json         - metadata: transcript, timestamp, corrected flag
    manifest.jsonl        - append-only log of all samples
"""

from __future__ import annotations

import base64
import json
import os
import struct
import sys
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np

# ── Configuration ────────────────────────────────────────────────────

TRAINING_DIR = Path(os.environ.get(
    "DICTATION_TRAINING_DIR",
    os.path.expanduser("~/.dictation-training"),
))
SAMPLES_DIR = TRAINING_DIR / "samples"
MANIFEST_PATH = TRAINING_DIR / "manifest.jsonl"

# Minimum audio duration (seconds) to bother saving
MIN_DURATION = 0.5

# Target sample rate for Whisper
TARGET_SR = 16000


def _ensure_dirs() -> None:
    """Create training data directories if needed."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def save_sample(
    audio_b64: str,
    transcript: str,
    sample_rate: int = 16000,
) -> Optional[str]:
    """Save an audio + transcript pair for future fine-tuning.

    Parameters
    ----------
    audio_b64 : str
        Base64-encoded raw audio bytes (float32, mono).
        Comes from recorder.last_transcription_bytes_b64.
    transcript : str
        The transcription produced by Whisper.
    sample_rate : int
        Sample rate of the audio (RealtimeSTT uses 16000).

    Returns
    -------
    sample_id or None if skipped.
    """
    if not transcript or not transcript.strip():
        return None

    try:
        raw_bytes = base64.b64decode(audio_b64)
        audio = np.frombuffer(raw_bytes, dtype=np.float32)
    except Exception as e:
        sys.stdout.write(f"[Training] Failed to decode audio: {e}\n")
        sys.stdout.flush()
        return None

    duration = len(audio) / sample_rate
    if duration < MIN_DURATION:
        return None

    _ensure_dirs()

    # Generate a sortable ID: timestamp_millis
    sample_id = f"{int(time.time() * 1000)}"

    # Write WAV (16-bit PCM, which is what Whisper training expects)
    wav_path = SAMPLES_DIR / f"{sample_id}.wav"
    pcm_16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_16.tobytes())

    # Write metadata
    meta = {
        "id": sample_id,
        "transcript": transcript.strip(),
        "corrected_transcript": None,  # filled in by correction tool
        "duration_s": round(duration, 2),
        "sample_rate": sample_rate,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corrected": False,
    }
    meta_path = SAMPLES_DIR / f"{sample_id}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Append to manifest (append-only log)
    with open(MANIFEST_PATH, "a") as f:
        f.write(json.dumps({"id": sample_id, "transcript": transcript.strip()}) + "\n")

    return sample_id


def get_sample_count() -> int:
    """Return the number of collected samples."""
    if not SAMPLES_DIR.exists():
        return 0
    return len(list(SAMPLES_DIR.glob("*.wav")))


def get_corrected_count() -> int:
    """Return the number of samples that have been corrected."""
    if not SAMPLES_DIR.exists():
        return 0
    count = 0
    for meta_path in SAMPLES_DIR.glob("*.json"):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("corrected"):
                count += 1
        except Exception:
            pass
    return count

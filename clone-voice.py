#!/usr/bin/env python3
"""
clone-voice.py -- Record a voice sample, build a clone prompt, launch the agent.

Three steps:
  1. RECORD  - Records 10-20s of a voice sample via PulseAudio
  2. EXTRACT - Builds a Qwen3-TTS voice clone prompt from the recording
  3. LAUNCH  - Starts the dictation agent with the cloned voice

Usage:
  python clone-voice.py                    # Full flow: record -> extract -> launch
  python clone-voice.py --record-only      # Just record a new sample
  python clone-voice.py --skip-record      # Use existing sample, re-extract + launch
  python clone-voice.py --launch-only      # Use existing clone prompt, just launch

Voice data is saved to ~/.dictation-voice/:
  sample.wav       - Your reference recording
  transcript.txt   - What you said during recording
  clone_prompt.pt  - Cached voice features (reused across sessions)
"""

from __future__ import annotations

import argparse
import json
import os
import readline  # noqa: F401 — enables line editing in input()
import signal
import subprocess
import sys
import time
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────

VOICE_DIR = Path.home() / ".dictation-voice"
SAMPLE_WAV = VOICE_DIR / "sample.wav"
TRANSCRIPT_FILE = VOICE_DIR / "transcript.txt"
CLONE_PROMPT_FILE = VOICE_DIR / "clone_prompt.pt"

REPO_DIR = Path(__file__).parent
DICTATION_SCRIPT = REPO_DIR / "dictation.py"

# Use the dictation-env venv
VENV_PYTHON = Path.home() / "dictation-env" / "bin" / "python"

# ── Colors ───────────────────────────────────────────────────────────

BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"


def info(msg: str) -> None:
    print(f"  {GREEN}[+]{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}[!]{RESET} {msg}")


def error(msg: str) -> None:
    print(f"  {RED}[ERROR]{RESET} {msg}", file=sys.stderr)


def header(msg: str) -> None:
    print(f"\n{BOLD}{CYAN}=== {msg} ==={RESET}")


# ── Step 1: Record ──────────────────────────────────────────────────

def list_sources() -> list[dict]:
    """List PulseAudio input sources."""
    try:
        result = subprocess.run(
            ["pactl", "-f", "json", "list", "sources"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            sources = json.loads(result.stdout)
            # Filter to actual input devices (not monitors)
            return [
                s for s in sources
                if not s.get("name", "").endswith(".monitor")
                and s.get("name", "")
            ]
    except Exception:
        pass
    return []


def pick_source() -> str:
    """Let the user pick a microphone."""
    sources = list_sources()
    if not sources:
        warn("Could not list PulseAudio sources, using default mic.")
        return ""

    print(f"\n  Available microphones:\n")
    for i, s in enumerate(sources):
        desc = s.get("description", s.get("name", "Unknown"))
        state = s.get("state", "")
        marker = f" {DIM}({state}){RESET}" if state else ""
        print(f"    {BOLD}{i + 1}.{RESET} {desc}{marker}")

    print(f"    {BOLD}0.{RESET} Default (let PulseAudio choose)")
    print()

    while True:
        try:
            choice = input(f"  Pick a mic [0-{len(sources)}]: ").strip()
            if not choice or choice == "0":
                return ""
            idx = int(choice) - 1
            if 0 <= idx < len(sources):
                name = sources[idx]["name"]
                desc = sources[idx].get("description", name)
                info(f"Using: {desc}")
                return name
        except (ValueError, EOFError):
            pass
        print(f"    Enter 0-{len(sources)}")


def record_sample(source: str = "") -> bool:
    """Record a voice sample interactively."""
    header("Step 1: Record Your Voice")

    VOICE_DIR.mkdir(parents=True, exist_ok=True)

    if SAMPLE_WAV.exists():
        print(f"\n  Existing recording found at {SAMPLE_WAV}")
        resp = input("  Overwrite? [y/N]: ").strip().lower()
        if resp not in ("y", "yes"):
            info("Keeping existing recording.")
            return True

    print(f"""
  {BOLD}Instructions:{RESET}
  - Read the prompt below naturally at your normal speaking pace.
  - Aim for 10-20 seconds of clear speech.
  - Press {BOLD}Ctrl+C{RESET} when you're done.

  {BOLD}Suggested prompt:{RESET}
  {DIM}"Hello, my name is [your name]. I'm testing a voice cloning system.
   The quick brown fox jumps over the lazy dog. Today I'm going to set up
   my personal voice assistant that sounds just like me."{RESET}
""")

    transcript = input(
        f"  {BOLD}What will you read?{RESET} (paste/type the text, or press Enter for the default)\n  > "
    ).strip()

    if not transcript:
        transcript = (
            "Hello, my name is Kenny. I'm testing a voice cloning system. "
            "The quick brown fox jumps over the lazy dog. Today I'm going to "
            "set up my personal voice assistant that sounds just like me."
        )
        info(f"Using default prompt.")

    # Save transcript
    TRANSCRIPT_FILE.write_text(transcript)

    # Build parec command
    cmd = [
        "parec",
        "--format=s16le",
        "--rate=24000",
        "--channels=1",
    ]
    if source:
        cmd.extend(["--device", source])

    print(f"\n  {BOLD}{RED}>>> Recording starts NOW -- speak! (Ctrl+C to stop) <<<{RESET}\n")
    sys.stdout.flush()

    # Record raw PCM, convert to WAV after
    raw_file = VOICE_DIR / "sample_raw.pcm"
    try:
        with open(raw_file, "wb") as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.DEVNULL)
            try:
                proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                proc.wait()
    except FileNotFoundError:
        error("'parec' not found. Install pulseaudio-utils.")
        return False

    # Check we got some audio
    raw_size = raw_file.stat().st_size
    duration = raw_size / (24000 * 2)  # 16-bit mono @ 24kHz
    if duration < 2.0:
        error(f"Recording too short ({duration:.1f}s). Need at least 5 seconds.")
        raw_file.unlink(missing_ok=True)
        return False

    info(f"Recorded {duration:.1f} seconds.")

    # Convert raw PCM to WAV using sox
    try:
        subprocess.run(
            [
                "sox",
                "-t", "raw", "-r", "24000", "-e", "signed", "-b", "16", "-c", "1",
                str(raw_file),
                str(SAMPLE_WAV),
            ],
            check=True, capture_output=True, timeout=10,
        )
        raw_file.unlink(missing_ok=True)
    except FileNotFoundError:
        # Fallback: write WAV header manually
        import struct
        import wave

        with open(raw_file, "rb") as f:
            pcm_data = f.read()
        with wave.open(str(SAMPLE_WAV), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(pcm_data)
        raw_file.unlink(missing_ok=True)
    except subprocess.CalledProcessError as e:
        error(f"sox conversion failed: {e}")
        return False

    # Playback for confirmation
    print(f"\n  Playing back your recording...")
    subprocess.run(
        ["paplay", str(SAMPLE_WAV)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    resp = input("\n  Sound good? [Y/n]: ").strip().lower()
    if resp in ("n", "no"):
        warn("Discarded. Run again to re-record.")
        SAMPLE_WAV.unlink(missing_ok=True)
        return False

    info(f"Saved to {SAMPLE_WAV}")
    return True


# ── Step 2: Extract voice features ──────────────────────────────────

def extract_voice(force: bool = False) -> bool:
    """Build the voice clone prompt from the recorded sample."""
    header("Step 2: Extract Voice Features")

    if not SAMPLE_WAV.exists():
        error(f"No recording found at {SAMPLE_WAV}. Run with --record first.")
        return False

    if CLONE_PROMPT_FILE.exists() and not force:
        info(f"Clone prompt already exists at {CLONE_PROMPT_FILE}")
        resp = input("  Rebuild? [y/N]: ").strip().lower()
        if resp not in ("y", "yes"):
            return True

    transcript = ""
    if TRANSCRIPT_FILE.exists():
        transcript = TRANSCRIPT_FILE.read_text().strip()
        info(f"Transcript: {transcript[:60]}{'...' if len(transcript) > 60 else ''}")

    info("Loading Qwen3-TTS Base model (this may download ~2GB on first run)...")
    sys.stdout.flush()

    try:
        import torch
        import soundfile as sf
        from qwen_tts import Qwen3TTSModel

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        info(f"Device: {device}")

        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map=device,
            dtype=torch.bfloat16,
        )
        info("Model loaded.")

        # Load reference audio
        ref_audio, ref_sr = sf.read(str(SAMPLE_WAV), dtype="float32")
        if ref_sr != 24000:
            info(f"Resampling from {ref_sr}Hz to 24000Hz...")
            import librosa
            ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=24000)
            ref_sr = 24000

        duration = len(ref_audio) / ref_sr
        info(f"Reference audio: {duration:.1f}s, {ref_sr}Hz")

        # Build clone prompt
        info("Extracting voice features...")
        t0 = time.monotonic()
        voice_prompt = model.create_voice_clone_prompt(
            ref_audio=(ref_audio, ref_sr),
            ref_text=transcript,
        )
        elapsed = time.monotonic() - t0
        info(f"Voice features extracted in {elapsed:.1f}s")

        # Save the prompt
        torch.save(voice_prompt, str(CLONE_PROMPT_FILE))
        info(f"Saved clone prompt to {CLONE_PROMPT_FILE}")

        # Quick test synthesis
        info("Test synthesis with your cloned voice...")
        wavs, sr = model.generate_voice_clone(
            text="Voice cloning is ready. Let's begin.",
            language="English",
            voice_clone_prompt=voice_prompt,
        )

        test_wav = VOICE_DIR / "test_output.wav"
        import numpy as np
        test_audio = wavs[0]
        peak = np.max(np.abs(test_audio))
        if peak > 1e-6:
            test_audio = test_audio * (0.89 / peak)  # normalize to -1 dBFS
        sf.write(str(test_wav), test_audio, sr)
        info(f"Playing test audio...")
        subprocess.run(
            ["paplay", str(test_wav)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        resp = input("\n  Does it sound like you? [Y/n]: ").strip().lower()
        if resp in ("n", "no"):
            warn("You can re-record (--record-only) or try a longer sample.")
        else:
            info("Voice clone ready!")

        return True

    except ImportError as e:
        error(f"Missing package: {e}")
        error("Run with: ~/dictation-env/bin/python clone-voice.py")
        return False
    except Exception as e:
        error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ── Step 3: Launch agent ─────────────────────────────────────────────

def launch_agent() -> None:
    """Start the dictation agent with the cloned voice."""
    header("Step 3: Launch Agent")

    if not CLONE_PROMPT_FILE.exists():
        error("No clone prompt found. Run extraction first.")
        sys.exit(1)

    if not SAMPLE_WAV.exists():
        error("No voice sample found.")
        sys.exit(1)

    transcript = ""
    if TRANSCRIPT_FILE.exists():
        transcript = TRANSCRIPT_FILE.read_text().strip()

    model = os.environ.get("LLM_MODEL", "qwen3.5:9b")
    info(f"LLM model: {model}")
    info(f"Voice sample: {SAMPLE_WAV}")
    info(f"Clone prompt: {CLONE_PROMPT_FILE}")
    info("Starting dictation agent with your cloned voice...\n")

    env = os.environ.copy()
    env["LLM_MODEL"] = model
    env["TTS_VOICE_SAMPLE"] = str(SAMPLE_WAV)
    env["TTS_VOICE_SAMPLE_TRANSCRIPT"] = transcript
    env["TTS_CLONE_PROMPT"] = str(CLONE_PROMPT_FILE)

    python = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

    os.execve(
        python,
        [python, "-u", str(DICTATION_SCRIPT), "--agent"],
        env,
    )


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Record a voice sample, build a clone prompt, launch the agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--record-only", action="store_true",
        help="Only record a new voice sample",
    )
    parser.add_argument(
        "--skip-record", action="store_true",
        help="Skip recording, use existing sample",
    )
    parser.add_argument(
        "--launch-only", action="store_true",
        help="Skip record+extract, just launch with existing clone prompt",
    )
    parser.add_argument(
        "--force-extract", action="store_true",
        help="Force re-extraction even if clone prompt exists",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}Voice Clone Setup{RESET}")
    print(f"  Data directory: {VOICE_DIR}")

    if args.launch_only:
        launch_agent()
        return

    # Step 1: Record
    if not args.skip_record:
        source = pick_source()
        if not record_sample(source):
            if args.record_only:
                sys.exit(1)
            if not SAMPLE_WAV.exists():
                sys.exit(1)

    if args.record_only:
        info("Recording saved. Run again without --record-only to extract and launch.")
        return

    # Step 2: Extract
    if not extract_voice(force=args.force_extract):
        sys.exit(1)

    # Step 3: Launch
    resp = input(f"\n  Launch the agent now? [Y/n]: ").strip().lower()
    if resp in ("n", "no"):
        info("Done. Launch later with: python clone-voice.py --launch-only")
        return

    launch_agent()


if __name__ == "__main__":
    main()

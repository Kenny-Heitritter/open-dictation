#!/usr/bin/env python3
"""
correct-transcripts.py -- Review and correct Whisper transcriptions.

Plays each audio clip and shows what Whisper heard.  You can:
  - Press Enter to accept (mark as correct)
  - Type the correction and press Enter
  - Type 's' to skip
  - Type 'd' to delete the sample
  - Type 'q' to quit

Corrected samples are used by finetune-whisper.py to train a
personalized Whisper model.

Usage:
  python correct-transcripts.py              # Review uncorrected samples
  python correct-transcripts.py --all        # Review all samples (re-correct)
  python correct-transcripts.py --stats      # Show collection stats
"""

from __future__ import annotations

import argparse
import json
import os
import readline  # noqa: F401 — enables line editing in input()
import subprocess
import sys
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────

TRAINING_DIR = Path(os.environ.get(
    "DICTATION_TRAINING_DIR",
    os.path.expanduser("~/.dictation-training"),
))
SAMPLES_DIR = TRAINING_DIR / "samples"

# ── Colors ───────────────────────────────────────────────────────────

BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"


def load_samples(include_corrected: bool = False) -> list[dict]:
    """Load sample metadata, sorted by timestamp."""
    if not SAMPLES_DIR.exists():
        return []

    samples = []
    for meta_path in sorted(SAMPLES_DIR.glob("*.json")):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            wav_path = SAMPLES_DIR / f"{meta['id']}.wav"
            if not wav_path.exists():
                continue
            if not include_corrected and meta.get("corrected"):
                continue
            meta["_wav_path"] = str(wav_path)
            meta["_meta_path"] = str(meta_path)
            samples.append(meta)
        except Exception:
            continue

    return samples


def play_audio(wav_path: str) -> None:
    """Play a WAV file through system speaker."""
    for cmd in [
        ["paplay", wav_path],
        ["aplay", "-q", wav_path],
        ["afplay", wav_path],  # macOS
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", wav_path],
    ]:
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=30)
            return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    print(f"  {RED}No audio player found{RESET}")


def show_stats() -> None:
    """Show training data collection stats."""
    all_samples = load_samples(include_corrected=True)
    corrected = [s for s in all_samples if s.get("corrected")]
    uncorrected = [s for s in all_samples if not s.get("corrected")]

    total_duration = sum(s.get("duration_s", 0) for s in all_samples)
    corrected_duration = sum(s.get("duration_s", 0) for s in corrected)

    print(f"\n{BOLD}Training Data Stats{RESET}")
    print(f"  Location:    {TRAINING_DIR}")
    print(f"  Total:       {len(all_samples)} samples ({total_duration:.0f}s audio)")
    print(f"  Corrected:   {GREEN}{len(corrected)}{RESET} ({corrected_duration:.0f}s)")
    print(f"  Uncorrected: {YELLOW}{len(uncorrected)}{RESET}")
    print()

    if len(corrected) < 50:
        needed = 50 - len(corrected)
        print(f"  {DIM}Need ~{needed} more corrected samples for fine-tuning.{RESET}")
        print(f"  {DIM}(50 minimum, 100+ recommended){RESET}")
    else:
        print(f"  {GREEN}Ready for fine-tuning!{RESET} Run: python finetune-whisper.py")
    print()


def review_samples(include_corrected: bool = False) -> None:
    """Interactive review loop."""
    samples = load_samples(include_corrected=include_corrected)

    if not samples:
        if include_corrected:
            print(f"\n  No samples found in {SAMPLES_DIR}")
            print(f"  Run the dictation system to collect training data.\n")
        else:
            print(f"\n  {GREEN}All samples have been reviewed!{RESET}")
            show_stats()
        return

    print(f"\n{BOLD}Review Transcriptions{RESET}")
    print(f"  {len(samples)} samples to review\n")
    print(f"  {DIM}Enter{RESET} = accept as correct")
    print(f"  {DIM}Type correction{RESET} = save corrected transcript")
    print(f"  {DIM}r{RESET} = replay audio")
    print(f"  {DIM}s{RESET} = skip")
    print(f"  {DIM}d{RESET} = delete sample")
    print(f"  {DIM}q{RESET} = quit\n")

    corrected_count = 0
    total = len(samples)

    for i, sample in enumerate(samples):
        transcript = sample["transcript"]
        duration = sample.get("duration_s", 0)
        existing_correction = sample.get("corrected_transcript")

        # Header
        print(f"  {BOLD}[{i + 1}/{total}]{RESET} {DIM}({duration:.1f}s){RESET}")
        if existing_correction:
            print(f"  Original:  {DIM}{transcript}{RESET}")
            print(f"  Corrected: {CYAN}{existing_correction}{RESET}")
        else:
            print(f"  Whisper:   {CYAN}{transcript}{RESET}")

        # Play audio
        play_audio(sample["_wav_path"])

        while True:
            try:
                response = input(f"  > ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n\n  {corrected_count} samples corrected.")
                return

            if response.lower() == "q":
                print(f"\n  {corrected_count} samples corrected.")
                return

            if response.lower() == "s":
                print()
                break

            if response.lower() == "r":
                play_audio(sample["_wav_path"])
                continue

            if response.lower() == "d":
                # Delete sample
                try:
                    os.unlink(sample["_wav_path"])
                    os.unlink(sample["_meta_path"])
                    print(f"  {RED}Deleted.{RESET}\n")
                except Exception as e:
                    print(f"  {RED}Delete failed: {e}{RESET}")
                break

            # Accept or correct
            if response == "":
                # Accept current transcript as correct
                correction = existing_correction or transcript
            else:
                correction = response

            # Save correction
            sample["corrected_transcript"] = correction
            sample["corrected"] = True
            with open(sample["_meta_path"], "w") as f:
                meta_to_save = {
                    k: v for k, v in sample.items()
                    if not k.startswith("_")
                }
                json.dump(meta_to_save, f, indent=2)

            corrected_count += 1
            status = "accepted" if response == "" else "corrected"
            print(f"  {GREEN}{status}{RESET}\n")
            break

    print(f"\n  Done! {corrected_count} samples corrected this session.")
    show_stats()


def main():
    parser = argparse.ArgumentParser(
        description="Review and correct Whisper transcriptions for fine-tuning.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Review all samples, including previously corrected ones",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show training data collection stats",
    )
    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    review_samples(include_corrected=args.all)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
finetune-whisper.py -- Fine-tune Whisper on your corrected transcriptions.

Uses LoRA (PEFT) for efficient fine-tuning on a small dataset, then
converts the result to CTranslate2 format for faster-whisper.

Prerequisites:
  pip install peft datasets

Usage:
  python finetune-whisper.py                  # Train + convert
  python finetune-whisper.py --epochs 5       # More training epochs
  python finetune-whisper.py --convert-only   # Just convert existing checkpoint
  python finetune-whisper.py --dry-run        # Show stats, don't train

Output:
  ~/.dictation-models/whisper-finetuned/      # CTranslate2 model for faster-whisper

Then set WHISPER_MODEL=~/.dictation-models/whisper-finetuned in your env,
or the system will auto-detect it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────

TRAINING_DIR = Path(os.environ.get(
    "DICTATION_TRAINING_DIR",
    os.path.expanduser("~/.dictation-training"),
))
SAMPLES_DIR = TRAINING_DIR / "samples"

MODELS_DIR = Path(os.environ.get(
    "DICTATION_MODELS_DIR",
    os.path.expanduser("~/.dictation-models"),
))
HF_CHECKPOINT_DIR = MODELS_DIR / "whisper-lora-merged"
CT2_OUTPUT_DIR = MODELS_DIR / "whisper-finetuned"

BASE_MODEL = "openai/whisper-large-v2"

# ── Colors ───────────────────────────────────────────────────────────

BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"


def info(msg: str) -> None:
    print(f"  {GREEN}[+]{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}[!]{RESET} {msg}")


def error(msg: str) -> None:
    print(f"  {RED}[ERROR]{RESET} {msg}", file=sys.stderr)


# ── Data loading ─────────────────────────────────────────────────────

def load_corrected_samples() -> list[dict]:
    """Load all corrected training samples."""
    if not SAMPLES_DIR.exists():
        return []

    samples = []
    for meta_path in sorted(SAMPLES_DIR.glob("*.json")):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if not meta.get("corrected"):
                continue
            wav_path = SAMPLES_DIR / f"{meta['id']}.wav"
            if not wav_path.exists():
                continue
            meta["_wav_path"] = str(wav_path)
            samples.append(meta)
        except Exception:
            continue

    return samples


def show_stats(samples: list[dict]) -> None:
    """Show training data stats."""
    total_in_dir = len(list(SAMPLES_DIR.glob("*.wav"))) if SAMPLES_DIR.exists() else 0
    total_duration = sum(s.get("duration_s", 0) for s in samples)

    print(f"\n{BOLD}Training Data{RESET}")
    print(f"  Total samples:     {total_in_dir}")
    print(f"  Corrected:         {len(samples)}")
    print(f"  Audio duration:    {total_duration:.0f}s ({total_duration / 60:.1f}min)")

    if len(samples) < 50:
        warn(f"Need at least 50 corrected samples (have {len(samples)}).")
        warn("Run correct-transcripts.py to review and correct more.")
        return False
    else:
        info(f"Enough data for fine-tuning!")
        return True


# ── Fine-tuning ──────────────────────────────────────────────────────

def finetune(samples: list[dict], epochs: int = 3, lr: float = 1e-4) -> Path:
    """Fine-tune Whisper large-v2 with LoRA on corrected samples."""
    try:
        import torch
        from datasets import Audio, Dataset
        from transformers import (
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        error(f"Missing package: {e}")
        error("Install with: pip install peft datasets")
        sys.exit(1)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    info(f"Device: {device}")
    info(f"Base model: {BASE_MODEL}")
    info(f"Training samples: {len(samples)}")
    info(f"Epochs: {epochs}, LR: {lr}")

    # Load processor and model
    info("Loading Whisper processor...")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    tokenizer = processor.tokenizer
    feature_extractor = processor.feature_extractor

    info("Loading Whisper model...")
    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # required for gradient checkpointing

    # Apply LoRA
    info("Applying LoRA adapter...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    info(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Build dataset
    info("Building dataset...")
    data_dict = {
        "audio": [s["_wav_path"] for s in samples],
        "transcript": [
            s.get("corrected_transcript") or s["transcript"]
            for s in samples
        ],
    }
    dataset = Dataset.from_dict(data_dict)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Split: 90% train, 10% eval (minimum 1 eval sample)
    if len(samples) > 10:
        split = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    # Preprocessing function
    def prepare_examples(batch):
        # Extract audio features
        audio_arrays = [a["array"] for a in batch["audio"]]
        sampling_rates = [a["sampling_rate"] for a in batch["audio"]]

        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=sampling_rates[0],
            return_tensors="pt",
            padding=True,
        )

        # Tokenize transcripts
        labels = tokenizer(
            batch["transcript"],
            padding=True,
            return_tensors="pt",
        )

        # Replace padding token id with -100 so it's ignored in loss
        label_ids = labels.input_ids.clone()
        label_ids[label_ids == tokenizer.pad_token_id] = -100

        return {
            "input_features": inputs.input_features,
            "labels": label_ids,
        }

    info("Preprocessing audio...")
    train_dataset = train_dataset.map(
        prepare_examples,
        batched=True,
        batch_size=8,
        remove_columns=train_dataset.column_names,
    )
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            prepare_examples,
            batched=True,
            batch_size=8,
            remove_columns=eval_dataset.column_names,
        )

    # Training arguments
    output_dir = str(MODELS_DIR / "whisper-lora-training")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        warmup_steps=min(50, len(train_dataset) // 2),
        logging_steps=10,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        predict_with_generate=False,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
    )

    # Train
    info("Starting training...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()

    # Merge LoRA weights into base model
    info("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()

    # Save merged HuggingFace checkpoint
    HF_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(HF_CHECKPOINT_DIR))
    processor.save_pretrained(str(HF_CHECKPOINT_DIR))
    info(f"Saved merged checkpoint to {HF_CHECKPOINT_DIR}")

    # Cleanup training checkpoints
    training_output = Path(output_dir)
    if training_output.exists():
        shutil.rmtree(training_output)

    return HF_CHECKPOINT_DIR


# ── CTranslate2 conversion ──────────────────────────────────────────

def convert_to_ct2(hf_model_path: Path, quantization: str = "float16") -> Path:
    """Convert a HuggingFace Whisper model to CTranslate2 format."""
    import subprocess

    CT2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    info(f"Converting to CTranslate2 ({quantization})...")
    cmd = [
        sys.executable, "-m", "ctranslate2.converters.transformers",
        "--model", str(hf_model_path),
        "--output_dir", str(CT2_OUTPUT_DIR),
        "--quantization", quantization,
        "--copy_files", "tokenizer.json", "preprocessor_config.json",
        "--force",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Fallback: try the CLI converter
        cmd_cli = [
            "ct2-transformers-converter",
            "--model", str(hf_model_path),
            "--output_dir", str(CT2_OUTPUT_DIR),
            "--quantization", quantization,
            "--copy_files", "tokenizer.json", "preprocessor_config.json",
            "--force",
        ]
        result = subprocess.run(cmd_cli, capture_output=True, text=True)
        if result.returncode != 0:
            error(f"CTranslate2 conversion failed:\n{result.stderr}")
            sys.exit(1)

    info(f"CTranslate2 model saved to {CT2_OUTPUT_DIR}")
    return CT2_OUTPUT_DIR


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper on your corrected transcriptions.",
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--quantization", default="float16",
        choices=["float16", "int8_float16", "float32"],
        help="CTranslate2 quantization (default: float16)",
    )
    parser.add_argument(
        "--convert-only", action="store_true",
        help="Skip training, just convert existing checkpoint to CTranslate2",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show stats and exit without training",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}Whisper Fine-Tuning{RESET}\n")

    if args.convert_only:
        if not HF_CHECKPOINT_DIR.exists():
            error(f"No checkpoint found at {HF_CHECKPOINT_DIR}")
            error("Run training first (without --convert-only)")
            sys.exit(1)
        ct2_path = convert_to_ct2(HF_CHECKPOINT_DIR, args.quantization)
        print(f"\n  {GREEN}Done!{RESET} Model at: {ct2_path}")
        print(f"  Set WHISPER_MODEL={ct2_path} or it will be auto-detected.\n")
        return

    # Load and validate data
    samples = load_corrected_samples()
    has_enough = show_stats(samples)

    if args.dry_run:
        return

    if not has_enough:
        resp = input(f"\n  Continue anyway? [y/N]: ").strip().lower()
        if resp not in ("y", "yes"):
            return

    print()

    # Train
    hf_path = finetune(samples, epochs=args.epochs, lr=args.lr)

    # Convert
    ct2_path = convert_to_ct2(hf_path, args.quantization)

    print(f"\n  {GREEN}Fine-tuning complete!{RESET}")
    print(f"  Model: {ct2_path}")
    print(f"  Set WHISPER_MODEL={ct2_path} or it will be auto-detected.\n")


if __name__ == "__main__":
    main()

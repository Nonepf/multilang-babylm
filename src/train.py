"""
BabyLM 2026 — Training Entry Point.
One training run: create or load model, load data, train, save checkpoints.

Usage:
  # Phase 1 (train from scratch)
  python src/train.py --mode phase1 --l1 en --seed 42

  # Phase 2 (continue from Phase 1 checkpoint)
  python src/train.py --mode phase2 --l1 nl --l2 en --seed 42 \
      --from_checkpoint checkpoints/phase1/nl_s42/final

  # Probe batch size
  python src/train.py --probe
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from transformers import Trainer, TrainingArguments

from .config import (
    HPARAMS,
    MAX_SEQ_LENGTH,
    PHASE1_UNITS,
    PHASE2_UNITS,
    TOKENIZER_NAME,
    VOCAB_SIZE,
    NUM_CHECKPOINTS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_GRAD_ACCUM,
    get_native_units,
    compute_total_steps,
)
from .data import get_tokenized_dataset, get_eval_split
from .model import create_model, load_model, load_tokenizer, count_parameters, auto_batch_size


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_batch_config(batch_size: int, grad_accum: int) -> tuple:
    """If using defaults, probe GPU. Otherwise use provided values."""
    if batch_size == DEFAULT_BATCH_SIZE and grad_accum == DEFAULT_GRAD_ACCUM:
        print("\n[Batch] Default batch — running VRAM probe...")
        try:
            bs, ga = auto_batch_size()
            return bs, ga
        except Exception as e:
            print(f"[Batch] Probe failed ({e}), using defaults")
            return DEFAULT_BATCH_SIZE, DEFAULT_GRAD_ACCUM
    return batch_size, grad_accum


def main():
    parser = argparse.ArgumentParser(description="BabyLM 2026 Training")
    # Run mode
    parser.add_argument("--mode", choices=["phase1", "phase2"])
    parser.add_argument("--probe", action="store_true",
                        help="VRAM probe only, then exit")
    # Languages
    parser.add_argument("--l1", required=False, choices=["en", "nl", "zh"])
    parser.add_argument("--l2", default=None, choices=["en", "nl", "zh", None])
    # Data budget
    parser.add_argument("--effective_words", type=int, default=None,
                        help="Effective words per phase (default: 50M)")
    # Seed
    parser.add_argument("--seed", type=int, default=42)
    # Checkpoint for Phase 2
    parser.add_argument("--from_checkpoint", default=None)
    # Batch
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int, default=DEFAULT_GRAD_ACCUM)
    # Paths
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--cache_dir", default="./data_cache")
    parser.add_argument("--log_dir", default="./logs")
    # Overrides (for smoke tests / debugging)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--no_bf16", action="store_true")
    parser.add_argument("--save_steps_override", type=int, default=None)
    parser.add_argument("--max_steps_override", type=int, default=None)

    args = parser.parse_args()

    # ── Probe-only mode ──────────────────────────────────────────────
    if args.probe:
        bs, ga = auto_batch_size()
        cfg_dir = os.path.join(os.path.dirname(args.output_dir) or ".", "configs")
        os.makedirs(cfg_dir, exist_ok=True)
        cfg_path = os.path.join(cfg_dir, "batch_config.json")
        with open(cfg_path, "w") as f:
            json.dump({"batch_size": bs, "grad_accum": ga}, f, indent=2)
        print(f"Saved: {cfg_path}")
        return

    # ── Validate ─────────────────────────────────────────────────────
    if not args.mode:
        parser.error("--mode is required unless --probe is used")
    if not args.l1:
        parser.error("--l1 is required (unless --probe)")
    if args.mode == "phase2" and not args.from_checkpoint:
        parser.error("--from_checkpoint required for phase2")
    if args.mode == "phase2" and not args.l2:
        parser.error("--l2 required for phase2")

    effective_words = args.effective_words or (
        PHASE2_UNITS if args.mode == "phase2" else PHASE1_UNITS
    )

    set_seed(args.seed)

    # ── Batch config ─────────────────────────────────────────────────
    batch_size, grad_accum = resolve_batch_config(args.batch_size, args.grad_accum)
    eff_batch_tokens = batch_size * grad_accum * MAX_SEQ_LENGTH

    # ── Tokenizer & Model ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BabyLM 2026 — {args.mode.upper()}")
    print(f"  L1={args.l1}  L2={args.l2 or 'N/A'}  Seed={args.seed}")
    print(f"{'='*60}\n")

    print("Loading tokenizer...")
    tokenizer = load_tokenizer()

    if args.mode == "phase2" and args.from_checkpoint:
        print(f"Loading checkpoint: {args.from_checkpoint}")
        model = load_model(args.from_checkpoint)
    else:
        print("Creating fresh GPT-2 Small + Qwen2.5 vocab...")
        model = create_model()

    params = count_parameters(model)
    print(f"  Params: {params['total']:,} total ({params['total']/1e6:.1f}M)")
    print(f"  Batch: per_device={batch_size}, grad_accum={grad_accum} → "
          f"effective {batch_size * grad_accum} × {MAX_SEQ_LENGTH} = "
          f"{eff_batch_tokens:,} tok/step")

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {gpu} ({vram:.1f} GB)")

    # ── Data ─────────────────────────────────────────────────────────
    data_lang = args.l2 or args.l1
    native_units = get_native_units(data_lang, effective_words)
    print(f"\n  Data: {effective_words:,} effective words → "
          f"{native_units:,} native units ({data_lang})")

    train_ds = get_tokenized_dataset(data_lang, native_units, tokenizer, args.cache_dir)
    eval_ds = get_eval_split(train_ds)

    # ── Steps ────────────────────────────────────────────────────────
    total_steps = args.max_steps_override or compute_total_steps(
        data_lang, effective_words, eff_batch_tokens
    )
    save_steps = args.save_steps_override or max(1, total_steps // NUM_CHECKPOINTS)

    # ── Run name & output ────────────────────────────────────────────
    if args.mode == "phase1":
        run_name = f"{args.l1}_s{args.seed}"
    else:
        run_name = f"{args.l1}2{args.l2}_s{args.seed}"

    output_dir = os.path.join(args.output_dir, args.mode, run_name)
    os.makedirs(args.log_dir, exist_ok=True)

    print(f"\n  Run: {run_name}")
    print(f"  Steps: {total_steps} total, save every {save_steps}")
    print(f"  Chunks: {len(train_ds):,}")
    print(f"  Output: {output_dir}\n")

    # ── Training args ────────────────────────────────────────────────
    lr = args.learning_rate or HPARAMS["learning_rate"]
    warmup = args.warmup_steps or HPARAMS["warmup_steps"]
    wd = args.weight_decay or HPARAMS["weight_decay"]
    use_bf16 = not args.no_bf16 and torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=wd,
        warmup_steps=warmup,
        lr_scheduler_type=HPARAMS["lr_scheduler_type"],
        max_steps=total_steps,
        logging_steps=10,
        save_steps=save_steps,
        eval_steps=save_steps,
        save_total_limit=1,
        save_only_model=False,
        bf16=use_bf16,
        fp16=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        report_to="none",
        remove_unused_columns=True,
        save_strategy="steps",
        eval_strategy="steps",
        load_best_model_at_end=False,
        seed=args.seed,
        data_seed=args.seed,
        max_grad_norm=HPARAMS["max_grad_norm"],
        adam_beta1=HPARAMS["adam_beta1"],
        adam_beta2=HPARAMS["adam_beta2"],
        adam_epsilon=HPARAMS["adam_epsilon"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    # ── Train ────────────────────────────────────────────────────────
    print(f"{'='*60}")
    print(f"  TRAINING: {run_name}")
    print(f"  Steps: {total_steps} | Save: every {save_steps}")
    print(f"  Eff batch: {eff_batch_tokens:,} tok/step")
    print(f"{'='*60}\n")

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed/60:.1f} min ({elapsed/3600:.2f} h)")

    # ── Save final ───────────────────────────────────────────────────
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # ── Metadata ─────────────────────────────────────────────────────
    meta = {
        "run_name": run_name,
        "mode": args.mode,
        "seed": args.seed,
        "l1": args.l1,
        "l2": args.l2,
        "effective_words": effective_words,
        "native_units": native_units,
        "native_unit_type": "words" if data_lang != "zh" else "chars",
        "total_chunks": len(train_ds),
        "total_steps": total_steps,
        "save_steps": save_steps,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "effective_batch_tokens": eff_batch_tokens,
        "learning_rate": lr,
        "weight_decay": wd,
        "warmup_steps": warmup,
        "bf16": use_bf16,
        "tokenizer": TOKENIZER_NAME,
        "tokenizer_base_vocab_size": tokenizer.vocab_size,
        "tokenizer_length": len(tokenizer),
        "vocab_size": VOCAB_SIZE,
        "total_params": params["total"],
        "trainable_params": params["trainable"],
        "max_seq_length": MAX_SEQ_LENGTH,
        "from_checkpoint": args.from_checkpoint,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0,
        "elapsed_seconds": elapsed,
    }

    with open(os.path.join(final_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Done -> {final_dir}")
    print(f"  Metadata -> {os.path.join(final_dir, 'run_meta.json')}")


if __name__ == "__main__":
    main()

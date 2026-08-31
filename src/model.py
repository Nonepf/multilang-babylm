"""
BabyLM 2026 — Model Factory.
Create GPT-2 Small with Qwen2.5 vocabulary (~202M params),
or load from a checkpoint for Phase 2 continuation.
"""
import torch
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer

from .config import TOKENIZER_NAME, VOCAB_SIZE, MODEL_CONFIG


def create_model() -> GPT2LMHeadModel:
    """
    Create a fresh GPT-2 Small model with Qwen2.5 vocabulary.
    Returns ~202M parameter model (vs. standard 124M due to vocab expansion).
    """
    config = GPT2Config(
        vocab_size=VOCAB_SIZE,
        **MODEL_CONFIG,
    )
    model = GPT2LMHeadModel(config)
    return model


def load_model(checkpoint_path: str) -> GPT2LMHeadModel:
    """Load model from a checkpoint directory (final/ or checkpoint-N/)."""
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
    return model


def load_tokenizer() -> AutoTokenizer:
    """Load Qwen2.5 tokenizer and validate it against the model vocabulary."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    if len(tokenizer) > VOCAB_SIZE:
        raise ValueError(
            f"Tokenizer has {len(tokenizer):,} entries but model VOCAB_SIZE is "
            f"{VOCAB_SIZE:,}. Do not clamp token IDs; resize the model explicitly."
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def count_parameters(model: GPT2LMHeadModel) -> dict:
    """Return parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def auto_batch_size(
    target_effective: int = 128,
    seq_len: int = 512,
    candidates: list = None,
) -> tuple:
    """
    Probe GPU memory to find the largest safe per_device_batch_size.
    Returns (batch_size, gradient_accumulation_steps).

    Runs a forward+backward pass with increasing batch sizes until OOM.
    Falls back to conservative defaults if probing fails.
    """
    if candidates is None:
        candidates = [64, 48, 32, 24, 16, 12, 8, 4]

    if not torch.cuda.is_available():
        print("[auto_batch] No GPU detected, using defaults")
        return 4, 32

    device = torch.device("cuda")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[auto_batch] GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB)")

    model = create_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    input_ids = torch.randint(0, VOCAB_SIZE, (1, seq_len), device=device)

    best = 4
    for bs in candidates:
        try:
            batch_ids = input_ids.repeat(bs, 1)
            loss = model(input_ids=batch_ids, labels=batch_ids).loss
            loss.backward()
            optimizer.zero_grad()
            best = bs
            torch.cuda.empty_cache()
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            # Candidates are ordered largest to smallest. An OOM at a large
            # candidate should not prevent probing the remaining smaller ones.
            continue

    del model, optimizer
    torch.cuda.empty_cache()

    grad_accum = max(1, target_effective // best)
    actual_effective = best * grad_accum
    print(f"[auto_batch] Best per_device: {best}, grad_accum: {grad_accum} "
          f"(effective batch: {actual_effective} × {seq_len} = {actual_effective * seq_len:,} tok/step)")
    return best, grad_accum

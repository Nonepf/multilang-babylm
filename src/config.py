"""Central configuration for the BabyLM 2026 experiments.

These values describe the current reproduction code. Historical-run caveats,
including the 151,643/151,665 embedding-row mismatch, are documented in the
repository audit and README rather than hidden in this module.
"""

import os

# ── Project paths ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Tokenizer & Model ──────────────────────────────────────────────────
TOKENIZER_NAME = "Qwen/Qwen2.5-0.5B"
# ``tokenizer.vocab_size`` reports the 151,643 base vocabulary, while
# ``len(tokenizer)`` includes 22 added/special tokens. New reproductions size
# the embedding matrix to the full tokenizer length.
VOCAB_SIZE = 151_665
MAX_SEQ_LENGTH = 512

# GPT-2 Small architecture
MODEL_CONFIG = {
    "n_embd": 768,
    "n_layer": 12,
    "n_head": 12,
    "activation_function": "gelu_new",
    "n_positions": MAX_SEQ_LENGTH,
    "n_ctx": MAX_SEQ_LENGTH,
    "resid_pdrop": 0.1,
    "embd_pdrop": 0.1,
    "attn_pdrop": 0.1,
    "layer_norm_epsilon": 1e-5,
}

# ── Language definitions ───────────────────────────────────────────────
# Byte Premium values from BabyLM 2026 CFP (Arnett et al., 2024)
BYTE_PREMIUM = {"en": 1.0000, "nl": 1.0516, "zh": 0.9894}

# Estimated TWR (Token-to-Word/Char Ratio) for Qwen2.5 tokenizer
# Measured empirically in Phase 0 diagnostics
TWR_QWEN = {"en": 1.20, "nl": 1.70, "zh": 0.70}

LANG_CONFIG = {
    "en": {
        "dataset": "BabyLM-community/BabyLM-2026-Strict",
        "name": "English",
        "unit": "words",
        "byte_premium": 1.0000,
        "twr": 1.20,
    },
    "nl": {
        "dataset": "BabyLM-community/babylm-nld",
        "name": "Dutch",
        "unit": "words",
        "byte_premium": 1.0516,
        "twr": 1.70,
    },
    "zh": {
        "dataset": "BabyLM-community/babylm-zho",
        "name": "Chinese",
        "unit": "chars",
        "byte_premium": 0.9894,
        "twr": 0.70,
    },
}

# Language groups for lm-eval (babylm-eval multilingual track)
EVAL_LANG_GROUPS = {
    "en": "zeroshot_eng",
    "nl": "zeroshot_nld",
    "zh": "zeroshot_zho",
}


def get_native_units(lang: str, effective_words: int) -> int:
    """
    Convert effective words to native units using Byte Premium.
    English: 50M effective → 50M words (50M / 1.0000)
    Dutch:   50M effective → 47.5M words (50M / 1.0516)
    Chinese: 50M effective → 50.5M chars (50M / 0.9894)
    """
    bp = BYTE_PREMIUM[lang]
    return int(effective_words / bp)


def estimate_tokens(lang: str, native_units: int) -> int:
    """Estimate Qwen2.5 token count from native units using TWR."""
    twr = TWR_QWEN[lang]
    return int(native_units * twr)


def compute_total_steps(lang: str, effective_words: int, eff_batch_tokens: int) -> int:
    """
    Compute total training steps for a given language and effective word budget.

    Args:
        lang: "en" | "nl" | "zh"
        effective_words: Byte-Premium-adjusted word budget
        eff_batch_tokens: effective batch size in tokens (per_device * grad_accum * seq_len)

    Returns:
        Total training steps (rounded up)
    """
    native = get_native_units(lang, effective_words)
    tokens = estimate_tokens(lang, native)
    steps = int(tokens / eff_batch_tokens) + 1  # +1 to ensure full coverage
    return steps


# ── Per-phase effective word budgets ───────────────────────────────────
EFFECTIVE_WORDS_PER_PHASE = 50_000_000
PHASE1_UNITS = EFFECTIVE_WORDS_PER_PHASE
PHASE2_UNITS = EFFECTIVE_WORDS_PER_PHASE
TOTAL_EFFECTIVE_WORDS = 100_000_000     # cumulative after Phase 2

# ── Checkpoint intervals (in effective words) ──────────────────────────
# Phase 1 saves: 10M, 20M, 30M, 40M, 50M
# Phase 2 saves: 60M, 70M, 80M, 90M, 100M (cumulative)
NUM_CHECKPOINTS = 5

# ── Hyperparameters ────────────────────────────────────────────────────
HPARAMS = {
    "learning_rate": 5e-4,
    "lr_scheduler_type": "cosine",
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "max_grad_norm": 1.0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
}

# ── Batch configuration ────────────────────────────────────────────────
# Target effective batch: 128 × 512 = 65,536 tokens/step
TARGET_EFFECTIVE_BATCH = 128
SEQ_LEN_TOKENS = MAX_SEQ_LENGTH

# Default per_device_batch_size (overridden by auto_batch_size probe)
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM = 32

# ── Random seeds ───────────────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 1024]

# ── Experiment task definitions ────────────────────────────────────────
PHASE1_TASKS = [
    {"name": "zh", "l1": "zh", "l2": None},
    {"name": "en", "l1": "en", "l2": None},
    {"name": "nl", "l1": "nl", "l2": None},
]

PHASE2_TASKS = [
    {"name": "nlen", "l1": "nl", "l2": "en"},    # NL→EN (main test)
    {"name": "zhen", "l1": "zh", "l2": "en"},    # ZH→EN (main test)
    {"name": "ennl", "l1": "en", "l2": "nl"},    # EN→NL (auxiliary)
    {"name": "zhnl", "l1": "zh", "l2": "nl"},    # ZH→NL (auxiliary)
]

# ── Evaluation tasks per language ──────────────────────────────────────
# lm-eval task groups from babylm-eval/multilingual/tasks/
EVAL_TASKS = {
    "en": "zeroshot_eng",       # includes blimp_babylm_filtered
    "nl": "zeroshot_nld",       # includes blimp_nl
    "zh": "zeroshot_zho",       # includes zhoblimp
}

# For BLiMP-only fast evaluation
BLIMP_TASKS = {
    "en": "blimp_babylm_filtered",
    "nl": "blimp_nl",
    "zh": "zhoblimp",
}

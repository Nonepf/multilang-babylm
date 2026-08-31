#!/bin/bash
# BabyLM 2026 — Environment Setup.
# Usage: bash scripts/setup.sh
#
# What this does:
#   1. Install Python dependencies
#   2. Clone babylm-eval (if not present)
#   3. Pre-cache all training data
#   4. Probe optimal batch size for this GPU
set -e

source "$(cd "$(dirname "$0")" && pwd)/common.sh"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "  BabyLM 2026 — SETUP"
echo "  $(date)"
echo "  Project root: $PROJECT_ROOT"
echo "============================================================"
echo ""

# ── Python dependencies ──────────────────────────────────────────
echo ">>> Installing Python dependencies..."
pip_install -r requirements.txt
echo ""

# ── babylm-eval ──────────────────────────────────────────────────
if [ -d "babylm-eval/.git" ]; then
    echo ">>> babylm-eval already exists, updating..."
    cd babylm-eval
    git pull 2>/dev/null || echo "    (pull failed, using existing version)"
    cd "$PROJECT_ROOT"
else
    echo ">>> Cloning babylm-eval (with mirror fallback)..."
    git_clone_mirror https://github.com/babylm-org/babylm-eval.git babylm-eval
fi
echo ""

# ── Pre-cache data ───────────────────────────────────────────────
echo ">>> Pre-caching training data (this may take a while)..."
echo "    Data will be cached to: $PROJECT_ROOT/data_cache/"
python -c "
from src.data import get_tokenized_dataset
from src.model import load_tokenizer
from src.config import get_native_units, PHASE1_UNITS

tokenizer = load_tokenizer()
for lang in ['en', 'nl', 'zh']:
    units = get_native_units(lang, PHASE1_UNITS)
    print(f'\n--- {lang}: {units:,} native units ---')
    ds = get_tokenized_dataset(lang, units, tokenizer, 'data_cache')
    print(f'  {lang}: {len(ds)} chunks cached')
print('\nAll data cached.')
"
echo ""

# ── Batch size probe ─────────────────────────────────────────────
echo ">>> Probing optimal batch size for this GPU..."
python -m src.train --probe
echo ""

echo "============================================================"
echo "  SETUP COMPLETE"
echo "  $(date)"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  bash scripts/train_phase1.sh    # Train 15 monolingual models"
echo "  bash scripts/train_phase2.sh    # Train 20 transfer models"
echo "  bash scripts/evaluate.sh        # Evaluate all checkpoints"
echo "  bash scripts/analyze.sh         # Run statistical analysis"

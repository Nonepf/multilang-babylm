#!/bin/bash
# BabyLM 2026 — Train a single run.
# Usage: bash scripts/train_single.sh <phase1|phase2> <task_name> <seed> [extra_args]
#
# Examples:
#   bash scripts/train_single.sh phase1 en 42
#   bash scripts/train_single.sh phase2 nlen 42
#   bash scripts/train_single.sh phase1 en 42 --effective_words 1000000 --max_steps_override 50
set -e

source "$(cd "$(dirname "$0")" && pwd)/common.sh"
cd "$PROJECT_ROOT"

PHASE="$1"
TASK="$2"
SEED="$3"
shift 3 || true
EXTRA_ARGS="$@"

if [ -z "$PHASE" ] || [ -z "$TASK" ] || [ -z "$SEED" ]; then
    echo "Usage: bash scripts/train_single.sh <phase1|phase2> <task> <seed> [extra_args]"
    echo ""
    echo "Phase 1 tasks: zh en nl"
    echo "Phase 2 tasks: nlen zhen ennl zhnl"
    echo "Seeds: 42 123 456 789 1024"
    exit 1
fi

# ── Resolve task config ───────────────────────────────────────────
case "$TASK" in
    zh)   L1="zh"; L2="" ;;
    en)   L1="en"; L2="" ;;
    nl)   L1="nl"; L2="" ;;
    nlen) L1="nl"; L2="en" ;;
    zhen) L1="zh"; L2="en" ;;
    ennl) L1="en"; L2="nl" ;;
    zhnl) L1="zh"; L2="nl" ;;
    *)
        echo "Unknown task: $TASK"
        echo "Phase 1: zh en nl"
        echo "Phase 2: nlen zhen ennl zhnl"
        exit 1
        ;;
esac

# ── Check if already done ─────────────────────────────────────────
RUN_NAME="${TASK}_s${SEED}"
FINAL_DIR="$PROJECT_ROOT/checkpoints/$PHASE/$RUN_NAME/final"
if [ -f "$FINAL_DIR/run_meta.json" ]; then
    echo "[$(date '+%H:%M:%S')] $PHASE/$RUN_NAME — already complete, skipping"
    exit 0
fi

# ── Phase 2: verify Phase 1 checkpoint ────────────────────────────
FROM_ARG=""
if [ "$PHASE" = "phase2" ]; then
    L1_CHECKPOINT="$PROJECT_ROOT/checkpoints/phase1/${L1}_s${SEED}/final"
    if [ ! -d "$L1_CHECKPOINT" ]; then
        echo "ERROR: Phase 1 checkpoint not found: $L1_CHECKPOINT"
        echo "Run train_phase1.sh first."
        exit 1
    fi
    FROM_ARG="--from_checkpoint $L1_CHECKPOINT"
fi

# ── Batch config ──────────────────────────────────────────────────
BATCH_CONFIG="$PROJECT_ROOT/configs/batch_config.json"
if [ -f "$BATCH_CONFIG" ]; then
    BATCH_SIZE=$(python -c "import json; print(json.load(open('$BATCH_CONFIG'))['batch_size'])")
    GRAD_ACCUM=$(python -c "import json; print(json.load(open('$BATCH_CONFIG'))['grad_accum'])")
else
    BATCH_SIZE=""
    GRAD_ACCUM=""
fi

BATCH_ARG=""
if [ -n "$BATCH_SIZE" ]; then
    BATCH_ARG="--batch_size $BATCH_SIZE --grad_accum $GRAD_ACCUM"
fi

# ── Run training ──────────────────────────────────────────────────
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

echo "[$(date '+%H:%M:%S')] START $PHASE/$RUN_NAME  (L1=$L1  L2=${L2:-N/A}  seed=$SEED)"
echo "  Log: $LOG_DIR/${PHASE}_${RUN_NAME}.log"

L2_ARG=""
[ -n "$L2" ] && L2_ARG="--l2 $L2"

python -u -m src.train \
    --mode "$PHASE" \
    --l1 "$L1" \
    $L2_ARG \
    --seed "$SEED" \
    $FROM_ARG \
    $BATCH_ARG \
    $EXTRA_ARGS \
    2>&1 | tee "$LOG_DIR/${PHASE}_${RUN_NAME}.log"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] DONE  $PHASE/$RUN_NAME"
else
    echo "[$(date '+%H:%M:%S')] FAIL  $PHASE/$RUN_NAME (exit=$EXIT_CODE)"
    exit $EXIT_CODE
fi

#!/bin/bash
# BabyLM 2026 — Train all Phase 2 transfer models.
# 4 directions × 5 seeds = 20 runs.
# Requires Phase 1 checkpoints to exist.
# Usage: bash scripts/train_phase2.sh [--seed N]  # single seed for debugging
set -e

source "$(cd "$(dirname "$0")" && pwd)/common.sh"
cd "$PROJECT_ROOT"

SEEDS=(42 123 456 789 1024)
TASKS="nlen zhen ennl zhnl"

# Allow single-seed override
if [ "$1" = "--seed" ] && [ -n "$2" ]; then
    SEEDS=("$2")
    shift 2
fi

echo "============================================================"
echo "  PHASE 2 TRAINING"
echo "  $(date)"
echo "  Tasks: $TASKS"
echo "  Seeds: ${SEEDS[*]}"
echo "============================================================"
echo ""

# Verify Phase 1 checkpoints exist
MISSING=""
for seed in "${SEEDS[@]}"; do
    for lang in zh en nl; do
        CKPT="$PROJECT_ROOT/checkpoints/phase1/${lang}_s${seed}/final"
        if [ ! -d "$CKPT" ]; then
            MISSING="$MISSING $CKPT"
        fi
    done
done

if [ -n "$MISSING" ]; then
    echo "ERROR: Missing Phase 1 checkpoints:"
    for m in $MISSING; do echo "  $m"; done
    echo "Run train_phase1.sh first."
    exit 1
fi

TOTAL=$(( ${#TASKS} * ${#SEEDS[@]} ))
SUCCESS=0
FAILED=0
FAILED_LIST=""

for task in $TASKS; do
    for seed in "${SEEDS[@]}"; do
        echo ""
        if bash "$SCRIPT_DIR/train_single.sh" phase2 "$task" "$seed"; then
            SUCCESS=$((SUCCESS + 1))
        else
            FAILED=$((FAILED + 1))
            FAILED_LIST="$FAILED_LIST phase2/$task/s$seed"
        fi
        echo "  Progress: $((SUCCESS + FAILED))/$TOTAL (${SUCCESS} ok, ${FAILED} failed)"
    done
done

echo ""
echo "============================================================"
echo "  PHASE 2 SUMMARY"
echo "  $(date)"
echo "  Total: $TOTAL | Success: $SUCCESS | Failed: $FAILED"
echo "============================================================"

if [ $FAILED -gt 0 ]; then
    echo "Failed runs:"
    for f in $FAILED_LIST; do echo "  $f"; done
    exit 1
fi

echo ""
echo "All training complete. Next: bash scripts/evaluate.sh"

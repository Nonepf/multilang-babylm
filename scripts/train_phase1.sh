#!/bin/bash
# BabyLM 2026 — Train all Phase 1 monolingual models.
# 3 languages × 5 seeds = 15 runs.
# Usage: bash scripts/train_phase1.sh [--seed N]  # single seed for debugging
set -e

source "$(cd "$(dirname "$0")" && pwd)/common.sh"
cd "$PROJECT_ROOT"

SEEDS=(42 123 456 789 1024)
TASKS="zh en nl"

# Allow single-seed override
if [ "$1" = "--seed" ] && [ -n "$2" ]; then
    SEEDS=("$2")
    shift 2
fi

echo "============================================================"
echo "  PHASE 1 TRAINING"
echo "  $(date)"
echo "  Tasks: $TASKS"
echo "  Seeds: ${SEEDS[*]}"
echo "============================================================"
echo ""

TOTAL=$(( ${#TASKS} * ${#SEEDS[@]} ))
SUCCESS=0
FAILED=0
FAILED_LIST=""

for task in $TASKS; do
    for seed in "${SEEDS[@]}"; do
        echo ""
        if bash "$SCRIPT_DIR/train_single.sh" phase1 "$task" "$seed"; then
            SUCCESS=$((SUCCESS + 1))
        else
            FAILED=$((FAILED + 1))
            FAILED_LIST="$FAILED_LIST phase1/$task/s$seed"
        fi
        echo "  Progress: $((SUCCESS + FAILED))/$TOTAL (${SUCCESS} ok, ${FAILED} failed)"
    done
done

echo ""
echo "============================================================"
echo "  PHASE 1 SUMMARY"
echo "  $(date)"
echo "  Total: $TOTAL | Success: $SUCCESS | Failed: $FAILED"
echo "============================================================"

if [ $FAILED -gt 0 ]; then
    echo "Failed runs:"
    for f in $FAILED_LIST; do echo "  $f"; done
    exit 1
fi

echo ""
echo "Phase 1 complete. Next: bash scripts/train_phase2.sh"

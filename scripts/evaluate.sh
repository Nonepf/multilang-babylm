#!/bin/bash
# BabyLM 2026 — Evaluate all final checkpoints.
# Usage: bash scripts/evaluate.sh [--blimp_only] [--langs eng,nld,zho] [--all_checkpoints]
#
# Finds all final/ checkpoints and runs lm-eval on each.
# Phase 1 models: evaluate on their own language + zero-shot on others
# Phase 2 models: evaluate on L2 (main metric)
set -e

source "$(cd "$(dirname "$0")" && pwd)/common.sh"
cd "$PROJECT_ROOT"

BLIMP_ONLY=""
LANGS=""
ALL_CHECKPOINTS=0

while [ $# -gt 0 ]; do
    case "$1" in
        --blimp_only) BLIMP_ONLY="--blimp_only"; shift ;;
        --langs) LANGS="--langs $2"; shift 2 ;;
        --all_checkpoints) ALL_CHECKPOINTS=1; shift ;;
        *) shift ;;
    esac
done

echo "============================================================"
echo "  EVALUATION"
echo "  $(date)"
echo "============================================================"
echo ""

# ── Phase 1 models ───────────────────────────────────────────────
echo ">>> Phase 1 models..."
PHASE1_DIR="$PROJECT_ROOT/checkpoints/phase1"
if [ -d "$PHASE1_DIR" ]; then
    for run_dir in "$PHASE1_DIR"/*/; do
        [ -d "$run_dir" ] || continue
        RUN_NAME=$(basename "$run_dir")

        ckpt_dirs=("$run_dir"final)
        [ "$ALL_CHECKPOINTS" -eq 1 ] && ckpt_dirs+=("$run_dir"checkpoint-*)
        for ckpt_dir in "${ckpt_dirs[@]}"; do
            [ -d "$ckpt_dir" ] || continue
            CKPT_NAME=$(basename "$ckpt_dir")

            # Determine which languages to evaluate
            # Phase 1 models: evaluate own language + cross-lingual zero-shot
            if [[ "$RUN_NAME" == en_* ]]; then
                EVAL_LANGS="eng,nld,zho"
            elif [[ "$RUN_NAME" == nl_* ]]; then
                EVAL_LANGS="eng,nld,zho"
            elif [[ "$RUN_NAME" == zh_* ]]; then
                EVAL_LANGS="eng,nld,zho"
            else
                EVAL_LANGS="eng,nld,zho"
            fi

            [ -n "$LANGS" ] && EVAL_LANGS="${LANGS#--langs }"

            echo ""
            echo "  [$RUN_NAME/$CKPT_NAME] langs=$EVAL_LANGS"

            python -m src.evaluate \
                --model_path "$ckpt_dir" \
                --langs "$EVAL_LANGS" \
                $BLIMP_ONLY \
                || echo "  [$RUN_NAME/$CKPT_NAME] EVAL FAILED (continuing)"
        done
    done
else
    echo "  No phase1 checkpoints found."
fi

# ── Phase 2 models ───────────────────────────────────────────────
echo ""
echo ">>> Phase 2 models..."
PHASE2_DIR="$PROJECT_ROOT/checkpoints/phase2"
if [ -d "$PHASE2_DIR" ]; then
    for run_dir in "$PHASE2_DIR"/*/; do
        [ -d "$run_dir" ] || continue
        RUN_NAME=$(basename "$run_dir")

        ckpt_dirs=("$run_dir"final)
        [ "$ALL_CHECKPOINTS" -eq 1 ] && ckpt_dirs+=("$run_dir"checkpoint-*)
        for ckpt_dir in "${ckpt_dirs[@]}"; do
            [ -d "$ckpt_dir" ] || continue
            CKPT_NAME=$(basename "$ckpt_dir")

            # Phase 2: evaluate target (L2) language primarily
            if [[ "$RUN_NAME" == *en ]]; then
                EVAL_LANGS="eng"
            elif [[ "$RUN_NAME" == *nl ]]; then
                EVAL_LANGS="nld"
            else
                EVAL_LANGS="eng,nld,zho"
            fi

            [ -n "$LANGS" ] && EVAL_LANGS="${LANGS#--langs }"

            echo ""
            echo "  [$RUN_NAME/$CKPT_NAME] langs=$EVAL_LANGS"

            python -m src.evaluate \
                --model_path "$ckpt_dir" \
                --langs "$EVAL_LANGS" \
                $BLIMP_ONLY \
                || echo "  [$RUN_NAME/$CKPT_NAME] EVAL FAILED (continuing)"
        done
    done
else
    echo "  No phase2 checkpoints found."
fi

echo ""
echo "============================================================"
echo "  EVALUATION COMPLETE"
echo "  $(date)"
echo "============================================================"
echo ""
echo "Next: bash scripts/analyze.sh"

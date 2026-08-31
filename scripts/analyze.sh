#!/bin/bash
# Recompute the camera-ready TI/TB arithmetic from the reported score snapshot.
# Usage: bash scripts/analyze.sh
set -e

source "$(cd "$(dirname "$0")" && pwd)/common.sh"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "  ANALYSIS"
echo "  $(date)"
echo "============================================================"
echo ""

python -m src.analyze \
    --scores "$PROJECT_ROOT/results/paper_blimp_scores.csv" \
    --output "$PROJECT_ROOT/results/paper_analysis.json"

echo ""
echo "Done. See results/paper_analysis.json."

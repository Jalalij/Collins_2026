#!/usr/bin/env bash

# ============================================================
# SIMPLE MODELING STUDY EXAMPLE
# ============================================================

PROVIDER="oneweb"
EXP_TYPE="dl"
TARGETS="rtt jitter loss"
OUT_DIR="results/example_study"

echo ""
echo "============================================================"
echo " TRAINING"
echo " Provider:   $PROVIDER"
echo " Experiment: $EXP_TYPE"
echo " Targets:    $TARGETS"
echo "============================================================"

python3 run.py train \
  --provider "$PROVIDER" \
  --experiment_type "$EXP_TYPE" \
  --targets $TARGETS \
  --models seq2seq lstm transformer \
  --out_dir "$OUT_DIR"

echo ""
echo "============================================================"
echo " MODEL COMPARISON"
echo "============================================================"

python3 run.py compare \
  --run_dirs \
    "$OUT_DIR/seq2seq" \
    "$OUT_DIR/lstm" \
    "$OUT_DIR/transformer" \
  --out_dir "$OUT_DIR/comparison"

echo ""
echo "============================================================"
echo " STUDY COMPLETE"
echo "============================================================"
#!/usr/bin/env bash
# =============================================================================
# ablation_seq2seq.sh
#
# Runs leave-one-out feature ablation for the Seq2Seq-GRU model across all
# three providers in sequence. A failed provider will not stop the others.
#
# Targets: rtt, jitter, loss
# Strategy: leave_one_out (one variant per feature dropped + full baseline)
#
# Variant counts (= default features + 1 baseline):
#   Starlink  : 20 training jobs  (19 features)
#   Quectel   : 11 training jobs  (10 features)
#   OneWeb    : 10 training jobs   (9 features)
#
# Usage:
#   bash ablation_seq2seq.sh
#   bash ablation_seq2seq.sh --epochs 50   # override epochs for all runs
# =============================================================================
 
set -uo pipefail
# NOTE: -e is intentionally omitted so a failed provider does not abort the
# whole script. Each block captures its own exit code instead.
 
# ---------------------------------------------------------------------------
# Optional epoch override from first argument
# ---------------------------------------------------------------------------
EPOCHS=${1:-30}
 
# ---------------------------------------------------------------------------
# Helper — runs one ablation block and reports pass/fail without exiting
# ---------------------------------------------------------------------------
run_ablation() {
  local label="$1"
  local provider="$2"
  local exp_type="$3"
  local out_dir="$4"
 
  echo ""
  echo "============================================================"
  echo "  ABLATION: ${label}"
  echo "  provider=${provider}  experiment_type=${exp_type}"
  echo "  out_dir=${out_dir}"
  echo "============================================================"
 
  python evaluation/ablation.py \
    --model            seq2seq \
    --provider         "${provider}" \
    --experiment_type  "${exp_type}" \
    --targets          rtt jitter loss \
    --study            feature \
    --strategy         leave_one_out \
    --epochs           "${EPOCHS}" \
    --out_dir          "${out_dir}" \
    --skip_existing
  local exit_code=$?
 
  if [ "${exit_code}" -eq 0 ]; then
    echo "  [OK] ${label} completed successfully."
  else
    echo "  [WARN] ${label} exited with code ${exit_code} — continuing to next provider."
  fi
 
  return 0   # always return 0 so the caller keeps going
}
 
# ---------------------------------------------------------------------------
# Run all three providers
# ---------------------------------------------------------------------------
 
run_ablation \
  "Starlink  latency_loss_throughput_dl" \
  "starlink" \
  "latency_loss_throughput_dl" \
  "ablation_results/starlink_seq2seq"
 
run_ablation \
  "Quectel   latency_loss_throughput_dl" \
  "quectel" \
  "latency_loss_throughput_dl" \
  "ablation_results/quectel_seq2seq"
 
run_ablation \
  "OneWeb    dl" \
  "oneweb" \
  "dl" \
  "ablation_results/oneweb_seq2seq"
 
# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " ALL ABLATION RUNS COMPLETE"
echo " Results:"
echo "   ablation_results/starlink_seq2seq/ablation_results.csv"
echo "   ablation_results/starlink_seq2seq/ablation_summary.png"
echo "   ablation_results/quectel_seq2seq/ablation_results.csv"
echo "   ablation_results/quectel_seq2seq/ablation_summary.png"
echo "   ablation_results/oneweb_seq2seq/ablation_results.csv"
echo "   ablation_results/oneweb_seq2seq/ablation_summary.png"
echo "============================================================"
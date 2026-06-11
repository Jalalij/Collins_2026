#!/usr/bin/env bash

TARGETS="rtt jitter loss"

run_train() {
  local provider="$1"
  local exp_type="$2"
  local out_dir="$3"
  echo ""
  echo "--- ${provider}  ${exp_type} ---"
  python run.py train \
    --provider "$provider" \
    --experiment_type "$exp_type" \
    --targets $TARGETS \
    --models all \
    --out_dir "$out_dir"
  local code=$?
  echo "  [exit $code] ${provider} ${exp_type}"
  sleep 10
}

# =============================================================================
# PHASE 1 — DOWNLINK
# =============================================================================
echo ""
echo "============================================================"
echo " PHASE 1: DOWNLINK (DL)"
echo "============================================================"

# run_train starlink latency_loss_throughput_dl results/starlink_llt_dl
# run_train quectel  latency_loss_throughput_dl results/quectel_llt_dl
run_train oneweb   dl                         results/oneweb_llt_dl

# =============================================================================
# PHASE 2 — UPLINK
# =============================================================================
echo ""
echo "============================================================"
echo " PHASE 2: UPLINK (UL)"
echo "============================================================"

run_train starlink latency_loss_throughput_ul results/starlink_llt_ul
run_train quectel  latency_loss_throughput_ul results/quectel_llt_ul
run_train oneweb   ul                         results/oneweb_llt_ul

# =============================================================================
# PHASE 3 — COMPARISONS
# =============================================================================
echo ""
echo "============================================================"
echo " PHASE 3: CROSS-MODEL COMPARISONS"
echo "============================================================"

compare_trio() {
  local base_dir="$1"
  echo ""
  echo "  Comparing: ${base_dir}"
  python run.py compare \
    --run_dirs \
      "${base_dir}/seq2seq" \
      "${base_dir}/lstm" \
      "${base_dir}/transformer" \
    --out_dir "${base_dir}/comparison" || echo "  [compare failed for ${base_dir}]"
}

compare_trio results/starlink_llt_dl
compare_trio results/quectel_llt_dl
compare_trio results/oneweb_llt_dl
compare_trio results/starlink_llt_ul
compare_trio results/quectel_llt_ul
compare_trio results/oneweb_llt_ul

echo ""
echo "============================================================"
echo " ALL DONE"
echo "============================================================"



# # ── Starlink + Quectel ────────────────────────────────────────────────────
# for PROVIDER in starlink quectel; do
#   python run.py train \
#     --provider $PROVIDER \
#     --experiment_type latency \
#     --models all \
#     --out_dir results/${PROVIDER}_latency

#   python run.py train \
#     --provider $PROVIDER \
#     --experiment_type latency_loss_throughput_dl \
#     --models all \
#     --out_dir results/${PROVIDER}_llt_dl

#   python run.py train \
#     --provider $PROVIDER \
#     --experiment_type latency_loss_throughput_ul \
#     --models all \
#     --out_dir results/${PROVIDER}_llt_ul
# done

# # ── OneWeb (latency + all targets, both directions) ───────────────────────
# for DIRECTION in dl ul; do
#   python run.py train \
#     --provider oneweb \
#     --experiment_type $DIRECTION \
#     --targets rtt \
#     --models all \
#     --out_dir results/oneweb_${DIRECTION}_latency

#   python run.py train \
#     --provider oneweb \
#     --experiment_type $DIRECTION \
#     --models all \
#     --out_dir results/oneweb_${DIRECTION}_all_targets
# done

# # ── Cross-model comparison: starlink + quectel ───────────────────────────
# for PROVIDER in starlink quectel; do
#   python run.py compare \
#     --run_dirs \
#       results/${PROVIDER}_latency/seq2seq \
#       results/${PROVIDER}_latency/lstm \
#       results/${PROVIDER}_latency/transformer \
#     --out_dir results/${PROVIDER}_latency/comparison

#   for SUFFIX in llt_dl llt_ul; do
#     python run.py compare \
#       --run_dirs \
#         results/${PROVIDER}_${SUFFIX}/seq2seq \
#         results/${PROVIDER}_${SUFFIX}/lstm \
#         results/${PROVIDER}_${SUFFIX}/transformer \
#       --out_dir results/${PROVIDER}_${SUFFIX}/comparison
#   done
# done

# # ── Cross-model comparison: oneweb ───────────────────────────────────────
# for DIRECTION in dl ul; do
#   python run.py compare \
#     --run_dirs \
#       results/oneweb_${DIRECTION}_latency/seq2seq \
#       results/oneweb_${DIRECTION}_latency/lstm \
#       results/oneweb_${DIRECTION}_latency/transformer \
#     --out_dir results/oneweb_${DIRECTION}_latency/comparison

#   python run.py compare \
#     --run_dirs \
#       results/oneweb_${DIRECTION}_all_targets/seq2seq \
#       results/oneweb_${DIRECTION}_all_targets/lstm \
#       results/oneweb_${DIRECTION}_all_targets/transformer \
#     --out_dir results/oneweb_${DIRECTION}_all_targets/comparison
# done





# for PROVIDER in starlink oneweb quectel; do

#   python run.py train \
#     --provider $PROVIDER \
#     --experiment_type latency \
#     --models gaussian_process autoencoder \
#     --out_dir results/${PROVIDER}_latency

#   python run.py train \
#     --provider $PROVIDER \
#     --experiment_type latency_loss_throughput \
#     --models gaussian_process autoencoder \
#     --out_dir results/${PROVIDER}_llt

# done





# ── Train all models on DL-only experiment types ──────────────────────────

# for PROVIDER in starlink oneweb quectel; do

#   # latency
#   python run.py train \
#     --provider $PROVIDER \
#     --experiment_type latency \
#     --models all \
#     --out_dir results/${PROVIDER}_latency

#   # latency + loss + throughput
#   python run.py train \
#     --provider $PROVIDER \
#     --experiment_type latency_loss_throughput \
#     --models all \
#     --out_dir results/${PROVIDER}_llt

# done


# # ── Cross-model comparison: latency ──────────────────────────────────────

# for PROVIDER in starlink oneweb quectel; do
#   python run.py compare \
#     --run_dirs \
#       results/${PROVIDER}_latency/seq2seq \
#       results/${PROVIDER}_latency/lstm \
#       results/${PROVIDER}_latency/transformer \
#     --out_dir results/${PROVIDER}_latency/comparison
# done


# # ── Cross-model comparison: latency_loss_throughput ──────────────────────

# for PROVIDER in starlink oneweb quectel; do
#   python run.py compare \
#     --run_dirs \
#       results/${PROVIDER}_llt/seq2seq \
#       results/${PROVIDER}_llt/lstm \
#       results/${PROVIDER}_llt/transformer \
#     --out_dir results/${PROVIDER}_llt/comparison
# done


















# # ── Train all models on all providers / experiment types ──────────────────

# for PROVIDER in starlink oneweb quectel; do
#   for EXP in latency latency_loss_throughput dl ul; do
#     python run.py train \
#       --provider $PROVIDER \
#       --experiment_type $EXP \
#       --models all \
#       --out_dir results/${PROVIDER}_${EXP}
#   done
# done


# # ── Explicit runs ─────────────────────────────────────────────────────────

# python run.py train --provider starlink --experiment_type latency        --models all --out_dir results/starlink_latency
# python run.py train --provider starlink --experiment_type latency_loss_throughput --models all --out_dir results/starlink_llt
# python run.py train --provider starlink --experiment_type dl             --models all --out_dir results/starlink_dl
# python run.py train --provider starlink --experiment_type ul             --models all --out_dir results/starlink_ul

# python run.py train --provider oneweb   --experiment_type dl             --models all --out_dir results/oneweb_dl
# python run.py train --provider oneweb   --experiment_type ul             --models all --out_dir results/oneweb_ul

# python run.py train --provider quectel  --experiment_type latency        --models all --out_dir results/quectel_latency
# python run.py train --provider quectel  --experiment_type latency_loss_throughput --models all --out_dir results/quectel_llt


# # ── Cross-model comparison ───────────────────────────────────────────────

# python run.py compare \
#   --run_dirs results/starlink_dl/seq2seq results/starlink_dl/lstm results/starlink_dl/transformer \
#   --out_dir results/starlink_dl/comparison


# # ── Ablation ──────────────────────────────────────────────────────────────

# python evaluation/ablation.py \
#   --provider starlink --experiment_type dl \
#   --study feature --strategy leave_one_out \
#   --out_dir ablation_results/starlink_dl

# python evaluation/ablation.py \
#   --provider quectel --experiment_type latency_loss_throughput \
#   --study feature --strategy groups \
#   --out_dir ablation_results/quectel_llt

# python evaluation/ablation.py \
#   --provider oneweb --experiment_type dl \
#   --study architecture --arch_param hidden_size \
#   --out_dir ablation_results/oneweb_dl_arch
for PROVIDER in starlink oneweb quectel; do

  python run.py train \
    --provider $PROVIDER \
    --experiment_type latency \
    --models gaussian_process autoencoder \
    --out_dir results/${PROVIDER}_latency

  python run.py train \
    --provider $PROVIDER \
    --experiment_type latency_loss_throughput \
    --models gaussian_process autoencoder \
    --out_dir results/${PROVIDER}_llt

done


# # ── Train all models on DL-only experiment types ──────────────────────────

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
# ── Train all models on all providers / experiment types ──────────────────

for PROVIDER in starlink oneweb quectel; do
  for EXP in $(python run.py --list_features --provider $PROVIDER 2>/dev/null | \
               grep "Valid choices" | grep -oP '\w+'); do
    python run.py train \
      --provider $PROVIDER --experiment_type $EXP \
      --models all \
      --out_dir results/${PROVIDER}_${EXP}
  done
done

# Or explicitly:
python run.py train --provider starlink --experiment_type latency        --models all --out_dir results/starlink_latency
python run.py train --provider starlink --experiment_type latency_loss_throughput --models all --out_dir results/starlink_llt
python run.py train --provider starlink --experiment_type throughput     --models all --out_dir results/starlink_throughput
python run.py train --provider oneweb   --experiment_type dl             --models all --out_dir results/oneweb_dl
python run.py train --provider oneweb   --experiment_type ul             --models all --out_dir results/oneweb_ul
python run.py train --provider quectel  --experiment_type latency        --models all --out_dir results/quectel_latency
python run.py train --provider quectel  --experiment_type latency_loss_throughput --models all --out_dir results/quectel_llt
python run.py train --provider quectel  --experiment_type throughput     --models all --out_dir results/quectel_throughput

# ── Cross-model comparison within each provider/exp ───────────────────────
python run.py compare \
  --run_dirs results/starlink_throughput/seq2seq results/starlink_throughput/lstm \
             results/starlink_throughput/transformer \
  --out_dir results/starlink_throughput/comparison

# ── Ablation on each provider/exp ─────────────────────────────────────────
python evaluation/ablation.py \
  --provider starlink --experiment_type throughput \
  --study feature --strategy leave_one_out \
  --out_dir ablation_results/starlink_throughput

python evaluation/ablation.py \
  --provider quectel --experiment_type latency_loss_throughput \
  --study feature --strategy groups \
  --out_dir ablation_results/quectel_llt

python evaluation/ablation.py \
  --provider oneweb --experiment_type dl \
  --study architecture --arch_param hidden_size \
  --out_dir ablation_results/oneweb_dl_arch
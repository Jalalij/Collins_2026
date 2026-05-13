# Throughput Prediction Framework

Unified framework for Starlink throughput forecasting. One entry point (`run.py`) covers training, evaluation, cross-model comparison, and ablation studies. Supports predicting one or more output variables simultaneously.


# Throughput Seq2Seq GRU

Sequence-to-sequence model with GRU encoder/decoder and attention for
predicting Starlink throughput 15 seconds into the future from a 30-second
history window.

---

## Architecture

```
Input (B, 30, F)
    │
    ├── PeriodicEmbedding(hour_cst, dow_cst) -> (B, 30, 16)
    │   sin/cos encoding of hour (0-23) and day-of-week (0-6),
    │   projected to embed_dim=16 via a Linear layer
    │
    └── concat -> (B, 30, F+16)
              │
         Encoder: 2-layer GRU, hidden=128
              │
         enc_outputs (B, 30, 128)   <- all timestep outputs
         enc_hidden  (2, B, 128)    <- final hidden state
              │
         ┌────────────────────────────────────┐
         │  Decoder: 2-layer GRU, hidden=128  │  iterates output_len=15 times
         │  GRU input = concat(               │
         │      context  (128)  <- attention  │
         │      last_out (F+16) <- prev pred  │
         │  )                                 │
         │         ↑                          │
         │  Attention: 3x 1-layer MLP (128)   │
         │  query = decoder top hidden        │
         │  keys  = encoder outputs           │
         └────────────────────────────────────┘
              │
         Projection: Linear(128->128) + LeakyReLU + Linear(128->1)
              │
         Output (B, 15)  predicted throughput in Mbps
```


---

## File structure

```
├── run.py                  ← Main entry point (train / evaluate / compare / ablation)
├── dataset.py              Data loading, session splitting, normalisation, multi-target support
├── model.py                ThroughputSeq2Seq (GRU encoder + attention decoder, multi-target)
├── train.py                Training loop (AdamW, EMA val smoothing, early stopping)
├── inference.py            Predict on a single session CSV from a saved checkpoint
│
├── models/                 Model trainer registry
│   ├── base.py             BaseTrainer + ModelRegistry
│   ├── seq2seq.py          Seq2Seq GRU (primary model, multi-target aware)
│   ├── lstm_model.py       2-layer LSTM next-step forecaster
│   ├── transformer_model.py Encoder-only Transformer next-step forecaster
│   ├── gaussian_process_model.py  GP regressor (small datasets)
│   └── autoencoder_models.py      Predictive AE (standard | sparse | variational)
│
├── evaluation/
│   ├── metrics.py          All metric functions (MAE/RMSE/MAPE/R²/PICP/MIS + diagnostics)
│   ├── evaluate.py         evaluate_predictions() + compare_saved_runs()
│   └── ablation.py         Feature and architecture ablation studies
│
└── requirements.txt
```

---

## Quick start

```bash
pip install -r requirements.txt

# Train Seq2Seq (default) — DL throughput, 11 features:
python run.py train --mode dl --data_root data/starlink

# Train predicting two targets at once:
python run.py train --mode dl --data_root data/starlink \
    --targets dl rtt

# Train multiple models and compare:
python run.py train --mode dl --data_root data/starlink \
    --models seq2seq lstm transformer

# Train all models:
python run.py train --mode dl --data_root data/starlink --models all
```

---

## Subcommands

### `train`

```bash
python run.py train \
    --mode dl \
    --data_root data/starlink \
    --models seq2seq lstm \
    --targets dl rtt \
    --features default \
    --epochs 100 \
    --out_dir results/my_run
```

After training, each model writes to `<out_dir>/<model_key>/`:
- `metadata.json` — config + checkpoint path
- `metrics_summary.csv` — per-key + AGGREGATE metrics
- `diagnostics.csv` — residual diagnostics
- `plots/scatter.png`, `plots/horizon_metrics.png`

If >1 model is run, a cross-model `comparison.csv` and `comparison.png` are written to `<out_dir>/`.

### `evaluate`

Re-evaluate a saved checkpoint on the test split:

```bash
# Using metadata.json (reads targets/features/checkpoint automatically):
python run.py evaluate \
    --checkpoint results/my_run/seq2seq/metadata.json \
    --data_root data/starlink --mode dl \
    --out_dir eval_results/

# Using a raw .pt file (supply targets/features manually):
python run.py evaluate \
    --checkpoint checkpoints/.../best_model.pt \
    --data_root data/starlink --mode dl \
    --targets dl rtt \
    --features default \
    --out_dir eval_results/
```

### `compare`

Load `metrics_summary.csv` from any number of run directories and produce a side-by-side table:

```bash
python run.py compare \
    --run_dirs results/seq2seq results/lstm results/transformer \
    --out_dir results/comparison
```

Writes `comparison.csv` and `comparison.png`.

### `ablation`

```bash
# Leave-one-out feature ablation (Seq2Seq):
python run.py ablation --mode dl --data_root data/starlink \
    --study feature --strategy leave_one_out

# Drop semantic feature groups:
python run.py ablation --mode dl --data_root data/starlink \
    --study feature --strategy groups

# Add features one at a time:
python run.py ablation --mode dl --data_root data/starlink \
    --study feature --strategy add_one

# Custom explicit subsets:
python run.py ablation --mode dl --data_root data/starlink \
    --study feature --strategy custom \
    --feature_sets "dish_snr,distance_km" "dish_snr,distance_km,dish_pop_ping_ms"

# Architecture sweep (hidden_size | input_len | output_len | embed_dim):
python run.py ablation --mode dl --data_root data/starlink \
    --study architecture --arch_param hidden_size

# Resume without retraining completed variants:
python run.py ablation ... --skip_existing
```

Outputs: `ablation_results.csv`, `ablation_summary.png`, `ablation_config.json`.

Two study types:

--study feature — varies what goes into the model:
- leave_one_out — drops one feature at a time from the base set; shows which features matter most
- groups — drops entire semantic groups (network_perf, satellite_geometry, dish_state, weather, satellite_meta)
- add_one — starts with 1 feature and adds one at a time; shows diminishing returns curve
- custom — you supply exact subsets: --feature_sets "dish_snr,distance_km" "dish_snr,distance_km,dish_pop_ping_ms"

--study architecture — sweeps a single hyperparameter while holding everything else fixed:

Use --skip_existing to resume a run that was interrupted without retraining completed variants

---

## Inference on a single CSV

```bash
# From metadata.json (targets/features/checkpoint resolved automatically):
python inference.py \
    --checkpoint results/my_run/seq2seq/metadata.json \
    --csv data/starlink/starlink_20260510_030024/starlink_20260510_030024__udp_dl_400.csv \
    --mode dl \
    --output_csv predictions.csv

# Multi-target checkpoint:
python inference.py \
    --checkpoint results/multi/seq2seq/metadata.json \
    --csv data/starlink/.../...csv \
    --mode dl \
    --output_csv multi_preds.csv
```

Output columns: `t+1s … t+Ns` (single target) or `dl_t+1s …`, `rtt_ms_t+1s …` (multi-target).

---

## Discovery commands

```bash
python run.py --list_models     # all registered model keys + descriptions
python run.py --list_features   # all 24 features + default set + groups
python run.py --list_targets    # all available target columns + shorthand keys
```

---

## Feature selection

| Flag | Meaning |
|------|---------|
| `--features default` | 11-feature default set (default) |
| `--features all` | All 24 available features |
| `--features f1 f2 f3` | Explicit column names |
| `--feature_group network_perf` | One semantic group |
| `--feature_groups g1 g2` | Union of multiple groups |

Semantic groups: `network_perf`, `satellite_geometry`, `dish_state`, `weather`, `satellite_meta`.

```
iperf_jitter_ms            iperf_loss_pct
ping_tunnel_rtt_ms_mean    ping_internet_rtt_ms_mean
dish_pop_ping_ms           distance_km
sat_lat                    sat_lon
sat_alt_km                 dish_snr
dish_gps_sats              dish_boresight_azimuth_deg
dish_boresight_elevation_deg  dish_tilt_angle_deg
dish_obstruction_fraction  satellite_dtc
satellite_handoff          satellite_age_s
weather_temp_f             weather_humidity
weather_pressure           weather_wind_speed
weather_wind_dir           weather_rain_rate

python main.py --mode dl --data_root data/starlink \
    --features iperf_jitter_ms iperf_loss_pct ping_tunnel_rtt_ms_mean \
               ping_internet_rtt_ms_mean dish_pop_ping_ms distance_km \
               sat_lat sat_lon sat_alt_km dish_snr dish_gps_sats \
               dish_boresight_azimuth_deg dish_boresight_elevation_deg \
               dish_tilt_angle_deg dish_obstruction_fraction satellite_dtc \
               satellite_handoff satellite_age_s weather_temp_f \
               weather_humidity weather_pressure weather_wind_speed \
               weather_wind_dir weather_rain_rate
```

---

## Target selection

| Shorthand | Full column name |
|-----------|-----------------|
| `dl` | `dish_dl_throughput_mbps` |
| `ul` | `dish_ul_throughput_mbps` |
| `iperf` | `iperf_throughput_mbps` |
| `rtt` | `ping_internet_rtt_ms_mean` |
| `snr` | `dish_snr` |

Multi-target example: `--targets dl rtt` predicts both DL throughput and RTT for every horizon step. Results are keyed `dl_t+1s`, `rtt_ms_t+1s`, etc.

---

## Available models

| Key | Description |
|-----|-------------|
| `seq2seq` | GRU Seq2Seq + attention + periodic time embed **(primary; multi-target)** |
| `lstm` | 2-layer LSTM next-step forecaster |
| `transformer` | Encoder-only Transformer next-step forecaster |
| `gaussian_process` | GP regressor — small datasets only |
| `autoencoder` | Predictive AE (standard \| sparse \| variational) |

Legacy models (lstm, transformer, gaussian_process, autoencoder) predict only the **first horizon step** of the **primary target**. `seq2seq` predicts all horizon steps for all targets simultaneously.

---

## Evaluation metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| MAPE | Mean Absolute % Error (rows where \|target\| < 1 excluded) |
| sMAPE | Symmetric MAPE |
| R² | Coefficient of determination |
| PICP-95 | Empirical 95% prediction interval coverage |
| MIS | Mean Interval Score |
| relative_rmse | RMSE(model) / RMSE(persistence) — <1 beats naive baseline |
| relative_mae | MAE(model) / MAE(persistence) |
| resid_pearson_lag1 | Lag-1 residual autocorrelation |
| resid_ljung_p_lag10 | Ljung-Box p-value (lag 10) |

- MAE: Average absolute error in Mbps — most interpretable
- RMSE: Penalises large errors more heavily than MAE
- MAPE: % error relative to actual throughput (near-zero rows excluded via --mape_eps)
- sMAPE: Symmetric version of MAPE, bounded and more stableR²How much variance the model explains (1.0 = perfect)
- PICP-95: What fraction of actuals fall inside the 95% empirical prediction interval — should be ~0.95
- MIS: Mean Interval Score — penalises both wide intervals and coverage misses simultaneously

Output

- metrics_summary.csv — full per-horizon table for model + both baselines
- baseline_comparison.csv — aggregate model vs baselines side-by-side
- predictions.npz — raw arrays for any offline analysis
- horizon_metrics.png — MAE and RMSE curves across the 15-step horizon
- scatter.png — predicted vs actual scatter (all horizons)
- attention_mean.png — mean attention heatmap showing which past timesteps the decoder relies on
- sample_forecasts.png — 6 individual forecast windows vs ground truth


---

## Data layout

```
data/starlink/
    starlink_20260506_030006/
        starlink_20260506_030006__udp_dl_400.csv
        starlink_20260506_030006__udp_ul_40.csv
    starlink_20260507_030024/
        ...
```

Split is **session-level** — all windows from one session folder land in one split. Normalisation stats are computed on training sessions only and applied to val/test.

---

## Training flags

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `dl` | `dl` or `ul` |
| `--targets` | mode default | Target column(s) to predict |
| `--features` | default (11) | Feature selection |
| `--input_len` | 30 | Encoder window (seconds) |
| `--output_len` | 15 | Prediction horizon (seconds) |
| `--normalise` | `zscore` | `zscore`, `minmax`, or `none` |
| `--train_frac` | 0.70 | Session fraction for training |
| `--val_frac` | 0.15 | Session fraction for validation |
| `--epochs` | 100 | Max epochs |
| `--lr` | 0.001 | AdamW learning rate |
| `--hidden_size` | 128 | GRU hidden size |
| `--embed_dim` | 16 | Periodic time embedding dim |
| `--dropout` | 0.3 | Dropout |
| `--out_dir` | `results` | Output directory |

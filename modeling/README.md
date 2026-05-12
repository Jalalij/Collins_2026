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

## Data layout required

```
data/starlink/
    starlink_20260506_030006/
        starlink_20260506_030006__udp_dl_200.csv
        starlink_20260506_030006__udp_dl_400.csv   <- DL training uses this
        starlink_20260506_030006__udp_ul_20.csv
        starlink_20260506_030006__udp_ul_40.csv    <- UL training uses this
    starlink_20260507_030024/
        ...
```

One folder per collection session. The code picks exactly one file per
session based on the mode:

| `--mode` | File used | Target column |
|----------|-----------|---------------|
| `dl` | `*__udp_dl_400.csv` | `dish_dl_throughput_mbps` |
| `ul` | `*__udp_ul_40.csv` | `dish_ul_throughput_mbps` |

---

## File structure

```
throughput_seq2seq/
├── model.py        <- PeriodicEmbedding, Encoder, Attention, Decoder, Seq2Seq
├── dataset.py      <- file discovery, CSV loading, sliding windows, DataLoaders
├── train.py        <- training loop (AdamW, LR decay, teacher forcing, checkpointing)
├── inference.py    <- load checkpoint, predict on a single CSV file
├── main.py         <- CLI entry point for training
└── requirements.txt
```

---

## Quick start

```bash
pip install -r requirements.txt

# Train DL model (default 11 features, 70/15/15 session split)
python main.py --mode dl --data_root data/starlink

# Train UL model
python main.py --mode ul --data_root data/starlink

# Custom feature subset
python main.py --mode dl --data_root data/starlink \
    --features iperf_jitter_ms iperf_loss_pct dish_snr distance_km \
               ping_internet_rtt_ms_mean weather_temp_f

# Quick smoke-test (3 epochs, small batch)
python main.py --mode dl --data_root data/starlink --epochs 3 --batch_size 64

# DL
python main.py --mode dl --data_root ~/modeling_and_data_collection/modeling/data/starlink

# UL
python main.py --mode ul --data_root ~/modeling_and_data_collection/modeling/data/starlink
```

---

## Inference on a single session file

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --csv  data/starlink/starlink_20260510_030024/starlink_20260510_030024__udp_dl_400.csv \
    --mode dl \
    --output_csv predictions.csv
```

Output is a CSV with columns `t+1s` … `t+15s`, one row per sliding window.

---

## Training options

| Flag | Default | Description |
|------|---------|-------------|
| `--data_root` | `data/starlink` | Folder containing session subdirectories |
| `--mode` | `dl` | `dl` (400 Mbps files) or `ul` (40 Mbps files) |
| `--target` | same as mode | Override predicted column (`dl`/`ul`/`iperf`) |
| `--features` | 11-feature set | Space-separated feature names |
| `--input_len` | `30` | History window fed to encoder (seconds) |
| `--output_len` | `15` | Prediction horizon (seconds) |
| `--normalise` | `zscore` | `zscore`, `minmax`, or `none` |
| `--train_frac` | `0.70` | Fraction of sessions for training |
| `--val_frac` | `0.15` | Fraction of sessions for validation |
| `--batch_size` | `512` | — |
| `--epochs` | `50` | — |
| `--lr` | `0.001` | AdamW learning rate |
| `--decay_rate` | `0.99` | Multiplicative LR decay per batch step |
| `--save_dir` | `checkpoints` | Where to save best model + training log |

---

## Available features

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

## Train / val / test split

Split is performed at the **session (folder) level**, not the sample level.
This prevents leakage: all sliding windows from one session stay in one split.
With 23 sessions and 70/15/15: ~16 train / ~3 val / ~4 test sessions.

Normalisation statistics (mean/std or min/max) are computed **only** from
training sessions and applied consistently to val and test.

---

## Training notes

- Teacher forcing is annealed from 0.5 → 0 over the first half of training.
- Gradient clipping at norm=5.
- Best checkpoint (lowest val MSE) is saved to `checkpoints/best_model.pt`.
- A CSV training log is written to `checkpoints/training_log.csv`.


---

## Evaluation

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

## Ablation

```
python ablation.py --study architecture --arch_param hidden_size   # [64, 128, 256]
python ablation.py --study architecture --arch_param input_len     # [10, 20, 30, 60]
python ablation.py --study architecture --arch_param embed_dim     # [0, 8, 16, 32]
```

Two study types:

--study feature — varies what goes into the model:
- leave_one_out — drops one feature at a time from the base set; shows which features matter most
- groups — drops entire semantic groups (network_perf, satellite_geometry, dish_state, weather, satellite_meta)
- add_one — starts with 1 feature and adds one at a time; shows diminishing returns curve
- custom — you supply exact subsets: --feature_sets "dish_snr,distance_km" "dish_snr,distance_km,dish_pop_ping_ms"

--study architecture — sweeps a single hyperparameter while holding everything else fixed:

Use --skip_existing to resume a run that was interrupted without retraining completed variants
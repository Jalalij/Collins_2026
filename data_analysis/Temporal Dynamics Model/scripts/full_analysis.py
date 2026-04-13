#!/usr/bin/env python3
"""
Layer Sweep Analysis (UL + DL)

- Sweeps num_layers=1..5 for LSTM and Transformer
- Runs both UL and DL directions
- Uses the argparse config (config/args.py)
- Saves per-model/layer CSVs, a combined per-run CSV, and a tidy R² summary CSV
"""

import os
import json
import warnings
from datetime import datetime
from pathlib import Path
import time 

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.evaluation.standardized_evaluation import StandardizedModelEvaluator
from src.models import VARTrainer, LSTMTrainer, TransformerTrainer
from src.data import TARGET_VARIABLES as THE_TARGET_VARIABLES
from src.data.process_data import ExperimentDataset
from config import args  # reuse the argparse

# -----------------------------
# Helpers
# -----------------------------
def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in s)

def _get_run_tag() -> str:
    # Use --run_tag if present, else env RUN_TAG, else timestamp
    run_tag = getattr(args, "run_tag", None)
    if not run_tag:
        run_tag = os.environ.get("RUN_TAG")
    if not run_tag:
        run_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    return _safe_name(run_tag)

def load_and_split_experiments(csv_dir, direction, minutes, target_hz, window_size):
    direction = direction.upper()
    files = [
        os.path.join(csv_dir, f)
        for f in os.listdir(csv_dir)
        if f.endswith(f"_{direction}.csv")
    ]
    if not files:
        raise FileNotFoundError(f"No CSVs for {direction} in {csv_dir}")

    train_set = ExperimentDataset(
        files, mode="train", direction=direction,
        window_size=window_size, minutes=minutes, target_hz=target_hz
    )
    test_set = ExperimentDataset(
        files, mode="test", direction=direction,
        window_size=window_size, minutes=minutes, target_hz=target_hz
    )
    return train_set, test_set

def calculate_target_r2(y_true, y_pred):
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return -999.0
    y_true, y_pred = y_true[mask], y_pred[mask]
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res/ss_tot) if ss_tot > 0 else -999.0

def extract_r2_per_target(y_true_dict, y_pred_dict):
    out = {}
    for t in THE_TARGET_VARIABLES:
        if t in y_true_dict and t in y_pred_dict:
            out[t] = calculate_target_r2(y_true_dict[t], y_pred_dict[t])
        else:
            out[t] = -999.0
    return out

def _append_tidy_r2(summary_path: Path, rows: list[dict]):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df_rows = pd.DataFrame(rows)
    header = not summary_path.exists()
    df_rows.to_csv(summary_path, mode="a", header=header, index=False)

# -----------------------------
# Main Analysis
# -----------------------------
def run_layer_sweep():
    print("LAYER SWEEP ANALYSIS")
    print("=" * 60)
    print(f"Targets: {THE_TARGET_VARIABLES}")
    print(f"Started: {datetime.now()}")

    run_tag = _get_run_tag()
    print(f"Run tag: {run_tag}")

    for direction in ["UL", "DL"]:
        print(f"\n=== Running {direction} ===")

        train_data, test_data = load_and_split_experiments(
            args.data_dir, direction, args.minutes, args.target_hz, args.window_size
        )
        evaluator = StandardizedModelEvaluator()

        # Build sweep: LSTM 1..5, Transformer 1..5, plus VAR
        models_to_test = []
        # for n in range(1, 6):
        #     models_to_test.append((f"LSTM_layers{n}", LSTMTrainer, {"num_layers": n}))
        # for n in range(1, 6):
        #     models_to_test.append((f"Transformer_layers{n}", TransformerTrainer, {"num_layers": n}))
        models_to_test.append(("VAR", VARTrainer, {}))

        # Output dirs
        base_dir = Path(args.results_dir) / f"{direction}_hz{args.target_hz}"
        per_model_dir = base_dir / "per_model"
        per_model_dir.mkdir(parents=True, exist_ok=True)

        # Accumulators
        combined_frames = []
        tidy_r2_rows = []

        for i, (name, Trainer, cfg) in enumerate(models_to_test, 1):
            print(f"\n{i}️⃣ {name} [{direction}]")

            start_time = time.time()
            # Train + evaluate
            if name != "VAR":
                results, y_true, y_pred, training_metadata = Trainer.train_and_evaluate(
                    train_data, test_data, evaluator, "/tmp", config=cfg
                )

            else:
                results, y_true, y_pred, training_metadata = Trainer.train_and_evaluate(
                    train_data, test_data, evaluator, "/tmp"
                )

            elapsed = time.time() - start_time

            if results is None or not y_true or not y_pred:
                print(f"⚠️  Skipping {name}: no results.")
                continue

            # R2 per target
            r2_dict = extract_r2_per_target(y_true, y_pred)

            # Per-model dataframe
            df_model = pd.DataFrame(
                {f"{name}_true_{t}": y_true[t] for t in y_true}
                | {f"{name}_pred_{t}": y_pred[t] for t in y_pred}
            )
            combined_frames.append(df_model)

            # Save per-model CSV and metadata (unique filename)
            safe_name = _safe_name(name)
            model_csv = per_model_dir / f"{safe_name}_{run_tag}.csv"
            model_meta = per_model_dir / f"meta_{safe_name}_{run_tag}.json"

            df_model.to_csv(model_csv, index=False)
            with open(model_meta, "w") as f:
                json.dump(
                    {
                        "model": name,
                        "direction": direction,
                        "run_tag": run_tag,
                        "config": cfg,
                        "timestamp": datetime.now().isoformat(),
                        "r2": r2_dict,
                        "train_time_sec": training_metadata['training_time_sec'],    
                        "test_time_sec": training_metadata['testing_time_sec'],    
                    },
                    f,
                    indent=2,
                )
            print(f"   ↳ saved {model_csv.name}")

            # Tidy R² rows for summary
            for t, r2 in r2_dict.items():
                tidy_r2_rows.append(
                    {
                        "run_tag": run_tag,
                        "direction": direction,
                        "model": name,
                        "target": t,
                        "r2": r2,
                        "train_time_sec": training_metadata['training_time_sec'],    
                        "test_time_sec": training_metadata['testing_time_sec'],    
                    }
                )

        # Save combined per-run CSV (unique) and update rolling "latest"
        if combined_frames:
            combined_df = pd.concat(combined_frames, axis=1)

            unique_csv = base_dir / f"results_{run_tag}.csv"
            latest_csv = base_dir / "results.csv"  # rolling latest for existing tools

            combined_df.to_csv(unique_csv, index=False)
            combined_df.to_csv(latest_csv, index=False)
            print(f"\n✅ Saved run → {unique_csv.name} (and updated {latest_csv.name}).")

            # Append tidy R² summary
            r2_summary = base_dir / "r2_summary.csv"
            _append_tidy_r2(r2_summary, tidy_r2_rows)
            print(f"   ↳ appended R² rows → {r2_summary.name}")
        else:
            print("⚠️ No model outputs collected; nothing to save for this direction.")

if __name__ == "__main__":
    # Optional: allow --run_tag without changing the existing args module.
    # If you add it to config/args.py, this will still work (getattr in _get_run_tag).
    run_layer_sweep()
"""
Dataset utilities for throughput Seq2Seq model.

Data layout
-----------
<data_root>/                                   e.g.  data/starlink/
    starlink_20260506_030006/
        starlink_20260506_030006__udp_dl_200.csv
        starlink_20260506_030006__udp_dl_400.csv   <- used for DL training
        starlink_20260506_030006__udp_ul_20.csv
        starlink_20260506_030006__udp_ul_40.csv    <- used for UL training
    starlink_20260507_030024/
        ...

Mode-to-filename mapping (fixed):
    mode="dl"  ->  *__udp_dl_400.csv
    mode="ul"  ->  *__udp_ul_40.csv

Train / val / test split is done at the SESSION (folder) level so that
all windows from one session land in one split only - no leakage.
Normalisation stats are computed on training sessions only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset


# ---------------------------------------------------------------------------
# Filename suffix that identifies each mode
# ---------------------------------------------------------------------------
MODE_SUFFIX: dict[str, str] = {
    "dl": "__udp_dl_400.csv",
    "ul": "__udp_ul_40.csv",
}

# ---------------------------------------------------------------------------
# All feature columns available for selection
# ---------------------------------------------------------------------------
ALL_FEATURES: list[str] = [
    # iPerf / ping
    "iperf_jitter_ms",
    "iperf_loss_pct",
    "ping_tunnel_rtt_ms_mean",
    "ping_internet_rtt_ms_mean",
    "dish_pop_ping_ms",
    # Satellite geometry
    "distance_km",
    "sat_lat",
    "sat_lon",
    "sat_alt_km",
    # Dish state
    "dish_snr",
    "dish_gps_sats",
    "dish_boresight_azimuth_deg",
    "dish_boresight_elevation_deg",
    "dish_tilt_angle_deg",
    "dish_obstruction_fraction",
    # Satellite metadata
    "satellite_dtc",
    "satellite_handoff",
    "satellite_age_s",
    # Weather
    "weather_temp_f",
    "weather_humidity",
    "weather_pressure",
    "weather_wind_speed",
    "weather_wind_dir",
    "weather_rain_rate",
]

# Default 11-feature set (matches "11 variants" in the paper)
DEFAULT_FEATURES: list[str] = [
    "iperf_jitter_ms",
    "iperf_loss_pct",
    "ping_tunnel_rtt_ms_mean",
    "ping_internet_rtt_ms_mean",
    "dish_pop_ping_ms",
    "distance_km",
    "dish_snr",
    "dish_boresight_elevation_deg",
    "dish_obstruction_fraction",
    "weather_temp_f",
    "weather_wind_speed",
]

MANDATORY_COLS: list[str] = ["timestamp_cst", "hour_cst", "dow_cst"]

TARGET_COL: dict[str, str] = {
    "dl":    "dish_dl_throughput_mbps",
    "ul":    "dish_ul_throughput_mbps",
    "iperf": "iperf_throughput_mbps",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    # Root containing the starlink_YYYYMMDD_HHMMSS/ session folders
    data_root: str = "data/starlink"

    # "dl" -> loads *__udp_dl_400.csv per session
    # "ul" -> loads *__udp_ul_40.csv  per session
    mode: str = "dl"

    # Column to predict (almost always matches mode; kept separate for flexibility)
    target: str = "dl"   # "dl" | "ul" | "iperf"

    # Subset of ALL_FEATURES; None -> DEFAULT_FEATURES
    features: list[str] | None = None

    input_len:  int = 30
    output_len: int = 15

    # Session-level fractions; test = remainder
    train_frac: float = 0.70
    val_frac:   float = 0.15

    # "zscore" | "minmax" | None
    normalise: str = "zscore"

    batch_size:  int = 512
    num_workers: int = 0
    seed:        int = 42

    def __post_init__(self) -> None:
        assert self.mode in MODE_SUFFIX, \
            f"mode must be one of {list(MODE_SUFFIX)}, got '{self.mode}'"
        assert self.target in TARGET_COL, \
            f"target must be one of {list(TARGET_COL)}, got '{self.target}'"
        assert self.train_frac + self.val_frac < 1.0, \
            "train_frac + val_frac must be < 1 to leave room for a test set"
        if self.features is None:
            self.features = list(DEFAULT_FEATURES)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_session_files(data_root: str, mode: str) -> list[Path]:
    """
    Walk data_root for session subdirectories and return the one matching CSV
    per session for the given mode.
    """
    suffix = MODE_SUFFIX[mode]
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"data_root '{data_root}' does not exist.")

    found: list[Path] = []
    for session_dir in sorted(root.iterdir()):
        if not session_dir.is_dir():
            continue
        matches = [f for f in session_dir.iterdir() if f.name.endswith(suffix)]
        if len(matches) == 1:
            found.append(matches[0])
        elif len(matches) > 1:
            print(f"  [warn] multiple matches in {session_dir.name}, using: {matches[0].name}")
            found.append(matches[0])
        else:
            print(f"  [warn] no '*{suffix}' in {session_dir.name}, skipping")
    return found


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame | None:
    """Load CSV, dropping any repeated header rows embedded mid-file."""
    try:
        df = pd.read_csv(path, low_memory=False)
        df = df[df["timestamp_cst"] != "timestamp_cst"].copy()
        df["timestamp_cst"] = pd.to_datetime(df["timestamp_cst"])
        for col in df.columns:
            if col != "timestamp_cst":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as exc:
        print(f"  [warn] could not load {path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Sliding-window Dataset for a single session file
# ---------------------------------------------------------------------------

class ThroughputFileDataset(Dataset):
    """
    Extracts overlapping windows of (input_len + output_len) rows.

    Each item:
        features : (input_len, num_features)  float32
        hour     : (input_len,)               float32  raw 0-23
        dow      : (input_len,)               float32  raw 0-6
        target   : (output_len,)              float32  future throughput
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        input_len: int = 30,
        output_len: int = 15,
        feature_stats: dict | None = None,
        target_stats:  dict | None = None,
        normalise: str = "zscore",
    ) -> None:
        total = input_len + output_len

        df = df.sort_values("timestamp_cst").reset_index(drop=True)
        df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0)
        df[target_col]   = df[target_col].ffill().bfill().fillna(0)

        features = df[feature_cols].values.astype(np.float32)
        targets  = df[target_col].values.astype(np.float32)
        hours    = df["hour_cst"].values.astype(np.float32)
        dows     = df["dow_cst"].values.astype(np.float32)

        if normalise == "zscore":
            if feature_stats is None:
                feature_stats = {"mean": features.mean(0), "std": features.std(0) + 1e-8}
            features = (features - feature_stats["mean"]) / feature_stats["std"]
            if target_stats is None:
                target_stats = {"mean": float(targets.mean()), "std": float(targets.std()) + 1e-8}
            targets = (targets - target_stats["mean"]) / target_stats["std"]
        elif normalise == "minmax":
            if feature_stats is None:
                feature_stats = {"min": features.min(0), "max": features.max(0) + 1e-8}
            features = (features - feature_stats["min"]) / (feature_stats["max"] - feature_stats["min"] + 1e-8)
            if target_stats is None:
                target_stats = {"min": float(targets.min()), "max": float(targets.max()) + 1e-8}
            targets = (targets - target_stats["min"]) / (target_stats["max"] - target_stats["min"] + 1e-8)

        self.feature_stats = feature_stats
        self.target_stats  = target_stats

        n = len(df)
        self.windows: list[tuple] = [
            (
                features[i          : i + input_len],
                hours   [i          : i + input_len],
                dows    [i          : i + input_len],
                targets [i + input_len : i + total],
            )
            for i in range(n - total + 1)
        ]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        feat, hour, dow, tgt = self.windows[idx]
        return (
            torch.from_numpy(feat.copy()),
            torch.from_numpy(hour.copy()),
            torch.from_numpy(dow.copy()),
            torch.from_numpy(tgt.copy()),
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_dataloaders(cfg: DataConfig):
    """
    Discover session files for cfg.mode, split by session, fit normalisation
    stats on train sessions only, and return DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader (test may be None if too few sessions)
    feature_stats : dict
    target_stats  : dict
    feature_cols  : list[str]
    """
    target_col   = TARGET_COL[cfg.target]
    feature_cols = cfg.features

    print(f"\n  Mode   : {cfg.mode}  (file suffix: *{MODE_SUFFIX[cfg.mode]})")
    print(f"  Target : {target_col}")
    print(f"  Root   : {cfg.data_root}")

    all_paths = find_session_files(cfg.data_root, cfg.mode)
    print(f"  Found {len(all_paths)} session file(s)")
    if not all_paths:
        raise FileNotFoundError(
            f"No '*{MODE_SUFFIX[cfg.mode]}' files found under '{cfg.data_root}'.\n"
            "Check that data_root points to the folder that contains the "
            "starlink_YYYYMMDD_HHMMSS/ session subdirectories."
        )

    # Load and validate
    need_cols = set(MANDATORY_COLS + feature_cols + [target_col])
    dfs: list[pd.DataFrame] = []
    for p in all_paths:
        df = load_csv(p)
        if df is None:
            continue
        missing = need_cols - set(df.columns)
        if missing:
            print(f"  [warn] {p.parent.name}/{p.name}: missing columns {missing}, skipping")
            continue
        min_rows = cfg.input_len + cfg.output_len + 1
        if len(df) < min_rows:
            print(f"  [warn] {p.name}: only {len(df)} rows (need {min_rows}), skipping")
            continue
        dfs.append(df[sorted(need_cols)])

    if not dfs:
        raise ValueError("No usable session files after loading — check column names and data_root.")
    print(f"  Usable : {len(dfs)} session(s)")

    # Session-level split
    rng     = np.random.default_rng(cfg.seed)
    idx     = rng.permutation(len(dfs))
    n_train = max(1, int(len(dfs) * cfg.train_frac))
    n_val   = max(1, int(len(dfs) * cfg.val_frac))

    train_dfs = [dfs[i] for i in idx[:n_train]]
    val_dfs   = [dfs[i] for i in idx[n_train : n_train + n_val]]
    test_dfs  = [dfs[i] for i in idx[n_train + n_val :]]
    print(f"  Split  : train={len(train_dfs)}  val={len(val_dfs)}  test={len(test_dfs)} session(s)")

    # Fit normalisation on train only
    if cfg.normalise in ("zscore", "minmax"):
        tr = pd.concat(train_dfs, ignore_index=True)
        tr_feat = tr[feature_cols].ffill().bfill().fillna(0).values.astype(np.float32)
        tr_tgt  = tr[target_col].ffill().bfill().fillna(0).values.astype(np.float32)
        if cfg.normalise == "zscore":
            feature_stats: dict = {"mean": tr_feat.mean(0), "std": tr_feat.std(0) + 1e-8}
            target_stats:  dict = {"mean": float(tr_tgt.mean()), "std": float(tr_tgt.std()) + 1e-8}
        else:
            feature_stats = {"min": tr_feat.min(0), "max": tr_feat.max(0) + 1e-8}
            target_stats  = {"min": float(tr_tgt.min()), "max": float(tr_tgt.max()) + 1e-8}
    else:
        feature_stats = target_stats = None

    def make_dataset(df_list: list[pd.DataFrame]) -> ConcatDataset | None:
        datasets = []
        for df in df_list:
            ds = ThroughputFileDataset(
                df, feature_cols, target_col,
                cfg.input_len, cfg.output_len,
                feature_stats=feature_stats,
                target_stats=target_stats,
                normalise=cfg.normalise,
            )
            if len(ds) > 0:
                datasets.append(ds)
        return ConcatDataset(datasets) if datasets else None

    train_ds = make_dataset(train_dfs)
    val_ds   = make_dataset(val_dfs)
    test_ds  = make_dataset(test_dfs)

    def to_loader(ds, shuffle: bool) -> DataLoader | None:
        if ds is None or len(ds) == 0:
            return None
        return DataLoader(
            ds,
            batch_size  = cfg.batch_size,
            shuffle     = shuffle,
            num_workers = cfg.num_workers,
            pin_memory  = torch.cuda.is_available(),
        )

    train_loader = to_loader(train_ds, shuffle=True)
    val_loader   = to_loader(val_ds,   shuffle=False)
    test_loader  = to_loader(test_ds,  shuffle=False)

    print(
        f"  Windows: train={len(train_ds) if train_ds else 0:,}  "
        f"val={len(val_ds) if val_ds else 0:,}  "
        f"test={len(test_ds) if test_ds else 0:,}"
    )

    return train_loader, val_loader, test_loader, feature_stats, target_stats, feature_cols

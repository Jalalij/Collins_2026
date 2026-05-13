"""
dataset.py - Unified dataset for the Throughput Prediction Framework.

Data layout (authoritative)
---------------------------
<data_root>/
    starlink_20260506_030006/
        starlink_20260506_030006__udp_dl_400.csv   <- DL sessions
        starlink_20260506_030006__udp_ul_40.csv    <- UL sessions
    starlink_20260507_030024/
        ...

Mode-to-filename:
    mode="dl"  ->  *__udp_dl_400.csv
    mode="ul"  ->  *__udp_ul_40.csv

Multi-target:
    targets is a list of column names. E.g.:
        targets=["dish_dl_throughput_mbps", "ping_internet_rtt_ms_mean"]
    Each target is normalised independently; target_stats is keyed by col name.
    Single-target case returns (output_len,) tensors for backward compat.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODE_SUFFIX: dict[str, str] = {
    "dl": "__udp_dl_400.csv",
    "ul": "__udp_ul_40.csv",
}

ALL_FEATURES: list[str] = [
    "iperf_jitter_ms", "iperf_loss_pct",
    "ping_tunnel_rtt_ms_mean", "ping_internet_rtt_ms_mean", "dish_pop_ping_ms",
    "distance_km", "sat_lat", "sat_lon", "sat_alt_km",
    "dish_snr", "dish_gps_sats", "dish_boresight_azimuth_deg",
    "dish_boresight_elevation_deg", "dish_tilt_angle_deg", "dish_obstruction_fraction",
    "satellite_dtc", "satellite_handoff", "satellite_age_s",
    "weather_temp_f", "weather_humidity", "weather_pressure",
    "weather_wind_speed", "weather_wind_dir", "weather_rain_rate",
]

DEFAULT_FEATURES: list[str] = [
    "iperf_jitter_ms", "iperf_loss_pct",
    "ping_tunnel_rtt_ms_mean", "ping_internet_rtt_ms_mean", "dish_pop_ping_ms",
    "distance_km", "dish_snr", "dish_boresight_elevation_deg",
    "dish_obstruction_fraction", "weather_temp_f", "weather_wind_speed",
]

MANDATORY_COLS: list[str] = ["timestamp_cst", "hour_cst", "dow_cst"]

# All columns available as prediction targets
ALL_TARGETS: list[str] = [
    "dish_dl_throughput_mbps",
    "dish_ul_throughput_mbps",
    "iperf_throughput_mbps",
    "ping_internet_rtt_ms_mean",
    "ping_tunnel_rtt_ms_mean",
    "iperf_jitter_ms",
    "iperf_loss_pct",
    "dish_pop_ping_ms",
    "dish_snr",
]

# Shorthand aliases → column name
TARGET_COL: dict[str, str] = {
    "dl":    "dish_dl_throughput_mbps",
    "ul":    "dish_ul_throughput_mbps",
    "iperf": "iperf_throughput_mbps",
    "rtt":   "ping_internet_rtt_ms_mean",
    "snr":   "dish_snr",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    data_root:   str                   = "data/starlink"
    mode:        str                   = "dl"
    # List of target columns (or shorthand keys). If None, uses mode default.
    targets:     Optional[list[str]]   = None
    features:    Optional[list[str]]   = None
    input_len:   int                   = 30
    output_len:  int                   = 15
    train_frac:  float                 = 0.70
    val_frac:    float                 = 0.15
    normalise:   str                   = "zscore"
    batch_size:  int                   = 512
    num_workers: int                   = 0
    seed:        int                   = 42
    # Legacy compat
    target:      Optional[str]         = None

    def __post_init__(self) -> None:
        assert self.mode in MODE_SUFFIX, \
            f"mode must be one of {list(MODE_SUFFIX)}, got '{self.mode}'"
        assert self.train_frac + self.val_frac < 1.0, "train_frac + val_frac must be < 1.0"

        # Resolve targets list (support shorthand keys)
        if self.targets is not None:
            self.targets = [TARGET_COL.get(t, t) for t in self.targets]
        elif self.target is not None:
            self.targets = [TARGET_COL.get(self.target, self.target)]
        else:
            self.targets = [TARGET_COL[self.mode]]

        if self.features is None:
            self.features = list(DEFAULT_FEATURES)

    @property
    def num_targets(self) -> int:
        return len(self.targets)

    @property
    def primary_target(self) -> str:
        return self.targets[0]


# ---------------------------------------------------------------------------
# File discovery & CSV loading
# ---------------------------------------------------------------------------

def find_session_files(data_root: str, mode: str) -> list[Path]:
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
            print(f"  [warn] multiple matches in {session_dir.name}, using {matches[0].name}")
            found.append(matches[0])
        else:
            print(f"  [warn] no '*{suffix}' in {session_dir.name}, skipping")
    return found


def load_csv(path: Path) -> Optional[pd.DataFrame]:
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
# Dataset
# ---------------------------------------------------------------------------

class ThroughputFileDataset(Dataset):
    """
    Sliding-window dataset supporting one or more prediction targets.

    Item:
        features : (input_len, num_features)   float32  normalised
        hour     : (input_len,)                float32  raw 0-23
        dow      : (input_len,)                float32  raw 0-6
        target   : (output_len,)               float32  single target (num_targets==1)
                   (output_len, num_targets)   float32  multi-target
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_cols: list[str],
        input_len: int = 30,
        output_len: int = 15,
        feature_stats: Optional[dict] = None,
        target_stats: Optional[dict] = None,   # {col_name: {mean,std} or {min,max}}
        normalise: str = "zscore",
    ) -> None:
        self.input_len    = input_len
        self.output_len   = output_len
        self.normalise    = normalise
        self.feature_cols = feature_cols
        self.target_cols  = target_cols
        self.num_targets  = len(target_cols)

        df = df.sort_values("timestamp_cst").reset_index(drop=True)
        df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0)
        for tc in target_cols:
            df[tc] = df[tc].ffill().bfill().fillna(0)

        features = df[feature_cols].values.astype(np.float32)
        targets  = df[target_cols].values.astype(np.float32)   # (N, T)
        hours    = df["hour_cst"].values.astype(np.float32)
        dows     = df["dow_cst"].values.astype(np.float32)

        # Normalise features
        if normalise == "zscore":
            if feature_stats is None:
                feature_stats = {"mean": features.mean(0), "std": features.std(0) + 1e-8}
            features = (features - feature_stats["mean"]) / feature_stats["std"]
        elif normalise == "minmax":
            if feature_stats is None:
                feature_stats = {"min": features.min(0), "max": features.max(0) + 1e-8}
            features = (features - feature_stats["min"]) / (feature_stats["max"] - feature_stats["min"] + 1e-8)

        self.feature_stats = feature_stats

        # Normalise each target column independently
        if target_stats is None:
            target_stats = {}

        normed = targets.copy()
        for i, tc in enumerate(target_cols):
            col_vals = targets[:, i]
            if tc not in target_stats:
                if normalise == "zscore":
                    target_stats[tc] = {"mean": float(col_vals.mean()), "std": float(col_vals.std()) + 1e-8}
                elif normalise == "minmax":
                    target_stats[tc] = {"min": float(col_vals.min()), "max": float(col_vals.max()) + 1e-8}
            stats = target_stats.get(tc, {})
            if normalise == "zscore" and stats:
                normed[:, i] = (col_vals - stats["mean"]) / stats["std"]
            elif normalise == "minmax" and stats:
                r = stats["max"] - stats["min"]
                normed[:, i] = (col_vals - stats["min"]) / (r + 1e-8)

        self.target_stats = target_stats

        # Sliding windows
        total = input_len + output_len
        N = len(df)
        self.windows = []
        for start in range(N - total + 1):
            self.windows.append((
                features[start          : start + input_len],
                hours   [start          : start + input_len],
                dows    [start          : start + input_len],
                normed  [start + input_len : start + total],   # (output_len, num_targets)
            ))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        feat, hour, dow, tgt = self.windows[idx]
        tgt_t = torch.from_numpy(tgt.copy())
        if self.num_targets == 1:
            tgt_t = tgt_t.squeeze(-1)      # (output_len,) for backward compat
        return (
            torch.from_numpy(feat.copy()),
            torch.from_numpy(hour.copy()),
            torch.from_numpy(dow.copy()),
            tgt_t,
        )

    def inverse_transform(self, x: torch.Tensor, target_idx: int = 0) -> torch.Tensor:
        """Denormalise tensor for the given target index."""
        if not self.target_stats or self.normalise not in ("zscore", "minmax"):
            return x
        col   = self.target_cols[target_idx]
        stats = self.target_stats.get(col)
        if stats is None:
            return x
        if self.normalise == "zscore":
            return x.float() * stats["std"] + stats["mean"]
        return x.float() * (stats["max"] - stats["min"]) + stats["min"]

    def inverse_transform_all(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalise a (..., num_targets) tensor for all targets.
        """
        if not self.target_stats or self.normalise not in ("zscore", "minmax"):
            return x
        result = x.clone().float()
        for i, col in enumerate(self.target_cols):
            stats = self.target_stats.get(col)
            if stats is None:
                continue
            if self.normalise == "zscore":
                result[..., i] = result[..., i] * stats["std"] + stats["mean"]
            else:
                result[..., i] = result[..., i] * (stats["max"] - stats["min"]) + stats["min"]
        return result


# ---------------------------------------------------------------------------
# ExperimentDataset — adapter for legacy trainers
# ---------------------------------------------------------------------------

class ExperimentDataset(Dataset):
    """
    Wraps ThroughputFileDataset into the (X [T,F], y [F]) interface
    expected by the legacy VAR/VARIMA/LSTM/Transformer trainers.
    y is the first prediction step across all targets.
    """

    def __init__(
        self,
        base_dataset,
        target_stats: dict,
        normalise: str,
        num_features: int,
        target_cols: Optional[list[str]] = None,
    ) -> None:
        self.base         = base_dataset
        self.target_stats = target_stats
        self.normalise    = normalise
        self.num_features = num_features
        self.target_cols  = target_cols or []

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        feat, hour, dow, tgt = self.base[idx]
        h_sin = torch.sin(2 * torch.pi * hour / 24).unsqueeze(-1)
        h_cos = torch.cos(2 * torch.pi * hour / 24).unsqueeze(-1)
        d_sin = torch.sin(2 * torch.pi * dow / 7).unsqueeze(-1)
        d_cos = torch.cos(2 * torch.pi * dow / 7).unsqueeze(-1)
        x = torch.cat([feat, h_sin, h_cos, d_sin, d_cos], dim=-1)  # (T, F+4)
        # First prediction step
        if tgt.dim() == 2:
            y_tgt = tgt[0, :]      # (num_targets,)
        else:
            y_tgt = tgt[0:1]       # (1,)
        y = torch.cat([x[-1, :], y_tgt.float()], dim=0)
        return x, y

    def inverse_transform(self, x: torch.Tensor, target_idx: int = 0) -> torch.Tensor:
        if not self.target_stats or self.normalise not in ("zscore", "minmax"):
            return x
        if self.target_cols:
            col   = self.target_cols[target_idx]
            stats = self.target_stats.get(col)
        else:
            stats = self.target_stats
        if stats is None:
            return x
        if isinstance(stats, dict) and "mean" in stats:
            return x.float() * stats["std"] + stats["mean"]
        if isinstance(stats, dict) and "min" in stats:
            return x.float() * (stats["max"] - stats["min"]) + stats["min"]
        return x


# ---------------------------------------------------------------------------
# Public build functions
# ---------------------------------------------------------------------------

def build_dataloaders(cfg: DataConfig):
    """
    Returns: train_loader, val_loader, test_loader, feature_stats, target_stats, feature_cols
    target_stats is {col_name: {mean,std}} for all targets.
    test_loader may be None.
    """
    target_cols  = cfg.targets
    feature_cols = cfg.features

    print(f"\n  Mode    : {cfg.mode}  (suffix: *{MODE_SUFFIX[cfg.mode]})")
    print(f"  Targets : {target_cols}")
    print(f"  Root    : {cfg.data_root}")

    all_paths = find_session_files(cfg.data_root, cfg.mode)
    print(f"  Found   : {len(all_paths)} session file(s)")
    if not all_paths:
        raise FileNotFoundError(f"No '*{MODE_SUFFIX[cfg.mode]}' files under '{cfg.data_root}'.")

    need_cols = set(MANDATORY_COLS + feature_cols + target_cols)
    dfs: list[pd.DataFrame] = []
    for p in all_paths:
        df = load_csv(p)
        if df is None:
            continue
        missing = need_cols - set(df.columns)
        if missing:
            print(f"  [warn] {p.parent.name}/{p.name}: missing {missing}, skipping")
            continue
        min_rows = cfg.input_len + cfg.output_len + 1
        if len(df) < min_rows:
            print(f"  [warn] {p.name}: only {len(df)} rows (need {min_rows}), skipping")
            continue
        keep = sorted(need_cols)
        dfs.append(df[[c for c in df.columns if c in keep or c == "timestamp_cst"]])

    if not dfs:
        raise ValueError("No usable session files — check column names and data_root.")
    print(f"  Usable  : {len(dfs)} session(s)")

    rng     = np.random.default_rng(cfg.seed)
    idx     = rng.permutation(len(dfs))
    n_train = max(1, int(len(dfs) * cfg.train_frac))
    n_val   = max(1, int(len(dfs) * cfg.val_frac))

    train_dfs = [dfs[i] for i in idx[:n_train]]
    val_dfs   = [dfs[i] for i in idx[n_train : n_train + n_val]]
    test_dfs  = [dfs[i] for i in idx[n_train + n_val :]]
    print(f"  Split   : train={len(train_dfs)}  val={len(val_dfs)}  test={len(test_dfs)}")

    feature_stats: Optional[dict] = None
    target_stats:  dict           = {}

    if cfg.normalise in ("zscore", "minmax"):
        tr      = pd.concat(train_dfs, ignore_index=True)
        tr_feat = tr[feature_cols].ffill().bfill().fillna(0).values.astype(np.float32)
        if cfg.normalise == "zscore":
            feature_stats = {"mean": tr_feat.mean(0), "std": tr_feat.std(0) + 1e-8}
        else:
            feature_stats = {"min": tr_feat.min(0), "max": tr_feat.max(0) + 1e-8}

        for tc in target_cols:
            tr_tgt = tr[tc].ffill().bfill().fillna(0).values.astype(np.float32)
            if cfg.normalise == "zscore":
                target_stats[tc] = {"mean": float(tr_tgt.mean()), "std": float(tr_tgt.std()) + 1e-8}
            else:
                target_stats[tc] = {"min": float(tr_tgt.min()), "max": float(tr_tgt.max()) + 1e-8}

    def make_ds(df_list):
        datasets = []
        for df in df_list:
            ds = ThroughputFileDataset(
                df, feature_cols, target_cols,
                cfg.input_len, cfg.output_len,
                feature_stats=feature_stats,
                target_stats=target_stats if target_stats else None,
                normalise=cfg.normalise,
            )
            if len(ds) > 0:
                datasets.append(ds)
        return ConcatDataset(datasets) if datasets else None

    train_ds = make_ds(train_dfs)
    val_ds   = make_ds(val_dfs)
    test_ds  = make_ds(test_dfs)

    def to_loader(ds, shuffle):
        if ds is None or len(ds) == 0:
            return None
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())

    print(f"  Windows : train={len(train_ds) if train_ds else 0:,}  "
          f"val={len(val_ds) if val_ds else 0:,}  "
          f"test={len(test_ds) if test_ds else 0:,}")

    return (to_loader(train_ds, True), to_loader(val_ds, False), to_loader(test_ds, False),
            feature_stats, target_stats, feature_cols)


def build_experiment_datasets(cfg: DataConfig):
    """
    Returns ExperimentDataset for legacy trainers.
    Returns: train_set, test_set, feature_stats, target_stats, feature_cols
    """
    (train_loader, _, test_loader,
     feature_stats, target_stats, feature_cols) = build_dataloaders(cfg)

    n_feat = len(feature_cols) + 4

    train_set = ExperimentDataset(
        train_loader.dataset, target_stats, cfg.normalise, n_feat, cfg.targets
    )
    test_set = ExperimentDataset(
        (test_loader.dataset if test_loader else train_loader.dataset),
        target_stats, cfg.normalise, n_feat, cfg.targets,
    )
    return train_set, test_set, feature_stats, target_stats, feature_cols

"""
dataset.py - Unified dataset for the Throughput Prediction Framework.

Supported providers
-------------------
starlink   : data/lsn-data/starlink/{latency,latency_loss_throughput,throughput}/
oneweb     : data/lsn-data/oneweb/{dl,ul}/
quectel    : data/lsn-data/quectel/{latency,latency_loss_throughput,throughput}/

Provider / experiment-type to filename pattern
-----------------------------------------------
Starlink:
  "latency"               -> starlink_*_latency.csv
  "latency_loss_throughput" -> starlink_*_latency_loss_throughput.csv
  "throughput"            -> starlink_*_throughput.csv

OneWeb:
  "dl"                    -> ex_*_DL.csv   (inside oneweb/dl/)
  "ul"                    -> ex_*_UL.csv   (inside oneweb/ul/)

Quectel:
  "latency"               -> quectel_*_latency.csv
  "latency_loss_throughput" -> quectel_*_latency_loss_throughput.csv
  "throughput"            -> quectel_*_throughput.csv

Multi-target:
  targets is a list of column names or shorthand keys.
  Each target is normalised independently; target_stats is keyed by col name.
  Single-target returns (output_len,) tensors for backward compat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

PROVIDER_STARLINK = "starlink"
PROVIDER_ONEWEB   = "oneweb"
PROVIDER_QUECTEL  = "quectel"

PROVIDERS = (PROVIDER_STARLINK, PROVIDER_ONEWEB, PROVIDER_QUECTEL)

# ---------------------------------------------------------------------------
# File-layout definitions per provider
# ---------------------------------------------------------------------------

# Maps (provider, experiment_type) -> (subdirectory, filename_suffix)
# subdirectory is relative to data_root/<provider>/
FILE_LAYOUT: dict[tuple[str, str], tuple[str, str]] = {
    # Starlink
    (PROVIDER_STARLINK, "latency"):                  ("latency",                 "_latency.csv"),
    (PROVIDER_STARLINK, "latency_loss_throughput"):  ("latency_loss_throughput", "_latency_loss_throughput.csv"),
    (PROVIDER_STARLINK, "throughput"):               ("throughput",              "_throughput.csv"),
    # OneWeb
    (PROVIDER_ONEWEB,   "dl"):                       ("dl",                      "_DL.csv"),
    (PROVIDER_ONEWEB,   "ul"):                       ("ul",                      "_UL.csv"),
    # Quectel
    (PROVIDER_QUECTEL,  "latency"):                  ("latency",                 "_latency.csv"),
    (PROVIDER_QUECTEL,  "latency_loss_throughput"):  ("latency_loss_throughput", "_latency_loss_throughput.csv"),
    (PROVIDER_QUECTEL,  "throughput"):               ("throughput",              "_throughput.csv"),
}

VALID_EXPERIMENT_TYPES: dict[str, tuple[str, ...]] = {
    PROVIDER_STARLINK: ("latency", "latency_loss_throughput", "throughput"),
    PROVIDER_ONEWEB:   ("dl", "ul"),
    PROVIDER_QUECTEL:  ("latency", "latency_loss_throughput", "throughput"),
}

# ---------------------------------------------------------------------------
# Timestamp / time-index columns per provider
# ---------------------------------------------------------------------------

TIMESTAMP_COL: dict[str, str] = {
    PROVIDER_STARLINK: "timestamp_cst",
    PROVIDER_ONEWEB:   "timestamp",
    PROVIDER_QUECTEL:  "timestamp_cst",
}

# Pre-computed hour/dow column names (None = must derive from timestamp)
HOUR_COL: dict[str, Optional[str]] = {
    PROVIDER_STARLINK: "hour_cst",
    PROVIDER_ONEWEB:   "hour_of_day",
    PROVIDER_QUECTEL:  "hour_cst",
}

DOW_COL: dict[str, Optional[str]] = {
    PROVIDER_STARLINK: "dow_cst",
    PROVIDER_ONEWEB:   "day_of_week",
    PROVIDER_QUECTEL:  "dow_cst",
}

# ---------------------------------------------------------------------------
# Feature definitions — per (provider, experiment_type)
# ---------------------------------------------------------------------------
# ALL_FEATURES  : every column that could appear as a feature for that combo
# DEFAULT_FEATURES : the default input set used when --features is not specified
# Both are keyed by (provider, experiment_type).
# ALL_FEATURES is also exposed per-provider (union) for argparse/listing.

# -- Starlink -----------------------------------------------------------------
_SL_COMMON = [
    "dish_pop_ping_ms", "satellite_dtc", "satellite_handoff", "satellite_age_s",
    "distance_km", "elevation_deg", "azimuth_deg",
    "dish_obstruction_fraction", "dish_snr",
    "dish_boresight_azimuth_deg", "dish_boresight_elevation_deg",
    "weather_humidity", "weather_pressure", "weather_rain_rate",
]

ALL_FEATURES_BY_EXP: dict[tuple[str, str], list[str]] = {
    # Starlink latency
    (PROVIDER_STARLINK, "latency"): _SL_COMMON + [
        "ping_server_rtt_ms", "ping_dns_rtt_ms",
    ],
    # Starlink latency_loss_throughput
    (PROVIDER_STARLINK, "latency_loss_throughput"): _SL_COMMON + [
        "ping_server_rtt_ms", "ping_dns_rtt_ms",
        "iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct",
    ],
    # Starlink throughput
    (PROVIDER_STARLINK, "throughput"): _SL_COMMON + [
        "iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct",
    ],
    # OneWeb dl / ul (identical column sets)
    (PROVIDER_ONEWEB, "dl"): [
        "throughput_mbps", "packet_loss_percent", "jitter_ms", "rtt_mean",
        "sinr_db", "elevation_deg", "azimuth_deg",
        "satellite_change", "time_since_handover",
    ],
    (PROVIDER_ONEWEB, "ul"): [
        "throughput_mbps", "packet_loss_percent", "jitter_ms", "rtt_mean",
        "sinr_db", "elevation_deg", "azimuth_deg",
        "satellite_change", "time_since_handover",
    ],
    # Quectel latency
    (PROVIDER_QUECTEL, "latency"): [
        "ping_rtt_ms",
        "rsrp", "rsrq", "sinr",
        "weather_humidity", "weather_pressure", "weather_rain_rate",
    ],
    # Quectel latency_loss_throughput
    (PROVIDER_QUECTEL, "latency_loss_throughput"): [
        "iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct",
        "ping_rtt_ms",
        "rsrp", "rsrq", "sinr",
        "weather_humidity", "weather_pressure", "weather_rain_rate",
    ],
    # Quectel throughput
    (PROVIDER_QUECTEL, "throughput"): [
        "iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct",
        "rsrp", "rsrq", "sinr",
        "weather_humidity", "weather_pressure", "weather_rain_rate",
    ],
}

DEFAULT_FEATURES_BY_EXP: dict[tuple[str, str], list[str]] = {
    # Starlink latency
    (PROVIDER_STARLINK, "latency"): [
        "ping_server_rtt_ms", "ping_dns_rtt_ms",
        "dish_pop_ping_ms", "satellite_dtc", "satellite_handoff", "satellite_age_s",
        "distance_km", "elevation_deg", "azimuth_deg",
        "dish_obstruction_fraction", "dish_snr",
        "dish_boresight_azimuth_deg", "dish_boresight_elevation_deg",
        "weather_humidity", "weather_pressure", "weather_rain_rate",
    ],
    # Starlink latency_loss_throughput
    (PROVIDER_STARLINK, "latency_loss_throughput"): [
        "ping_server_rtt_ms", "ping_dns_rtt_ms",
        "iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct",
        "dish_pop_ping_ms", "satellite_dtc", "satellite_handoff", "satellite_age_s",
        "distance_km", "elevation_deg", "azimuth_deg",
        "dish_obstruction_fraction", "dish_snr",
        "dish_boresight_azimuth_deg", "dish_boresight_elevation_deg",
        "weather_humidity", "weather_pressure", "weather_rain_rate",
    ],
    # Starlink throughput
    (PROVIDER_STARLINK, "throughput"): [
        "iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct",
        "dish_pop_ping_ms", "satellite_dtc", "satellite_handoff", "satellite_age_s",
        "distance_km", "elevation_deg", "azimuth_deg",
        "dish_obstruction_fraction", "dish_snr",
        "dish_boresight_azimuth_deg", "dish_boresight_elevation_deg",
        "weather_humidity", "weather_pressure", "weather_rain_rate",
    ],
    # OneWeb dl / ul
    (PROVIDER_ONEWEB, "dl"): [
        "throughput_mbps", "packet_loss_percent", "jitter_ms", "rtt_mean",
        "sinr_db", "elevation_deg", "azimuth_deg",
        "satellite_change", "time_since_handover",
    ],
    (PROVIDER_ONEWEB, "ul"): [
        "throughput_mbps", "packet_loss_percent", "jitter_ms", "rtt_mean",
        "sinr_db", "elevation_deg", "azimuth_deg",
        "satellite_change", "time_since_handover",
    ],
    # Quectel latency
    (PROVIDER_QUECTEL, "latency"): [
        "ping_rtt_ms",
        "rsrp", "rsrq", "sinr",
        "weather_humidity", "weather_pressure", "weather_rain_rate",
    ],
    # Quectel latency_loss_throughput
    (PROVIDER_QUECTEL, "latency_loss_throughput"): [
        "iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct",
        "ping_rtt_ms",
        "rsrp", "rsrq", "sinr",
        "weather_humidity", "weather_pressure", "weather_rain_rate",
    ],
    # Quectel throughput
    (PROVIDER_QUECTEL, "throughput"): [
        "iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct",
        "rsrp", "rsrq", "sinr",
        "weather_humidity", "weather_pressure", "weather_rain_rate",
    ],
}

# Per-provider union of all features (for listing / argparse help)
ALL_FEATURES: dict[str, list[str]] = {
    p: sorted({
        f
        for (prov, _), feats in ALL_FEATURES_BY_EXP.items()
        if prov == p
        for f in feats
    })
    for p in PROVIDERS
}

# Per-provider default features (union, for backward-compat imports)
DEFAULT_FEATURES: dict[str, list[str]] = {
    p: sorted({
        f
        for (prov, _), feats in DEFAULT_FEATURES_BY_EXP.items()
        if prov == p
        for f in feats
    })
    for p in PROVIDERS
}

# ---------------------------------------------------------------------------
# Target definitions — per (provider, experiment_type)
# ---------------------------------------------------------------------------

DEFAULT_TARGETS_BY_EXP: dict[tuple[str, str], list[str]] = {
    (PROVIDER_STARLINK, "latency"):                 ["ping_server_rtt_ms", "ping_dns_rtt_ms"],
    (PROVIDER_STARLINK, "latency_loss_throughput"): ["iperf_throughput_mbps", "iperf_jitter_ms",
                                                     "iperf_loss_pct",
                                                     "ping_server_rtt_ms", "ping_dns_rtt_ms"],
    (PROVIDER_STARLINK, "throughput"):              ["iperf_throughput_mbps", "iperf_jitter_ms",
                                                     "iperf_loss_pct"],
    (PROVIDER_ONEWEB,   "dl"):                      ["throughput_mbps", "packet_loss_percent",
                                                     "jitter_ms", "rtt_mean"],
    (PROVIDER_ONEWEB,   "ul"):                      ["throughput_mbps", "packet_loss_percent",
                                                     "jitter_ms", "rtt_mean"],
    (PROVIDER_QUECTEL,  "latency"):                 ["ping_rtt_ms"],
    (PROVIDER_QUECTEL,  "latency_loss_throughput"): ["iperf_throughput_mbps", "iperf_jitter_ms",
                                                     "iperf_loss_pct", "ping_rtt_ms"],
    (PROVIDER_QUECTEL,  "throughput"):              ["iperf_throughput_mbps", "iperf_jitter_ms",
                                                     "iperf_loss_pct"],
}

ALL_TARGETS: dict[str, list[str]] = {
    PROVIDER_STARLINK: [
        "ping_server_rtt_ms", "ping_dns_rtt_ms",
        "iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct",
        "dish_dl_throughput_mbps", "dish_ul_throughput_mbps",
        "dish_pop_ping_ms", "dish_snr",
    ],
    PROVIDER_ONEWEB: [
        "throughput_mbps", "packet_loss_percent",
        "jitter_ms", "rtt_mean", "sinr_db",
    ],
    PROVIDER_QUECTEL: [
        "iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct",
        "ping_rtt_ms", "sinr", "rsrp",
    ],
}

# Shorthand aliases per provider: short key -> column name
TARGET_ALIASES: dict[str, dict[str, str]] = {
    PROVIDER_STARLINK: {
        "iperf":    "iperf_throughput_mbps",
        "jitter":   "iperf_jitter_ms",
        "loss":     "iperf_loss_pct",
        "rtt":      "ping_server_rtt_ms",
        "dns":      "ping_dns_rtt_ms",
        "dl":       "dish_dl_throughput_mbps",
        "ul":       "dish_ul_throughput_mbps",
        "pop":      "dish_pop_ping_ms",
        "snr":      "dish_snr",
    },
    PROVIDER_ONEWEB: {
        "tput":     "throughput_mbps",
        "loss":     "packet_loss_percent",
        "jitter":   "jitter_ms",
        "rtt":      "rtt_mean",
        "sinr":     "sinr_db",
    },
    PROVIDER_QUECTEL: {
        "iperf":    "iperf_throughput_mbps",
        "jitter":   "iperf_jitter_ms",
        "loss":     "iperf_loss_pct",
        "rtt":      "ping_rtt_ms",
        "sinr":     "sinr",
        "rsrp":     "rsrp",
    },
}

# First target of the default set — used as fallback in legacy single-target paths
DEFAULT_TARGET: dict[tuple[str, str], str] = {
    k: v[0] for k, v in DEFAULT_TARGETS_BY_EXP.items()
}


# ---------------------------------------------------------------------------
# Feature groups  (single source of truth — imported by run.py, ablation.py)
# ---------------------------------------------------------------------------

FEATURE_GROUPS: dict[str, list[str]] = {
    # Starlink
    "network_perf":       [
        "iperf_jitter_ms", "iperf_loss_pct",
        "ping_tunnel_rtt_ms", "dish_pop_ping_ms",
    ],
    "satellite_geometry": [
        "distance_km", "sat_lat", "sat_lon", "sat_alt_km",
    ],
    "dish_state":         [
        "dish_snr", "dish_gps_sats",
        "dish_boresight_azimuth_deg", "dish_boresight_elevation_deg",
        "dish_tilt_angle_deg", "dish_obstruction_fraction",
    ],
    "weather":            [
        "weather_temp_f", "weather_humidity", "weather_pressure",
        "weather_wind_speed", "weather_wind_dir", "weather_rain_rate",
    ],
    "satellite_meta":     [
        "satellite_dtc", "satellite_handoff", "satellite_age_s",
    ],
    # OneWeb
    "oneweb_signal":      [
        "sinr_db", "elevation_deg", "azimuth_deg",
    ],
    "oneweb_handover":    [
        "inter_satellite_ho_rate", "inter_beam_ho_rate",
        "beam_change", "satellite_change", "time_since_handover",
    ],
    # Quectel
    "cellular_signal":    [
        "rsrp", "rsrq", "sinr",
    ],
    "cellular_perf":      [
        "iperf_jitter_ms", "iperf_loss_pct", "ping_rtt_ms",
    ],
    "cellular_meta":      [
        "cell_change", "handover", "arfcn",
    ],
}

# Flat sorted list of every feature name across all providers — useful for
# argparse metavar help text and cross-provider validation.
ALL_FEATURE_NAMES: list[str] = sorted({
    f for feats in ALL_FEATURES.values() for f in feats
})


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    # Required
    provider:         str                  = PROVIDER_STARLINK
    experiment_type:  str                  = "throughput"

    # Paths
    data_root:        str                  = "data/lsn-data"

    # Targets / features — if None, sensible defaults are chosen
    targets:          Optional[list[str]]  = None
    features:         Optional[list[str]]  = None

    # Window / split
    input_len:        int                  = 30
    output_len:       int                  = 15
    train_frac:       float                = 0.70
    val_frac:         float                = 0.15

    # Normalisation
    normalise:        str                  = "zscore"

    # DataLoader
    batch_size:       int                  = 512
    num_workers:      int                  = 0
    seed:             int                  = 42

    # Legacy single-target compat
    target:           Optional[str]        = None

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {PROVIDERS}, got '{self.provider}'")
        valid_exp = VALID_EXPERIMENT_TYPES[self.provider]
        if self.experiment_type not in valid_exp:
            raise ValueError(
                f"experiment_type '{self.experiment_type}' invalid for provider "
                f"'{self.provider}'. Choose from {valid_exp}"
            )
        assert self.train_frac + self.val_frac < 1.0, "train_frac + val_frac must be < 1.0"

        aliases = TARGET_ALIASES[self.provider]
        exp_key = (self.provider, self.experiment_type)

        # Resolve targets list (support shorthand keys)
        if self.targets is not None:
            self.targets = [aliases.get(t, t) for t in self.targets]
        elif self.target is not None:
            self.targets = [aliases.get(self.target, self.target)]
        else:
            # Default: all recommended targets for this (provider, experiment_type)
            self.targets = list(DEFAULT_TARGETS_BY_EXP[exp_key])

        if self.features is None:
            self.features = list(DEFAULT_FEATURES_BY_EXP[exp_key])

    @property
    def num_targets(self) -> int:
        return len(self.targets)

    @property
    def primary_target(self) -> str:
        return self.targets[0]

    @property
    def provider_root(self) -> Path:
        """Root directory for the chosen provider."""
        return Path(self.data_root) / self.provider

    @property
    def timestamp_col(self) -> str:
        return TIMESTAMP_COL[self.provider]

    @property
    def hour_col(self) -> Optional[str]:
        return HOUR_COL[self.provider]

    @property
    def dow_col(self) -> Optional[str]:
        return DOW_COL[self.provider]


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_session_files(cfg: DataConfig) -> list[Path]:
    """
    Locate all CSV files for the given provider/experiment_type.

    Layout:
        Starlink / Quectel: <data_root>/<provider>/<subdir>/<files>
        OneWeb:             <data_root>/oneweb/<subdir>/<files>
                            (flat directory — no per-session sub-folders)
    """
    subdir, suffix = FILE_LAYOUT[(cfg.provider, cfg.experiment_type)]
    search_dir = cfg.provider_root / subdir

    if not search_dir.exists():
        raise FileNotFoundError(
            f"Expected directory '{search_dir}' does not exist. "
            f"Check data_root='{cfg.data_root}' and provider='{cfg.provider}'."
        )

    found = sorted(p for p in search_dir.iterdir()
                   if p.is_file() and p.name.endswith(suffix))

    if not found:
        print(f"  [warn] No '*{suffix}' files found in '{search_dir}'")

    return found


# ---------------------------------------------------------------------------
# CSV loading (provider-aware)
# ---------------------------------------------------------------------------

def _parse_timestamp(series: pd.Series) -> pd.Series:
    """Parse a timestamp column, stripping timezone info to naive UTC-equivalent."""
    parsed = pd.to_datetime(series, utc=False, errors="coerce")
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed


def load_csv(path: Path, cfg: DataConfig) -> Optional[pd.DataFrame]:
    """
    Load and minimally clean a session CSV.

    Returns a DataFrame with at minimum:
        - <timestamp_col>  : datetime64[ns]
        - hour             : float32  (0-23)
        - dow              : float32  (0-6)
        - all feature/target columns as float32 where applicable
    Non-numeric columns that are not the timestamp are left as-is so the
    caller can decide what to do with them.
    """
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        print(f"  [warn] could not read {path}: {exc}")
        return None

    ts_col = cfg.timestamp_col

    # Drop duplicate header rows (common in streaming-appended files)
    if ts_col in df.columns:
        df = df[df[ts_col] != ts_col].copy()
        df[ts_col] = _parse_timestamp(df[ts_col])
    else:
        print(f"  [warn] {path.name}: missing timestamp column '{ts_col}', skipping")
        return None

    # Derive hour/dow if not already present
    _ensure_time_cols(df, cfg)

    # Coerce numeric columns
    skip = {ts_col, "hour", "dow"}
    if cfg.provider == PROVIDER_QUECTEL:
        # cellid, inner_cell, base_station, band are categorical — keep as str
        skip |= {"cellid", "inner_cell", "base_station", "band", "weather_code_nws"}
    if cfg.provider == PROVIDER_ONEWEB:
        skip |= {"track_id"}  # large int ID — keep but won't be a feature

    for col in df.columns:
        if col in skip:
            continue
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Starlink throughput: dish_snr can be "True"/"False" (boolean bug in data)
    if cfg.provider == PROVIDER_STARLINK and "dish_snr" in df.columns:
        df["dish_snr"] = pd.to_numeric(df["dish_snr"], errors="coerce")

    return df


def _ensure_time_cols(df: pd.DataFrame, cfg: DataConfig) -> None:
    """
    Ensure 'hour' and 'dow' columns exist, deriving from timestamp if needed.
    Works in-place.
    """
    ts_col  = cfg.timestamp_col
    h_col   = cfg.hour_col
    d_col   = cfg.dow_col

    if h_col and h_col in df.columns:
        df["hour"] = pd.to_numeric(df[h_col], errors="coerce").astype("float32")
    else:
        df["hour"] = df[ts_col].dt.hour.astype("float32")

    if d_col and d_col in df.columns:
        df["dow"] = pd.to_numeric(df[d_col], errors="coerce").astype("float32")
    else:
        df["dow"] = df[ts_col].dt.dayofweek.astype("float32")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ThroughputFileDataset(Dataset):
    """
    Sliding-window dataset supporting one or more prediction targets.

    Item:
        features : (input_len, num_features)    float32  normalised
        hour     : (input_len,)                 float32  raw 0-23
        dow      : (input_len,)                 float32  raw 0-6
        target   : (output_len,)                float32  single target (num_targets==1)
                   (output_len, num_targets)    float32  multi-target

    feature_cols is the *intersection* of requested features and columns
    present in df, so partial column sets (e.g. Starlink latency missing
    iperf columns) are handled gracefully.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_cols: list[str],
        input_len: int = 30,
        output_len: int = 15,
        feature_stats: Optional[dict] = None,
        target_stats: Optional[dict] = None,
        normalise: str = "zscore",
    ) -> None:
        self.input_len    = input_len
        self.output_len   = output_len
        self.normalise    = normalise
        self.num_targets  = len(target_cols)

        # Use only features that are actually present in this file
        self.feature_cols = [c for c in feature_cols if c in df.columns]
        self.target_cols  = target_cols

        df = df.sort_values("timestamp" if "timestamp" in df.columns else df.columns[0])
        df = df.reset_index(drop=True)

        # Sort by whichever timestamp column exists
        ts_candidates = [c for c in df.columns if "timestamp" in c.lower()]
        if ts_candidates:
            df = df.sort_values(ts_candidates[0]).reset_index(drop=True)

        df[self.feature_cols] = (df[self.feature_cols]
                                 .apply(pd.to_numeric, errors="coerce")
                                 .ffill().bfill().fillna(0))
        for tc in target_cols:
            if tc in df.columns:
                df[tc] = (pd.to_numeric(df[tc], errors="coerce")
                          .ffill().bfill().fillna(0))

        features = df[self.feature_cols].values.astype(np.float32)
        targets  = df[target_cols].values.astype(np.float32)
        hours    = df["hour"].values.astype(np.float32)
        dows     = df["dow"].values.astype(np.float32)

        # Normalise features
        if normalise == "zscore":
            if feature_stats is None:
                feature_stats = {
                    "mean": features.mean(0),
                    "std":  features.std(0) + 1e-8,
                }
            # Align stats to present features (stats may cover superset)
            mean = _align_stats_array(feature_stats.get("mean"), features.shape[1])
            std  = _align_stats_array(feature_stats.get("std"),  features.shape[1], fill=1.0)
            features = (features - mean) / std
        elif normalise == "minmax":
            if feature_stats is None:
                feature_stats = {
                    "min": features.min(0),
                    "max": features.max(0) + 1e-8,
                }
            fmin = _align_stats_array(feature_stats.get("min"), features.shape[1])
            fmax = _align_stats_array(feature_stats.get("max"), features.shape[1], fill=1.0)
            features = (features - fmin) / (fmax - fmin + 1e-8)

        self.feature_stats = feature_stats

        # Normalise each target column independently
        if target_stats is None:
            target_stats = {}

        normed = targets.copy()
        for i, tc in enumerate(target_cols):
            col_vals = targets[:, i]
            if tc not in target_stats:
                if normalise == "zscore":
                    target_stats[tc] = {
                        "mean": float(col_vals.mean()),
                        "std":  float(col_vals.std()) + 1e-8,
                    }
                elif normalise == "minmax":
                    target_stats[tc] = {
                        "min": float(col_vals.min()),
                        "max": float(col_vals.max()) + 1e-8,
                    }
            stats = target_stats.get(tc, {})
            if normalise == "zscore" and stats:
                normed[:, i] = (col_vals - stats["mean"]) / stats["std"]
            elif normalise == "minmax" and stats:
                r = stats["max"] - stats["min"]
                normed[:, i] = (col_vals - stats["min"]) / (r + 1e-8)

        self.target_stats = target_stats

        # Build sliding windows
        total = input_len + output_len
        N = len(df)
        self.windows: list[tuple] = []
        for start in range(N - total + 1):
            self.windows.append((
                features[start          : start + input_len],
                hours   [start          : start + input_len],
                dows    [start          : start + input_len],
                normed  [start + input_len : start + total],
            ))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        feat, hour, dow, tgt = self.windows[idx]
        tgt_t = torch.from_numpy(tgt.copy())
        if self.num_targets == 1:
            tgt_t = tgt_t.squeeze(-1)
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
        """Denormalise a (..., num_targets) tensor for all targets."""
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
        d_sin = torch.sin(2 * torch.pi * dow  /  7).unsqueeze(-1)
        d_cos = torch.cos(2 * torch.pi * dow  /  7).unsqueeze(-1)
        x = torch.cat([feat, h_sin, h_cos, d_sin, d_cos], dim=-1)  # (T, F+4)
        # First prediction step
        if tgt.dim() == 2:
            y_tgt = tgt[0, :]
        else:
            y_tgt = tgt[0:1]
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
# Internal helpers
# ---------------------------------------------------------------------------

def _align_stats_array(
    arr: Optional[np.ndarray],
    n: int,
    fill: float = 0.0,
) -> np.ndarray:
    """
    Return a 1-D array of length n from arr.
    If arr is None or wrong length, pad/truncate with `fill`.
    """
    if arr is None:
        out = np.full(n, fill, dtype=np.float32)
    elif len(arr) == n:
        out = arr.astype(np.float32)
    elif len(arr) > n:
        out = arr[:n].astype(np.float32)
    else:
        out = np.full(n, fill, dtype=np.float32)
        out[:len(arr)] = arr
    return out


def _validate_and_filter_df(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    cfg: DataConfig,
    path: Path,
) -> Optional[pd.DataFrame]:
    """
    Check that the required target columns are present and the file has
    enough rows.  Returns the filtered df or None.
    """
    missing_targets = [tc for tc in target_cols if tc not in df.columns]
    if missing_targets:
        print(f"  [warn] {path.name}: missing target(s) {missing_targets}, skipping")
        return None

    min_rows = cfg.input_len + cfg.output_len + 1
    if len(df) < min_rows:
        print(f"  [warn] {path.name}: only {len(df)} rows (need {min_rows}), skipping")
        return None

    # Warn about missing features (soft — dataset handles intersection)
    missing_feat = [fc for fc in feature_cols if fc not in df.columns]
    if missing_feat:
        print(f"  [info] {path.name}: features absent (will be skipped): {missing_feat}")

    return df


# ---------------------------------------------------------------------------
# Public build functions
# ---------------------------------------------------------------------------

def build_dataloaders(cfg: DataConfig):
    """
    Build train / val / test DataLoaders for the given DataConfig.

    Returns
    -------
    train_loader   : DataLoader  (shuffled)
    val_loader     : DataLoader
    test_loader    : DataLoader | None
    feature_stats  : dict  (normalisation stats for features)
    target_stats   : dict  {col_name: {mean,std} | {min,max}}
    feature_cols   : list[str]  (features actually used)
    """
    target_cols  = cfg.targets
    feature_cols = cfg.features

    subdir, suffix = FILE_LAYOUT[(cfg.provider, cfg.experiment_type)]
    print(f"\n  Provider    : {cfg.provider}")
    print(f"  Exp type    : {cfg.experiment_type}  (suffix: *{suffix})")
    print(f"  Targets     : {target_cols}")
    print(f"  Root        : {cfg.data_root}")

    all_paths = find_session_files(cfg)
    print(f"  Found       : {len(all_paths)} file(s)")
    if not all_paths:
        raise FileNotFoundError(
            f"No '*{suffix}' files found for provider='{cfg.provider}', "
            f"experiment_type='{cfg.experiment_type}'."
        )

    dfs: list[pd.DataFrame] = []
    for p in all_paths:
        df = load_csv(p, cfg)
        if df is None:
            continue
        df = _validate_and_filter_df(df, feature_cols, target_cols, cfg, p)
        if df is not None:
            dfs.append(df)

    if not dfs:
        raise ValueError(
            "No usable session files — check column names, data_root, "
            "provider, and experiment_type."
        )
    print(f"  Usable      : {len(dfs)} file(s)")

    rng     = np.random.default_rng(cfg.seed)
    idx     = rng.permutation(len(dfs))
    n_train = max(1, int(len(dfs) * cfg.train_frac))
    n_val   = max(1, int(len(dfs) * cfg.val_frac))

    train_dfs = [dfs[i] for i in idx[:n_train]]
    val_dfs   = [dfs[i] for i in idx[n_train : n_train + n_val]]
    test_dfs  = [dfs[i] for i in idx[n_train + n_val :]]
    print(f"  Split       : train={len(train_dfs)}  val={len(val_dfs)}  test={len(test_dfs)}")

    feature_stats: Optional[dict] = None
    target_stats:  dict           = {}

    if cfg.normalise in ("zscore", "minmax"):
        tr      = pd.concat(train_dfs, ignore_index=True)
        # Use only the intersection of requested features with available cols
        avail_feat = [f for f in feature_cols if f in tr.columns]
        tr_feat = (tr[avail_feat]
                   .apply(pd.to_numeric, errors="coerce")
                   .ffill().bfill().fillna(0)
                   .values.astype(np.float32))

        if cfg.normalise == "zscore":
            feature_stats = {"mean": tr_feat.mean(0), "std": tr_feat.std(0) + 1e-8}
        else:
            feature_stats = {"min": tr_feat.min(0), "max": tr_feat.max(0) + 1e-8}

        for tc in target_cols:
            tr_tgt = (pd.to_numeric(tr[tc], errors="coerce")
                      .ffill().bfill().fillna(0)
                      .values.astype(np.float32))
            if cfg.normalise == "zscore":
                target_stats[tc] = {
                    "mean": float(tr_tgt.mean()),
                    "std":  float(tr_tgt.std()) + 1e-8,
                }
            else:
                target_stats[tc] = {
                    "min": float(tr_tgt.min()),
                    "max": float(tr_tgt.max()) + 1e-8,
                }

    def make_ds(df_list: list[pd.DataFrame]):
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

    def to_loader(ds, shuffle: bool):
        if ds is None or len(ds) == 0:
            return None
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    print(f"  Windows     : train={len(train_ds) if train_ds else 0:,}  "
          f"val={len(val_ds) if val_ds else 0:,}  "
          f"test={len(test_ds) if test_ds else 0:,}")

    return (
        to_loader(train_ds, True),
        to_loader(val_ds,   False),
        to_loader(test_ds,  False),
        feature_stats,
        target_stats,
        feature_cols,
    )


def build_experiment_datasets(cfg: DataConfig):
    """
    Build ExperimentDataset wrappers for legacy trainers.

    Returns
    -------
    train_set      : ExperimentDataset
    test_set       : ExperimentDataset
    feature_stats  : dict
    target_stats   : dict
    feature_cols   : list[str]
    """
    (train_loader, _, test_loader,
     feature_stats, target_stats, feature_cols) = build_dataloaders(cfg)

    n_feat = len(feature_cols) + 4  # +4 for sin/cos hour/dow encodings

    train_set = ExperimentDataset(
        train_loader.dataset, target_stats, cfg.normalise, n_feat, cfg.targets
    )
    test_set = ExperimentDataset(
        (test_loader.dataset if test_loader else train_loader.dataset),
        target_stats, cfg.normalise, n_feat, cfg.targets,
    )
    return train_set, test_set, feature_stats, target_stats, feature_cols
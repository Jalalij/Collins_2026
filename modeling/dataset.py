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
  "latency"                  -> starlink/latency/starlink_*_latency.csv
  "latency_loss_throughput_dl" -> starlink/latency_loss_throughput/dl/starlink_*_latency_loss_throughput_dl.csv
  "latency_loss_throughput_ul" -> starlink/latency_loss_throughput/ul/starlink_*_latency_loss_throughput_ul.csv
  "throughput_dl"            -> starlink/throughput/dl/starlink_*_throughput_dl.csv
  "throughput_ul"            -> starlink/throughput/ul/starlink_*_throughput_ul.csv

OneWeb:
  "dl"                       -> oneweb/dl/ex_*_DL.csv
  "ul"                       -> oneweb/ul/ex_*_UL.csv

Quectel:
  "latency"                  -> quectel/latency/quectel_*_latency.csv
  "latency_loss_throughput_dl" -> quectel/latency_loss_throughput/dl/quectel_*_latency_loss_throughput_dl.csv
  "latency_loss_throughput_ul" -> quectel/latency_loss_throughput/ul/quectel_*_latency_loss_throughput_ul.csv
  "throughput_dl"            -> quectel/throughput/dl/quectel_*_throughput_dl.csv
  "throughput_ul"            -> quectel/throughput/ul/quectel_*_throughput_ul.csv

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

# Maps (provider, experiment_type) -> (subdirectory_relative_to_provider_root, filename_suffix)
# For split types (latency_loss_throughput_dl/ul, throughput_dl/ul):
#   subdirectory includes the dl/ or ul/ subfolder.
FILE_LAYOUT: dict[tuple[str, str], tuple[str, str]] = {
    # Starlink
    (PROVIDER_STARLINK, "latency"):                    ("latency",                     "_latency.csv"),
    (PROVIDER_STARLINK, "latency_loss_throughput_dl"): ("latency_loss_throughput/dl",  "_latency_loss_throughput_dl.csv"),
    (PROVIDER_STARLINK, "latency_loss_throughput_ul"): ("latency_loss_throughput/ul",  "_latency_loss_throughput_ul.csv"),
    (PROVIDER_STARLINK, "throughput_dl"):              ("throughput/dl",               "_throughput_dl.csv"),
    (PROVIDER_STARLINK, "throughput_ul"):              ("throughput/ul",               "_throughput_ul.csv"),
    # OneWeb (unchanged — already split at top level)
    (PROVIDER_ONEWEB,   "dl"):                         ("dl",                          "_DL.csv"),
    (PROVIDER_ONEWEB,   "ul"):                         ("ul",                          "_UL.csv"),
    # Quectel
    (PROVIDER_QUECTEL,  "latency"):                    ("latency",                     "_latency.csv"),
    (PROVIDER_QUECTEL,  "latency_loss_throughput_dl"): ("latency_loss_throughput/dl",  "_latency_loss_throughput_dl.csv"),
    (PROVIDER_QUECTEL,  "latency_loss_throughput_ul"): ("latency_loss_throughput/ul",  "_latency_loss_throughput_ul.csv"),
    (PROVIDER_QUECTEL,  "throughput_dl"):              ("throughput/dl",               "_throughput_dl.csv"),
    (PROVIDER_QUECTEL,  "throughput_ul"):              ("throughput/ul",               "_throughput_ul.csv"),
}

VALID_EXPERIMENT_TYPES: dict[str, tuple[str, ...]] = {
    PROVIDER_STARLINK: ("latency", "latency_loss_throughput_dl", "latency_loss_throughput_ul",
                        "throughput_dl", "throughput_ul"),
    PROVIDER_ONEWEB:   ("dl", "ul"),
    PROVIDER_QUECTEL:  ("latency", "latency_loss_throughput_dl", "latency_loss_throughput_ul",
                        "throughput_dl", "throughput_ul"),
}

# ---------------------------------------------------------------------------
# Timestamp / time-index columns per provider
# ---------------------------------------------------------------------------

TIMESTAMP_COL: dict[str, str] = {
    PROVIDER_STARLINK: "timestamp_cst",
    PROVIDER_ONEWEB:   "timestamp",
    PROVIDER_QUECTEL:  "timestamp_cst",
}

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

_SL_COMMON = [
    "dish_pop_ping_ms", "satellite_dtc", "satellite_handoff", "satellite_age_s",
    "distance_km", "elevation_deg", "azimuth_deg",
    "dish_obstruction_fraction", "dish_snr",
    "dish_boresight_azimuth_deg", "dish_boresight_elevation_deg",
    "weather_humidity", "weather_pressure", "weather_rain_rate",
]

ALL_FEATURES_BY_EXP: dict[tuple[str, str], list[str]] = {
    # Starlink latency (unchanged)
    (PROVIDER_STARLINK, "latency"): _SL_COMMON + [
        "ping_server_rtt_ms", "ping_dns_rtt_ms",
    ],
    # Starlink latency_loss_throughput dl/ul — iperf columns are direction-specific
    (PROVIDER_STARLINK, "latency_loss_throughput_dl"): _SL_COMMON + [
        "ping_server_rtt_ms", "ping_dns_rtt_ms",
        "iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct",
    ],
    (PROVIDER_STARLINK, "latency_loss_throughput_ul"): _SL_COMMON + [
        "ping_server_rtt_ms", "ping_dns_rtt_ms",
        "iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct",
    ],
    # Starlink throughput dl/ul
    (PROVIDER_STARLINK, "throughput_dl"): _SL_COMMON + [
        "iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct",
    ],
    (PROVIDER_STARLINK, "throughput_ul"): _SL_COMMON + [
        "iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct",
    ],
    # OneWeb dl / ul (unchanged)
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
    # Quectel latency (unchanged)
    (PROVIDER_QUECTEL, "latency"): [
        "ping_rtt_ms",
        "rsrp", "rsrq", "sinr",
        "weather_humidity", "weather_pressure", "weather_rain_rate",
    ],
    # Quectel latency_loss_throughput dl/ul
    (PROVIDER_QUECTEL, "latency_loss_throughput_dl"): [
        "iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct",
        "ping_rtt_ms",
        "rsrp", "rsrq", "sinr",
        "weather_humidity", "weather_pressure", "weather_rain_rate",
    ],
    (PROVIDER_QUECTEL, "latency_loss_throughput_ul"): [
        "iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct",
        "ping_rtt_ms",
        "rsrp", "rsrq", "sinr",
        "weather_humidity", "weather_pressure", "weather_rain_rate",
    ],
    # Quectel throughput dl/ul
    (PROVIDER_QUECTEL, "throughput_dl"): [
        "iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct",
        "rsrp", "rsrq", "sinr",
        "weather_humidity", "weather_pressure", "weather_rain_rate",
    ],
    (PROVIDER_QUECTEL, "throughput_ul"): [
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
    # Starlink latency_loss_throughput dl/ul
    (PROVIDER_STARLINK, "latency_loss_throughput_dl"): [
        "ping_server_rtt_ms", "ping_dns_rtt_ms",
        "iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct",
        "dish_pop_ping_ms", "satellite_dtc", "satellite_handoff", "satellite_age_s",
        "distance_km", "elevation_deg", "azimuth_deg",
        "dish_obstruction_fraction", "dish_snr",
        "dish_boresight_azimuth_deg", "dish_boresight_elevation_deg",
        "weather_humidity", "weather_pressure", "weather_rain_rate",
    ],
    (PROVIDER_STARLINK, "latency_loss_throughput_ul"): [
        "ping_server_rtt_ms", "ping_dns_rtt_ms",
        "iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct",
        "dish_pop_ping_ms", "satellite_dtc", "satellite_handoff", "satellite_age_s",
        "distance_km", "elevation_deg", "azimuth_deg",
        "dish_obstruction_fraction", "dish_snr",
        "dish_boresight_azimuth_deg", "dish_boresight_elevation_deg",
        "weather_humidity", "weather_pressure", "weather_rain_rate",
    ],
    # Starlink throughput dl/ul
    (PROVIDER_STARLINK, "throughput_dl"): [
        "iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct",
        "dish_pop_ping_ms", "satellite_dtc", "satellite_handoff", "satellite_age_s",
        "distance_km", "elevation_deg", "azimuth_deg",
        "dish_obstruction_fraction", "dish_snr",
        "dish_boresight_azimuth_deg", "dish_boresight_elevation_deg",
        "weather_humidity", "weather_pressure", "weather_rain_rate",
    ],
    (PROVIDER_STARLINK, "throughput_ul"): [
        "iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct",
        "dish_pop_ping_ms", "satellite_dtc", "satellite_handoff", "satellite_age_s",
        "distance_km", "elevation_deg", "azimuth_deg",
        "dish_obstruction_fraction", "dish_snr",
        "dish_boresight_azimuth_deg", "dish_boresight_elevation_deg",
        "weather_humidity", "weather_pressure", "weather_rain_rate",
    ],
    # OneWeb dl / ul (unchanged)
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
    # Quectel latency (unchanged)
    (PROVIDER_QUECTEL, "latency"): [
        "ping_rtt_ms",
        "rsrp", "rsrq", "sinr",
        "weather_humidity", "weather_pressure", "weather_rain_rate",
    ],
    # Quectel latency_loss_throughput dl/ul
    (PROVIDER_QUECTEL, "latency_loss_throughput_dl"): [
        "iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct",
        "ping_rtt_ms",
        "rsrp", "rsrq", "sinr",
        "weather_humidity", "weather_pressure", "weather_rain_rate",
    ],
    (PROVIDER_QUECTEL, "latency_loss_throughput_ul"): [
        "iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct",
        "ping_rtt_ms",
        "rsrp", "rsrq", "sinr",
        "weather_humidity", "weather_pressure", "weather_rain_rate",
    ],
    # Quectel throughput dl/ul
    (PROVIDER_QUECTEL, "throughput_dl"): [
        "iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct",
        "rsrp", "rsrq", "sinr",
        "weather_humidity", "weather_pressure", "weather_rain_rate",
    ],
    (PROVIDER_QUECTEL, "throughput_ul"): [
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
    (PROVIDER_STARLINK, "latency"):                    ["ping_server_rtt_ms", "ping_dns_rtt_ms"],
    (PROVIDER_STARLINK, "latency_loss_throughput_dl"): ["iperf_throughput_mbps", "iperf_jitter_ms",
                                                        "iperf_loss_pct",
                                                        "ping_server_rtt_ms", "ping_dns_rtt_ms"],
    (PROVIDER_STARLINK, "latency_loss_throughput_ul"): ["iperf_throughput_mbps", "iperf_jitter_ms",
                                                        "iperf_loss_pct",
                                                        "ping_server_rtt_ms", "ping_dns_rtt_ms"],
    (PROVIDER_STARLINK, "throughput_dl"):              ["iperf_throughput_mbps", "iperf_jitter_ms",
                                                        "iperf_loss_pct"],
    (PROVIDER_STARLINK, "throughput_ul"):              ["iperf_throughput_mbps", "iperf_jitter_ms",
                                                        "iperf_loss_pct"],
    (PROVIDER_ONEWEB,   "dl"):                         ["throughput_mbps", "packet_loss_percent",
                                                        "jitter_ms", "rtt_mean"],
    (PROVIDER_ONEWEB,   "ul"):                         ["throughput_mbps", "packet_loss_percent",
                                                        "jitter_ms", "rtt_mean"],
    (PROVIDER_QUECTEL,  "latency"):                    ["ping_rtt_ms"],
    (PROVIDER_QUECTEL,  "latency_loss_throughput_dl"): ["iperf_throughput_mbps", "iperf_jitter_ms",
                                                        "iperf_loss_pct", "ping_rtt_ms"],
    (PROVIDER_QUECTEL,  "latency_loss_throughput_ul"): ["iperf_throughput_mbps", "iperf_jitter_ms",
                                                        "iperf_loss_pct", "ping_rtt_ms"],
    (PROVIDER_QUECTEL,  "throughput_dl"):              ["iperf_throughput_mbps", "iperf_jitter_ms",
                                                        "iperf_loss_pct"],
    (PROVIDER_QUECTEL,  "throughput_ul"):              ["iperf_throughput_mbps", "iperf_jitter_ms",
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

DEFAULT_TARGET: dict[tuple[str, str], str] = {
    k: v[0] for k, v in DEFAULT_TARGETS_BY_EXP.items()
}


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

FEATURE_GROUPS: dict[str, list[str]] = {
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
    "oneweb_signal":      [
        "sinr_db", "elevation_deg", "azimuth_deg",
    ],
    "oneweb_handover":    [
        "inter_satellite_ho_rate", "inter_beam_ho_rate",
        "beam_change", "satellite_change", "time_since_handover",
    ],
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
    experiment_type:  str                  = "throughput_dl"

    # Paths
    data_root:        str                  = "data/lsn-data"

    # Targets / features
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

        if self.targets is not None:
            self.targets = [aliases.get(t, t) for t in self.targets]
        elif self.target is not None:
            self.targets = [aliases.get(self.target, self.target)]
        else:
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

    @property
    def direction(self) -> Optional[str]:
        """Return 'dl', 'ul', or None for experiment types without a direction."""
        if self.experiment_type.endswith("_dl"):
            return "dl"
        if self.experiment_type.endswith("_ul"):
            return "ul"
        if self.experiment_type == "dl":
            return "dl"
        if self.experiment_type == "ul":
            return "ul"
        return None


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_session_files(cfg: DataConfig) -> list[Path]:
    """
    Locate all CSV files for the given provider/experiment_type.

    Layout (new):
        latency         : <provider_root>/latency/*.csv
        throughput_dl   : <provider_root>/throughput/dl/*.csv
        throughput_ul   : <provider_root>/throughput/ul/*.csv
        llt_dl          : <provider_root>/latency_loss_throughput/dl/*.csv
        llt_ul          : <provider_root>/latency_loss_throughput/ul/*.csv
        oneweb dl/ul    : <provider_root>/dl/*.csv  /  ul/*.csv  (unchanged)
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
    parsed = pd.to_datetime(series, utc=False, errors="coerce")
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed


def load_csv(path: Path, cfg: DataConfig) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        print(f"  [warn] could not read {path}: {exc}")
        return None

    ts_col = cfg.timestamp_col

    if ts_col in df.columns:
        df = df[df[ts_col] != ts_col].copy()
        df[ts_col] = _parse_timestamp(df[ts_col])
    else:
        print(f"  [warn] {path.name}: missing timestamp column '{ts_col}', skipping")
        return None

    _ensure_time_cols(df, cfg)

    skip = {ts_col, "hour", "dow"}
    if cfg.provider == PROVIDER_QUECTEL:
        skip |= {"cellid", "inner_cell", "base_station", "band", "weather_code_nws"}
    if cfg.provider == PROVIDER_ONEWEB:
        skip |= {"track_id"}

    for col in df.columns:
        if col in skip:
            continue
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if cfg.provider == PROVIDER_STARLINK and "dish_snr" in df.columns:
        df["dish_snr"] = pd.to_numeric(df["dish_snr"], errors="coerce")

    return df


def _ensure_time_cols(df: pd.DataFrame, cfg: DataConfig) -> None:
    ts_col = cfg.timestamp_col
    h_col  = cfg.hour_col
    d_col  = cfg.dow_col

    if h_col and h_col in df.columns:
        df["hour"] = pd.to_numeric(df[h_col], errors="coerce").astype("float32")
    else:
        df["hour"] = df[ts_col].dt.hour.astype("float32")

    if d_col and d_col in df.columns:
        df["dow"] = pd.to_numeric(df[d_col], errors="coerce").astype("float32")
    else:
        df["dow"] = df[ts_col].dt.dayofweek.astype("float32")


# ---------------------------------------------------------------------------
# Dataset  (unchanged from original — all logic below here is unmodified)
# ---------------------------------------------------------------------------

class ThroughputFileDataset(Dataset):
    """
    Sliding-window dataset supporting one or more prediction targets.
    (Unchanged from original.)
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

        self.feature_cols = [c for c in feature_cols if c in df.columns]
        self.target_cols  = target_cols

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

        if normalise == "zscore":
            if feature_stats is None:
                feature_stats = {
                    "mean": features.mean(0),
                    "std":  features.std(0) + 1e-8,
                }
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
        tgt_t = torch.from_numpy(tgt.copy())   # (output_len, num_targets)

        if self.output_len == 1:
            # Next-step mode: drop the time dimension entirely.
            # Result: (num_targets,) — clean 1-D target per sample.
            tgt_t = tgt_t.squeeze(0)
        # output_len > 1: keep (output_len, num_targets) for seq2seq trainers.

        return (
            torch.from_numpy(feat.copy()),
            torch.from_numpy(hour.copy()),
            torch.from_numpy(dow.copy()),
            tgt_t,
        )

    def inverse_transform(self, x: torch.Tensor, target_idx: int = 0) -> torch.Tensor:
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
# ExperimentDataset — adapter for legacy trainers (unchanged)
# ---------------------------------------------------------------------------

class ExperimentDataset(Dataset):
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

        # y is ONLY the target values at the first prediction step.
        # Shape: (num_targets,) for multi-target, (1,) for single-target.
        # NOTE: previously this was cat([x[-1,:], y_tgt]) which mixed the last
        # input timestep with the target — that was a legacy VAR-trainer design
        # that caused shape mismatches in LSTM/Transformer output heads.
        if tgt.dim() == 2:
            y = tgt[0, :].float()   # (num_targets,)
        else:
            y = tgt[0:1].float()    # (1,)
        return x, y

    def inverse_transform(self, x: torch.Tensor, target_idx: int = 0) -> torch.Tensor:
        """
        Denormalise a (B, num_targets) or (B,) tensor of model predictions.
        Each column is denormalised with its own target stats.
        """
        if not self.target_stats or self.normalise not in ("zscore", "minmax"):
            return x
        x = x.float()
        # Multi-target: denormalise each column independently
        if x.dim() == 2 and x.shape[1] == len(self.target_cols) and len(self.target_cols) > 1:
            result = x.clone()
            for i, col in enumerate(self.target_cols):
                stats = self.target_stats.get(col)
                if stats is None:
                    continue
                if self.normalise == "zscore":
                    result[:, i] = result[:, i] * stats["std"] + stats["mean"]
                else:
                    result[:, i] = result[:, i] * (stats["max"] - stats["min"]) + stats["min"]
            return result
        # Single-target (or explicit target_idx)
        if self.target_cols:
            col = self.target_cols[target_idx]
        else:
            col = None
        stats = self.target_stats.get(col) if col else None
        if stats is None:
            return x
        if self.normalise == "zscore":
            return x * stats["std"] + stats["mean"]
        return x * (stats["max"] - stats["min"]) + stats["min"]


# ---------------------------------------------------------------------------
# Internal helpers (unchanged)
# ---------------------------------------------------------------------------

def _align_stats_array(
    arr: Optional[np.ndarray],
    n: int,
    fill: float = 0.0,
) -> np.ndarray:
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
    missing_targets = [tc for tc in target_cols if tc not in df.columns]
    if missing_targets:
        print(f"  [warn] {path.name}: missing target(s) {missing_targets}, skipping")
        return None

    min_rows = cfg.input_len + cfg.output_len + 1
    if len(df) < min_rows:
        print(f"  [warn] {path.name}: only {len(df)} rows (need {min_rows}), skipping")
        return None

    missing_feat = [fc for fc in feature_cols if fc not in df.columns]
    if missing_feat:
        print(f"  [info] {path.name}: features absent (will be skipped): {missing_feat}")

    return df


# ---------------------------------------------------------------------------
# Public build functions (unchanged)
# ---------------------------------------------------------------------------

def build_dataloaders(cfg: DataConfig):
    target_cols  = cfg.targets
    feature_cols = cfg.features

    subdir, suffix = FILE_LAYOUT[(cfg.provider, cfg.experiment_type)]
    print(f"\n  Provider    : {cfg.provider}")
    print(f"  Exp type    : {cfg.experiment_type}  (suffix: *{suffix})")
    print(f"  Direction   : {cfg.direction or 'n/a'}")
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
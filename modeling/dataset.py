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
  "latency"                    -> starlink/latency/starlink_*_latency.csv
  "latency_loss_throughput_dl" -> starlink/latency_loss_throughput/dl/starlink_*_latency_loss_throughput_dl.csv
  "latency_loss_throughput_ul" -> starlink/latency_loss_throughput/ul/starlink_*_latency_loss_throughput_ul.csv
  "throughput_dl"              -> starlink/throughput/dl/starlink_*_throughput_dl.csv
  "throughput_ul"              -> starlink/throughput/ul/starlink_*_throughput_ul.csv

OneWeb:
  "dl"                         -> oneweb/dl/ex_*_DL.csv
  "ul"                         -> oneweb/ul/ex_*_UL.csv

Quectel:
  "latency"                    -> quectel/latency/quectel_*_latency.csv
  "latency_loss_throughput_dl" -> quectel/latency_loss_throughput/dl/quectel_*_latency_loss_throughput_dl.csv
  "latency_loss_throughput_ul" -> quectel/latency_loss_throughput/ul/quectel_*_latency_loss_throughput_ul.csv
  "throughput_dl"              -> quectel/throughput/dl/quectel_*_throughput_dl.csv
  "throughput_ul"              -> quectel/throughput/ul/quectel_*_throughput_ul.csv

IO Mode contract
----------------
MIMO  (Multi-Input Multi-Output):  target shape (output_len, num_targets)
        Full multi-horizon, multi-target prediction.
        Models: LSTM, Transformer, Seq2Seq, Attention, StarNet

MISO  (Multi-Input Single-Output): target shape (output_len,)
        Full multi-horizon, single target (target_cols[0] only).
        Models: GP, Autoencoder, MLP baselines, statistical

SISO  (Single-Input Single-Output): target shape () scalar
        Single-step, single-target. output_len forced to 1.
        Models: ARIMA

Multi-target notes:
  - targets is a list of column names or shorthand keys.
  - Each target is normalised independently; target_stats is keyed by col name.
  - MISO / SISO silently retain only targets[0] with a UserWarning.
  - MIMO with a single target returns shape (output_len, 1) — not squeezed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset


# ---------------------------------------------------------------------------
# IO Mode registry
# ---------------------------------------------------------------------------

IO_MODE_MIMO = "MIMO"
IO_MODE_MISO = "MISO"
IO_MODE_SISO = "SISO"
IO_MODES = (IO_MODE_MIMO, IO_MODE_MISO, IO_MODE_SISO)

# Maps model key (as used in run.py / argparse) -> IO mode.
# Import this in run.py and trainer dispatch to resolve the correct mode.
MODEL_IO_MODE: dict[str, str] = {
    # MIMO — multi-target, full horizon
    "lstm":        IO_MODE_MIMO,
    "transformer": IO_MODE_MIMO,
    "seq2seq":     IO_MODE_MIMO,
    "attention":   IO_MODE_MIMO,
    "starnet":     IO_MODE_MIMO,
    # MISO — single-target, full horizon
    "gp":          IO_MODE_MISO,
    "autoencoder": IO_MODE_MISO,
    "mlp":         IO_MODE_MISO,
    "ridge":       IO_MODE_MISO,
    "xgboost":     IO_MODE_MISO,
    "persistence": IO_MODE_MISO,
    # SISO — single-target, single-step
    "arima":       IO_MODE_SISO,
}

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

FILE_LAYOUT: dict[tuple[str, str], tuple[str, str]] = {
    # Starlink
    (PROVIDER_STARLINK, "latency"):                    ("latency",                    "_latency.csv"),
    (PROVIDER_STARLINK, "latency_loss_throughput_dl"): ("latency_loss_throughput/dl", "_latency_loss_throughput_dl.csv"),
    (PROVIDER_STARLINK, "latency_loss_throughput_ul"): ("latency_loss_throughput/ul", "_latency_loss_throughput_ul.csv"),
    (PROVIDER_STARLINK, "throughput_dl"):              ("throughput/dl",              "_throughput_dl.csv"),
    (PROVIDER_STARLINK, "throughput_ul"):              ("throughput/ul",              "_throughput_ul.csv"),
    # OneWeb
    (PROVIDER_ONEWEB,   "dl"):                         ("dl",                         "_DL.csv"),
    (PROVIDER_ONEWEB,   "ul"):                         ("ul",                         "_UL.csv"),
    # Quectel
    (PROVIDER_QUECTEL,  "latency"):                    ("latency",                    "_latency.csv"),
    (PROVIDER_QUECTEL,  "latency_loss_throughput_dl"): ("latency_loss_throughput/dl", "_latency_loss_throughput_dl.csv"),
    (PROVIDER_QUECTEL,  "latency_loss_throughput_ul"): ("latency_loss_throughput/ul", "_latency_loss_throughput_ul.csv"),
    (PROVIDER_QUECTEL,  "throughput_dl"):              ("throughput/dl",              "_throughput_dl.csv"),
    (PROVIDER_QUECTEL,  "throughput_ul"):              ("throughput/ul",              "_throughput_ul.csv"),
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
    (PROVIDER_STARLINK, "latency"): _SL_COMMON + [
        "ping_server_rtt_ms", "ping_dns_rtt_ms",
    ],
    (PROVIDER_STARLINK, "latency_loss_throughput_dl"): _SL_COMMON + [
        "ping_server_rtt_ms", "ping_dns_rtt_ms",
        "iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct",
    ],
    (PROVIDER_STARLINK, "latency_loss_throughput_ul"): _SL_COMMON + [
        "ping_server_rtt_ms", "ping_dns_rtt_ms",
        "iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct",
    ],
    (PROVIDER_STARLINK, "throughput_dl"): _SL_COMMON + [
        "iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct",
    ],
    (PROVIDER_STARLINK, "throughput_ul"): _SL_COMMON + [
        "iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct",
    ],
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
    (PROVIDER_QUECTEL, "latency"): [
        "ping_rtt_ms",
        "rsrp", "rsrq", "sinr",
        "weather_humidity", "weather_pressure", "weather_rain_rate",
    ],
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
    (PROVIDER_STARLINK, "latency"): [
        "ping_server_rtt_ms", "ping_dns_rtt_ms",
        "dish_pop_ping_ms", "satellite_dtc", "satellite_handoff", "satellite_age_s",
        "distance_km", "elevation_deg", "azimuth_deg",
        "dish_obstruction_fraction", "dish_snr",
        "dish_boresight_azimuth_deg", "dish_boresight_elevation_deg",
        "weather_humidity", "weather_pressure", "weather_rain_rate",
    ],
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
    (PROVIDER_QUECTEL, "latency"): [
        "ping_rtt_ms",
        "rsrp", "rsrq", "sinr",
        "weather_humidity", "weather_pressure", "weather_rain_rate",
    ],
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
        "iperf":  "iperf_throughput_mbps",
        "jitter": "iperf_jitter_ms",
        "loss":   "iperf_loss_pct",
        "rtt":    "ping_server_rtt_ms",
        "dns":    "ping_dns_rtt_ms",
        "dl":     "dish_dl_throughput_mbps",
        "ul":     "dish_ul_throughput_mbps",
        "pop":    "dish_pop_ping_ms",
        "snr":    "dish_snr",
    },
    PROVIDER_ONEWEB: {
        "tput":   "throughput_mbps",
        "loss":   "packet_loss_percent",
        "jitter": "jitter_ms",
        "rtt":    "rtt_mean",
        "sinr":   "sinr_db",
    },
    PROVIDER_QUECTEL: {
        "iperf":  "iperf_throughput_mbps",
        "jitter": "iperf_jitter_ms",
        "loss":   "iperf_loss_pct",
        "rtt":    "ping_rtt_ms",
        "sinr":   "sinr",
        "rsrp":   "rsrp",
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
    provider:        str                 = PROVIDER_STARLINK
    experiment_type: str                 = "throughput_dl"

    # Paths
    data_root:       str                 = "data/lsn-data"

    # Targets / features
    targets:         Optional[list[str]] = None
    features:        Optional[list[str]] = None

    # Window / split
    input_len:       int                 = 30
    output_len:      int                 = 15
    train_frac:      float               = 0.70
    val_frac:        float               = 0.15

    # Normalisation
    normalise:       str                 = "zscore"

    # IO mode — set automatically from MODEL_IO_MODE in run.py,
    # or override explicitly here.
    io_mode:         str                 = IO_MODE_MIMO

    # DataLoader
    batch_size:      int                 = 512
    num_workers:     int                 = 0
    seed:            int                 = 42

    # Legacy single-target compat
    target:          Optional[str]       = None

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {PROVIDERS}, got '{self.provider}'")
        valid_exp = VALID_EXPERIMENT_TYPES[self.provider]
        if self.experiment_type not in valid_exp:
            raise ValueError(
                f"experiment_type '{self.experiment_type}' invalid for provider "
                f"'{self.provider}'. Choose from {valid_exp}"
            )
        if self.io_mode not in IO_MODES:
            raise ValueError(f"io_mode must be one of {IO_MODES}, got '{self.io_mode}'")
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

        # MISO / SISO only ever operate on the primary target.
        if self.io_mode in (IO_MODE_MISO, IO_MODE_SISO) and len(self.targets) > 1:
            warnings.warn(
                f"io_mode='{self.io_mode}' uses only a single target. "
                f"Retaining targets[0]='{self.targets[0]}' and dropping "
                f"{self.targets[1:]}. Pass targets=[single_col] to suppress this warning.",
                UserWarning, stacklevel=2,
            )
            self.targets = [self.targets[0]]

        # SISO forces output_len=1.
        if self.io_mode == IO_MODE_SISO and self.output_len != 1:
            warnings.warn(
                f"io_mode='SISO' requires output_len=1. "
                f"Overriding output_len={self.output_len} -> 1.",
                UserWarning, stacklevel=2,
            )
            self.output_len = 1

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
        if self.experiment_type.endswith("_dl"):
            return "dl"
        if self.experiment_type.endswith("_ul"):
            return "ul"
        if self.experiment_type == "dl":
            return "dl"
        if self.experiment_type == "ul":
            return "ul"
        return None

    @property
    def target_shape(self) -> tuple:
        """
        Canonical shape of the target tensor returned by __getitem__:

          MIMO -> (output_len, num_targets)
          MISO -> (output_len,)
          SISO -> ()   scalar
        """
        if self.io_mode == IO_MODE_MIMO:
            return (self.output_len, self.num_targets)
        if self.io_mode == IO_MODE_MISO:
            return (self.output_len,)
        return ()   # SISO

    def __repr__(self) -> str:
        return (
            f"DataConfig("
            f"provider={self.provider!r}, "
            f"experiment_type={self.experiment_type!r}, "
            f"io_mode={self.io_mode!r}, "
            f"targets={self.targets}, "
            f"input_len={self.input_len}, "
            f"output_len={self.output_len}, "
            f"target_shape={self.target_shape}, "
            f"normalise={self.normalise!r}"
            f")"
        )


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_session_files(cfg: DataConfig) -> list[Path]:
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
# Dataset
# ---------------------------------------------------------------------------

class ThroughputFileDataset(Dataset):
    """
    Sliding-window dataset supporting MIMO, MISO, and SISO output contracts.

    io_mode controls the shape of the target tensor returned by __getitem__:

      MIMO  (default) : (output_len, num_targets)
            Full multi-horizon, multi-target. Used by LSTM / Transformer /
            Seq2Seq / Attention / StarNet.

      MISO            : (output_len,)
            Full multi-horizon, primary target only (target_cols[0]).
            Used by GP, Autoencoder, MLP baselines, and statistical models.

      SISO            : scalar tensor shape ()
            Single-step, single-target. DataConfig forces output_len=1.
            Used by ARIMA wrappers fed through the PyTorch pipeline.

    Storage layout
    --------------
    Windows are always stored as (output_len, num_targets) regardless of
    io_mode, so that a single dataset object can serve multiple modes if
    needed. The mode-specific slicing happens in __getitem__, not at
    construction time — this avoids ConcatDataset incompatibilities and
    keeps memory cost independent of mode.

    actual_feature_cols
    -------------------
    The list of feature columns actually present in the supplied DataFrame
    after intersection with feature_cols. Callers should read this after
    construction to know the true input dimensionality (dim 1 of the
    feature tensor), rather than assuming len(feature_cols).
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
        io_mode: str = IO_MODE_MIMO,
    ) -> None:
        if io_mode not in IO_MODES:
            raise ValueError(f"io_mode must be one of {IO_MODES}, got '{io_mode}'")

        self.input_len   = input_len
        self.output_len  = output_len
        self.normalise   = normalise
        self.io_mode     = io_mode
        self.num_targets = len(target_cols)
        self.target_cols = target_cols

        # Intersect requested features with columns present in the DataFrame.
        # Store the actual list so callers can read the true input dimensionality.
        self.actual_feature_cols: list[str] = [c for c in feature_cols if c in df.columns]

        ts_candidates = [c for c in df.columns if "timestamp" in c.lower()]
        if ts_candidates:
            df = df.sort_values(ts_candidates[0]).reset_index(drop=True)

        df[self.actual_feature_cols] = (
            df[self.actual_feature_cols]
            .apply(pd.to_numeric, errors="coerce")
            .ffill().bfill().fillna(0)
        )
        for tc in target_cols:
            if tc in df.columns:
                df[tc] = (
                    pd.to_numeric(df[tc], errors="coerce")
                    .ffill().bfill().fillna(0)
                )

        features = df[self.actual_feature_cols].values.astype(np.float32)
        targets  = df[target_cols].values.astype(np.float32)
        hours    = df["hour"].values.astype(np.float32)
        dows     = df["dow"].values.astype(np.float32)

        # ---- Feature normalisation ----------------------------------------
        if normalise == "zscore":
            if feature_stats is None:
                warnings.warn(
                    "ThroughputFileDataset: feature_stats=None with normalise='zscore'. "
                    "Stats computed from this file alone. "
                    "Pass feature_stats from build_dataloaders() during training.",
                    UserWarning, stacklevel=2,
                )
                feature_stats = {
                    "mean": features.mean(0),
                    "std":  features.std(0) + 1e-8,
                }
            mean = _align_stats_array(feature_stats.get("mean"), features.shape[1])
            std  = _align_stats_array(feature_stats.get("std"),  features.shape[1], fill=1.0)
            features = (features - mean) / std

        elif normalise == "minmax":
            if feature_stats is None:
                warnings.warn(
                    "ThroughputFileDataset: feature_stats=None with normalise='minmax'. "
                    "Stats computed from this file alone. "
                    "Pass feature_stats from build_dataloaders() during training.",
                    UserWarning, stacklevel=2,
                )
                feature_stats = {
                    "min": features.min(0),
                    "max": features.max(0) + 1e-8,
                }
            fmin = _align_stats_array(feature_stats.get("min"), features.shape[1])
            fmax = _align_stats_array(feature_stats.get("max"), features.shape[1], fill=1.0)
            features = (features - fmin) / (fmax - fmin + 1e-8)

        self.feature_stats = feature_stats

        # ---- Target normalisation -----------------------------------------
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

        # ---- Build sliding windows ----------------------------------------
        # Always stored as (output_len, num_targets) regardless of io_mode.
        # Mode-specific slicing is deferred to __getitem__.
        total = input_len + output_len
        N = len(df)
        self.windows: list[tuple] = []
        for start in range(N - total + 1):
            self.windows.append((
                features[start             : start + input_len],
                hours   [start             : start + input_len],
                dows    [start             : start + input_len],
                normed  [start + input_len : start + total],
            ))

    # Keep the old attribute name as a read-only alias so any existing code
    # that references ds.feature_cols still works.
    @property
    def feature_cols(self) -> list[str]:
        return self.actual_feature_cols

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        feat, hour, dow, tgt = self.windows[idx]
        # tgt is always stored as (output_len, num_targets).

        if self.io_mode == IO_MODE_MIMO:
            # (output_len, num_targets)
            tgt_t = torch.from_numpy(tgt.copy())

        elif self.io_mode == IO_MODE_MISO:
            # (output_len,) — primary target only, full horizon
            tgt_t = torch.from_numpy(tgt[:, 0].copy())

        else:  # SISO
            # scalar — primary target, single step.
            # DataConfig forces output_len=1, so tgt is (1, num_targets).
            tgt_t = torch.tensor(tgt[0, 0], dtype=torch.float32)

        return (
            torch.from_numpy(feat.copy()),
            torch.from_numpy(hour.copy()),
            torch.from_numpy(dow.copy()),
            tgt_t,
        )

    # ------------------------------------------------------------------
    # Inverse-transform helpers
    # ------------------------------------------------------------------

    def inverse_transform(self, x: torch.Tensor, target_idx: int = 0) -> torch.Tensor:
        """Inverse-normalise a tensor for the target at target_idx."""
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
        Inverse-normalise a MIMO tensor of shape (..., num_targets).
        For MISO/SISO callers use inverse_transform(x, 0) instead.
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

    def assert_target_shape(self, batch_target: torch.Tensor) -> None:
        """
        Validate that a batch target tensor matches the expected io_mode shape.

        Call once after the first batch to catch trainer / dataset mode mismatches
        early rather than letting silent broadcasting hide bugs downstream.

        Example
        -------
        for feat, hour, dow, tgt in train_loader:
            ds.assert_target_shape(tgt)   # raises if shape is wrong
            break
        """
        # batch_target has a leading batch dimension; strip it for comparison.
        got = tuple(batch_target.shape[1:])
        if self.io_mode == IO_MODE_MIMO:
            expected = (self.output_len, self.num_targets)
        elif self.io_mode == IO_MODE_MISO:
            expected = (self.output_len,)
        else:  # SISO
            expected = ()

        if got != expected:
            raise AssertionError(
                f"Target shape mismatch for io_mode='{self.io_mode}': "
                f"expected {expected}, got {got}. "
                f"Check that DataConfig.io_mode matches the model's IO mode."
            )


# ---------------------------------------------------------------------------
# Internal helpers
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
# Public build functions
# ---------------------------------------------------------------------------

def build_dataloaders(cfg: DataConfig):
    """
    Build train/val/test DataLoaders honouring cfg.io_mode.

    The io_mode is threaded through to every ThroughputFileDataset so that
    __getitem__ returns the correct target shape for the calling model.

    Returns
    -------
    train_loader, val_loader, test_loader, feature_stats, target_stats, actual_feature_cols

    Notes
    -----
    The returned ``actual_feature_cols`` reflects which feature columns were
    actually present across the training files after intersection.  This may be
    shorter than ``cfg.features`` when some columns are absent in the data, so
    always use this return value — not ``cfg.features`` — to size model input
    layers (e.g. ``in_channels = len(actual_feature_cols)``).
    """
    target_cols  = cfg.targets
    feature_cols = cfg.features

    subdir, suffix = FILE_LAYOUT[(cfg.provider, cfg.experiment_type)]
    print(f"\n  Provider    : {cfg.provider}")
    print(f"  Exp type    : {cfg.experiment_type}  (suffix: *{suffix})")
    print(f"  Direction   : {cfg.direction or 'n/a'}")
    print(f"  IO mode     : {cfg.io_mode}  (target shape: {cfg.target_shape})")
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

    # ---- Compute global stats from the training split --------------------
    feature_stats: Optional[dict] = None
    target_stats:  dict           = {}

    if cfg.normalise in ("zscore", "minmax"):
        tr      = pd.concat(train_dfs, ignore_index=True)
        avail_feat = [f for f in feature_cols if f in tr.columns]
        tr_feat = (
            tr[avail_feat]
            .apply(pd.to_numeric, errors="coerce")
            .ffill().bfill().fillna(0)
            .values.astype(np.float32)
        )

        if cfg.normalise == "zscore":
            feature_stats = {"mean": tr_feat.mean(0), "std": tr_feat.std(0) + 1e-8}
        else:
            feature_stats = {"min": tr_feat.min(0), "max": tr_feat.max(0) + 1e-8}

        for tc in target_cols:
            tr_tgt = (
                pd.to_numeric(tr[tc], errors="coerce")
                .ffill().bfill().fillna(0)
                .values.astype(np.float32)
            )
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

    # ---- Build dataset objects -------------------------------------------
    # Collect actual_feature_cols from the first successfully-built dataset
    # so callers know the true input dimensionality.
    _actual_feature_cols: list[str] = []

    def make_ds(df_list: list[pd.DataFrame]):
        nonlocal _actual_feature_cols
        datasets = []
        for df in df_list:
            ds = ThroughputFileDataset(
                df, feature_cols, target_cols,
                cfg.input_len, cfg.output_len,
                feature_stats=feature_stats,
                target_stats=target_stats if target_stats else None,
                normalise=cfg.normalise,
                io_mode=cfg.io_mode,
            )
            if len(ds) > 0:
                if not _actual_feature_cols:
                    _actual_feature_cols = ds.actual_feature_cols
                datasets.append(ds)
        return ConcatDataset(datasets) if datasets else None

    train_ds = make_ds(train_dfs)
    val_ds   = make_ds(val_dfs)
    test_ds  = make_ds(test_dfs)

    if not _actual_feature_cols:
        _actual_feature_cols = [f for f in feature_cols
                                 if any(f in df.columns for df in dfs)]

    missing = set(feature_cols) - set(_actual_feature_cols)
    if missing:
        warnings.warn(
            f"build_dataloaders: {len(missing)} requested feature(s) were absent "
            f"from all data files and will be excluded from the model input: "
            f"{sorted(missing)}. "
            f"Use the returned actual_feature_cols to size model input layers.",
            UserWarning, stacklevel=2,
        )

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
    print(f"  Input dim   : {len(_actual_feature_cols)} feature(s)")

    return (
        to_loader(train_ds, True),
        to_loader(val_ds,   False),
        to_loader(test_ds,  False),
        feature_stats,
        target_stats,
        _actual_feature_cols,   # actual, not cfg.features
    )
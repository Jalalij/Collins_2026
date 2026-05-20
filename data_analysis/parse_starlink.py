"""
parse_starlink.py
-----------------

Parses Starlink experiment sessions into 3 dataset types:

1) latency
   - Uses baseline/
   - ~200ms native ping cadence
   - Output: parsed/starlink/latency/<session>_latency.csv

2) latency_loss_throughput
   - Uses udp_dl_100 + udp_ul_10
   - 1s native iperf cadence
   - Output: parsed/starlink/latency_loss_throughput/<session>_llt.csv

3) throughput
   - Uses iperf_udp_*/ or udp_dl_*/udp_ul_*
   - 1s native iperf cadence
   - Output: parsed/starlink/throughput/<session>_throughput.csv

Timing strategy
---------------
Every source file has its own sub-second timestamp cadence. Rather than
snapping everything to an artificial grid (which invents measurement
co-occurrences that did not happen), we:

  1. Parse each file keeping its exact original timestamps as the index.
  2. Union all timestamps into one index — every real measurement event
     gets its own row.
  3. Fill gaps per signal type:
       - ping RTT            : linear interpolation (continuous signal)
       - iperf / dish / sat  : forward-fill (value valid until next reading)
       - weather             : forward-fill (changes over minutes)
       - satellite_handoff   : 0 between known events (binary)
       - satellite_age_s     : recomputed deterministically from wall-clock
       - ZERO_FILL_COLS      : 0 (recently added, older sessions = no obstruction)
  4. Clip to [max(all starts), min(all ends)] overlap window.
  5. Convert to US Central, strip tz suffix.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CENTRAL_TZ    = "America/Chicago"
SAT_CYCLE_LEN = 15
SAT_ANCHOR_SECONDS = {12, 27, 42, 57}

# Cols added recently to data collection: null in older sessions = 0
ZERO_FILL_COLS = {"dish_obstruction_fraction", "dish_time_obstructed", "dish_valid_s"}

# Ping RTT cols: filled by linear interpolation
INTERP_COLS = {"ping_tunnel_rtt_ms", "ping_internet_rtt_ms",
               "ping_server_rtt_ms", "ping_dns_rtt_ms"}

# Binary event cols: absence = 0
BINARY_ZERO_COLS = {"satellite_handoff", "satellite_dtc"}

# Recomputed after alignment from wall-clock anchor
RECOMPUTED_COLS = {"satellite_age_s"}

SCHEMA = {
    # Integers filled with -1 when missing (true unknown)
    "integers_neg1": [
        "satellite_id", "satellite_age_s",
        "hour_cst", "dow_cst", "dish_gps_sats",
    ],
    # Booleans: event cols — absence = 0, never -1
    "integers_bool": [
        "satellite_handoff", "satellite_dtc",
    ],
    "rounding": {
        4: ["iperf_jitter_ms", "ping_tunnel_rtt_ms", "ping_internet_rtt_ms",
            "ping_server_rtt_ms", "ping_dns_rtt_ms", "dish_pop_ping_ms"],
        5: ["iperf_loss_pct"],
        3: ["iperf_throughput_mbps", "dish_dl_throughput_mbps", "dish_ul_throughput_mbps",
            "dish_snr", "dish_obstruction_fraction",
            "dish_boresight_azimuth_deg", "dish_boresight_elevation_deg", "dish_tilt_angle_deg",
            "distance_km", "elevation_deg", "azimuth_deg",
            "weather_temp_f", "weather_pressure", "weather_wind_speed",
            "weather_rain_rate", "weather_humidity", "weather_wind_dir", "weather_visibility"],
        6: ["sat_lat", "sat_lon", "sat_alt_km",
            "dish_latitude", "dish_longitude", "dish_altitude_m"],
    },
}


def apply_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce integer types and round floats per SCHEMA."""
    # True unknown integers: missing → -1
    for col in SCHEMA["integers_neg1"]:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(-1)
                .astype("int64")
            )
    # Boolean event cols: absence = 0, never -1
    for col in SCHEMA["integers_bool"]:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(0)
                .astype("int64")
            )
    # Float rounding
    for decimals, cols in SCHEMA["rounding"].items():
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").round(decimals)
    return df



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def trim_edges(df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    if len(df) <= n * 2:
        return df
    return df.iloc[n:-n]


def report_nulls(df: pd.DataFrame, label: str, folder: str) -> None:
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if null_counts.empty:
        return
    total = len(df)
    print(f"  [nulls] {folder} | {label}")
    for col, count in null_counts.items():
        print(f"         {col}: {count}/{total} rows ({100*count/total:.1f}%)")


def recompute_satellite_age(index: pd.DatetimeIndex,
                             anchor_ts: pd.Timestamp) -> np.ndarray:
    """
    Deterministically recompute satellite_age_s for every timestamp.
    anchor_ts = first row of satellite.csv, which is always at a
    :12/:27/:42/:57 wall-clock second boundary = age 1.
    """
    elapsed = (index - anchor_ts).total_seconds()
    return (np.floor(elapsed) % SAT_CYCLE_LEN + 1).astype(int)


def report_missing_cols(merged: pd.DataFrame, folder: Path,
                        include_iperf: bool, include_ping: bool) -> None:
    """Warn only about columns that should exist given this folder's file set."""
    expected = []

    # Always present (satellite + dish + weather are in every folder type)
    expected += [
        "satellite_id", "satellite_dtc", "satellite_handoff", "satellite_age_s",
        "dish_dl_throughput_mbps", "dish_ul_throughput_mbps", "dish_pop_ping_ms",
        "dish_obstruction_fraction", "dish_snr", "dish_gps_sats",
        "dish_boresight_azimuth_deg", "dish_boresight_elevation_deg",
        "dish_tilt_angle_deg", "dish_latitude", "dish_longitude", "dish_altitude_m",
        "weather_temp_f", "weather_humidity", "weather_pressure", "weather_wind_speed",
        "weather_wind_dir", "weather_rain_rate", "weather_visibility", "weather_code_nws",
    ]

    if include_iperf:
        expected += ["iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct"]

    if include_ping:
        # Warn only for ping files that actually exist in the folder
        ping_file_map = {
            "ping_tunnel_rtt_ms":   ["ping.csv", "ping_tunnel.csv"],
            "ping_server_rtt_ms":   ["ping_server.csv"],
            "ping_dns_rtt_ms":      ["ping_dns.csv"],
            "ping_internet_rtt_ms": ["ping_1_1_1_1.csv"],
        }
        for col, filenames in ping_file_map.items():
            if any((folder / fn).exists() for fn in filenames):
                expected.append(col)

    missing = [c for c in expected if c not in merged.columns]
    if missing:
        print(f"  [missing cols] {folder.name}: {', '.join(missing)}")


# ---------------------------------------------------------------------------
# File parsers  (preserve exact original timestamps)
# ---------------------------------------------------------------------------

def parse_ping_csv(path: Path, col_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = to_utc(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[["rtt_ms"]].rename(columns={"rtt_ms": col_name})
    df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
    return df


def parse_iperf_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = to_utc(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    keep = {
        "throughput_mbps":     "iperf_throughput_mbps",
        "jitter_ms":           "iperf_jitter_ms",
        "packet_loss_percent": "iperf_loss_pct",
    }
    existing = {k: v for k, v in keep.items() if k in df.columns}
    df = df.rename(columns=existing)
    for v in keep.values():
        if v not in df.columns:
            df[v] = np.nan
        else:
            df[v] = pd.to_numeric(df[v], errors="coerce")
    return df[list(keep.values())]


def parse_iperf_server_txt(path: Path) -> pd.DataFrame:
    pattern = re.compile(
        r"\[\s*\d+\]\s+"
        r"(\d+\.\d+)-(\d+\.\d+)\s+sec"
        r".*?"
        r"([\d.]+)\s+Mbits/sec"
        r"\s+([\d.]+)\s+ms"
        r"\s+(\d+)/(\d+)"
        r"\s+\(([\d.]+)%\)"
    )
    rows = []
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                rows.append({
                    "interval_idx":           int(round(float(m.group(1)))),
                    "server_throughput_mbps":  float(m.group(3)),
                    "server_jitter_ms":        float(m.group(4)),
                    "server_loss_pct":         float(m.group(7)),
                })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("interval_idx")


def merge_ul_server(iperf_df: pd.DataFrame,
                    server_df: pd.DataFrame) -> pd.DataFrame:
    """Replace UL throughput/jitter/loss with server-side received values."""
    if server_df.empty:
        return iperf_df
    iperf_df = iperf_df.copy()
    iperf_df["interval_idx"] = range(len(iperf_df))
    merged = (
        iperf_df.reset_index()
        .merge(server_df.reset_index(), on="interval_idx", how="left")
        .set_index("timestamp")
    )
    merged["iperf_throughput_mbps"] = merged["server_throughput_mbps"]
    merged["iperf_jitter_ms"]       = merged["server_jitter_ms"]
    merged["iperf_loss_pct"]        = merged["server_loss_pct"]
    return merged[["iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct"]]


def parse_satellite_csv(path: Path):
    """Returns (df, anchor_ts). anchor_ts used to recompute satellite_age_s."""
    df = pd.read_csv(path)
    df["timestamp"] = to_utc(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    raw_id = df["satellite_id"].astype(str)
    df["satellite_dtc"] = raw_id.str.contains(r"\[DTC\]", regex=True).astype(int)
    df["satellite_id"] = (
        raw_id
        .str.replace(r"\s*\[DTC\]", "", regex=True)
        .str.replace("STARLINK-", "", regex=False)
        .str.strip()
        .astype(int)
    )
    df["satellite_handoff"] = (
        df["satellite_id"].diff().ne(0) & df["satellite_id"].notna()
    ).astype(int)
    df.iloc[0, df.columns.get_loc("satellite_handoff")] = 0
    df["satellite_age_s"] = 0   # placeholder; recomputed after alignment

    anchor_ts = df.index[0]

    pos_cols     = ["distance_km", "elevation_deg", "azimuth_deg",
                    "sat_lat", "sat_lon", "sat_alt_km"]
    pos_existing = [c for c in pos_cols if c in df.columns]
    base_cols    = ["satellite_id", "satellite_dtc",
                    "satellite_handoff", "satellite_age_s"]
    return df[base_cols + pos_existing], anchor_ts


def parse_starlink_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = to_utc(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    keep = {
        "dl_throughput_mbps":      "dish_dl_throughput_mbps",
        "ul_throughput_mbps":      "dish_ul_throughput_mbps",
        "pop_ping_latency_ms":     "dish_pop_ping_ms",
        "obstruction_fraction":    "dish_obstruction_fraction",
        "snr_above_noise_floor":   "dish_snr",
        "gps_sats":                "dish_gps_sats",
        "boresight_azimuth_deg":   "dish_boresight_azimuth_deg",
        "boresight_elevation_deg": "dish_boresight_elevation_deg",
        "tilt_angle_deg":          "dish_tilt_angle_deg",
        "latitude":                "dish_latitude",
        "longitude":               "dish_longitude",
        "altitude_m":              "dish_altitude_m",
    }
    existing = {k: v for k, v in keep.items() if k in df.columns}
    df = df.rename(columns=existing)[list(existing.values())]
    return df.apply(pd.to_numeric, errors="coerce")


def parse_weather_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = to_utc(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    keep = {
        "Temperature":    "weather_temp_f",
        "Humidity":       "weather_humidity",
        "Pressure":       "weather_pressure",
        "WindSpeed":      "weather_wind_speed",
        "WindDirection":  "weather_wind_dir",
        "RainRate":       "weather_rain_rate",
        "MORVisibility":  "weather_visibility",
        "WeatherCodeNWS": "weather_code_nws",
    }
    existing = {k: v for k, v in keep.items() if k in df.columns}
    return df.rename(columns=existing)[list(existing.values())]


# ---------------------------------------------------------------------------
# Alignment and filling
# ---------------------------------------------------------------------------

def align_and_fill(frames: dict,
                   sat_anchor_ts,
                   folder: Path):
    """
    Union all frame timestamps. Fill per signal type. Clip to overlap window.
    """
    if not frames:
        return None

    # Overlap window
    t_start = max(df.index.min() for df in frames.values())
    t_end   = min(df.index.max() for df in frames.values())
    if t_start >= t_end:
        print(f"  [skip] empty overlap window in {folder.name}")
        return None

    frames = {k: v.loc[t_start:t_end] for k, v in frames.items()}

    # Union of all timestamps — no snapping, no invented grid
    all_ts = sorted(set().union(*[set(df.index) for df in frames.values()]))
    union_index = pd.DatetimeIndex(all_ts)

    merged = pd.DataFrame(index=union_index)
    merged.index.name = "timestamp"

    for name, df in frames.items():
        report_nulls(df, name, str(folder))
        merged = merged.join(df, how="left")

    # Per-signal filling
    for col in merged.columns:

        if col in RECOMPUTED_COLS:
            # Deterministic from anchor timestamp — no fill needed
            if sat_anchor_ts is not None:
                merged[col] = recompute_satellite_age(merged.index, sat_anchor_ts)
            else:
                merged[col] = merged[col].ffill().bfill().fillna(0)

        elif col in BINARY_ZERO_COLS:
            # Binary events: absence between events = 0
            merged[col] = merged[col].fillna(0)

        elif col in INTERP_COLS:
            # Ping RTT: linear interpolation between real measurements
            # limit_area="inside" prevents extrapolation beyond first/last ping
            merged[col] = (
                merged[col]
                .interpolate(method="time", limit_area="inside")
                .ffill()
                .bfill()
            )

        elif col.startswith("weather_"):
            if col == "weather_code_nws":
                merged[col] = merged[col].ffill().bfill()
            else:
                merged[col] = (
                    pd.to_numeric(merged[col], errors="coerce")
                    .ffill().bfill()
                )

        elif col in ZERO_FILL_COLS:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

        else:
            # dish telemetry, satellite position, iperf: forward-fill
            merged[col] = (
                pd.to_numeric(merged[col], errors="coerce")
                .ffill()
                .bfill()
            )

    return merged


# ---------------------------------------------------------------------------
# Core folder parser
# ---------------------------------------------------------------------------

def parse_folder(folder: Path,
                 include_iperf: bool,
                 include_ping: bool):

    frames: dict = {}
    sat_anchor_ts = None

    # iperf
    if include_iperf:
        iperf_path = folder / "iperf.csv"
        if iperf_path.exists():
            iperf_df = trim_edges(parse_iperf_csv(iperf_path))
            is_ul = bool(re.match(r"(iperf_udp_ul|udp_ul)_\d+", folder.name, re.IGNORECASE))
            if is_ul:
                server_path = folder / "iperf_server.txt"
                if server_path.exists():
                    iperf_df = merge_ul_server(
                        iperf_df, parse_iperf_server_txt(server_path)
                    )
                else:
                    print(f"  [warn] no iperf_server.txt for UL in {folder.name}")
            frames["iperf"] = iperf_df

    # ping
    if include_ping:
        ping_map = {
            "ping.csv":         "ping_tunnel_rtt_ms",
            "ping_tunnel.csv":  "ping_tunnel_rtt_ms",
            "ping_1_1_1_1.csv": "ping_internet_rtt_ms",
            "ping_server.csv":  "ping_server_rtt_ms",
            "ping_dns.csv":     "ping_dns_rtt_ms",
        }
        seen: set = set()
        for filename, col_name in ping_map.items():
            if col_name in seen:
                continue
            path = folder / filename
            if path.exists():
                frames[col_name] = trim_edges(parse_ping_csv(path, col_name))
                seen.add(col_name)

    # satellite
    sat_path = folder / "satellite.csv"
    if sat_path.exists():
        sat_df, sat_anchor_ts = parse_satellite_csv(sat_path)
        frames["satellite"] = trim_edges(sat_df)
    else:
        print(f"  [warn] no satellite.csv in {folder.name}")

    # starlink dish
    sl_path = folder / "starlink.csv"
    if sl_path.exists():
        frames["starlink"] = trim_edges(parse_starlink_csv(sl_path))

    # weather
    weather_files = sorted(folder.glob("weather_*.csv"))
    if weather_files:
        frames["weather"] = trim_edges(parse_weather_csv(weather_files[0]))

    if not frames:
        print(f"  [skip] no parseable files in {folder.name}")
        return None

    merged = align_and_fill(frames, sat_anchor_ts, folder)
    if merged is None or merged.empty:
        return None

    report_missing_cols(merged, folder, include_iperf, include_ping)

    merged = apply_schema(merged)

    # Convert to US Central, strip tz suffix
    merged.index = merged.index.tz_convert(CENTRAL_TZ).tz_localize(None)
    merged.index.name = "timestamp_cst"

    # Time features
    merged["hour_cst"] = merged.index.hour
    merged["dow_cst"]  = merged.index.isocalendar().day.values  # 1=Mon ISO

    # Column ordering
    time_cols    = ["hour_cst", "dow_cst"]
    tput_cols    = ["iperf_throughput_mbps", "dish_dl_throughput_mbps",
                    "dish_ul_throughput_mbps"]
    latency_cols = ["iperf_jitter_ms", "iperf_loss_pct",
                    "ping_tunnel_rtt_ms", "ping_internet_rtt_ms",
                    "ping_server_rtt_ms", "ping_dns_rtt_ms",
                    "dish_pop_ping_ms"]
    sat_cols     = ["satellite_id", "satellite_dtc", "satellite_handoff",
                    "satellite_age_s", "distance_km", "elevation_deg",
                    "azimuth_deg", "sat_lat", "sat_lon", "sat_alt_km"]
    dish_cols    = ["dish_obstruction_fraction", "dish_snr", "dish_gps_sats",
                    "dish_boresight_azimuth_deg", "dish_boresight_elevation_deg",
                    "dish_tilt_angle_deg", "dish_latitude", "dish_longitude",
                    "dish_altitude_m"]
    weather_cols = ["weather_temp_f", "weather_humidity", "weather_pressure",
                    "weather_wind_speed", "weather_wind_dir", "weather_rain_rate",
                    "weather_visibility", "weather_code_nws"]

    ordered   = time_cols + tput_cols + latency_cols + sat_cols + dish_cols + weather_cols
    ordered   = [c for c in ordered if c in merged.columns]
    remaining = [c for c in merged.columns if c not in ordered]
    return merged[ordered + remaining].sort_index()


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def build_latency_dataset(session_dir: Path, out_dir: Path) -> None:
    baseline_dir = session_dir / "baseline"
    if not baseline_dir.exists():
        return
    df = parse_folder(baseline_dir, include_iperf=False, include_ping=True)
    if df is None or df.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{session_dir.name}_latency.csv"
    df.to_csv(out_path)
    print(f"  -> {out_path}  ({len(df)} rows)")


def build_latency_loss_dataset(session_dir: Path, out_dir: Path) -> None:

    dl_folders = sorted([
        d for d in session_dir.iterdir()
        if d.is_dir() and re.match(r"udp_dl_\d+", d.name)
    ])

    ul_folders = sorted([
        d for d in session_dir.iterdir()
        if d.is_dir() and re.match(r"udp_ul_\d+", d.name)
    ])

    for direction, folders in [("dl", dl_folders), ("ul", ul_folders)]:

        if not folders:
            continue

        dfs = []

        for folder in folders:
            df = parse_folder(folder, include_iperf=True, include_ping=True)

            if df is not None and not df.empty:
                dfs.append(df)

        if not dfs:
            continue

        final_df = pd.concat(dfs).sort_index()

        direction_out = out_dir / direction
        direction_out.mkdir(parents=True, exist_ok=True)

        out_path = (
            direction_out /
            f"{session_dir.name}_{direction}_latency_loss_throughput.csv"
        )

        final_df.to_csv(out_path)

        print(f"  -> {out_path}  ({len(final_df)} rows)")


def build_throughput_dataset(session_dir: Path, out_dir: Path) -> None:

    dl_folders = sorted([
        d for d in session_dir.iterdir()
        if d.is_dir() and re.match(r"iperf_udp_dl_\d+", d.name)
    ])

    ul_folders = sorted([
        d for d in session_dir.iterdir()
        if d.is_dir() and re.match(r"iperf_udp_ul_\d+", d.name)
    ])

    for direction, folders in [("dl", dl_folders), ("ul", ul_folders)]:

        if not folders:
            continue

        dfs = []

        for folder in folders:
            df = parse_folder(folder, include_iperf=True, include_ping=False)

            if df is not None and not df.empty:
                dfs.append(df)

        if not dfs:
            continue

        final_df = pd.concat(dfs).sort_index()

        direction_out = out_dir / direction
        direction_out.mkdir(parents=True, exist_ok=True)

        out_path = (
            direction_out /
            f"{session_dir.name}_{direction}_throughput.csv"
        )

        final_df.to_csv(out_path)

        print(f"  -> {out_path}  ({len(final_df)} rows)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse Starlink session data into modelling-ready CSVs."
    )
    parser.add_argument("--raw_dir",  default="./raw",             help="Root raw data dir")
    parser.add_argument("--out_dir",  default="./parsed/starlink", help="Output root dir")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    session_dirs = sorted([
        d for d in raw_dir.iterdir()
        if d.is_dir() and d.name.startswith("starlink_")
    ])

    if not session_dirs:
        print(f"No starlink_* folders found in {raw_dir}")
        sys.exit(1)

    print(f"Found {len(session_dirs)} session(s)\n")

    for session_dir in session_dirs:
        print(f"Session: {session_dir.name}")
        build_latency_dataset(session_dir,      out_dir / "latency")
        build_latency_loss_dataset(session_dir, out_dir / "latency_loss_throughput")
        build_throughput_dataset(session_dir,   out_dir / "throughput")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
"""
parse_quectel.py
----------------

python3 parse_quectel.py \
  --raw_dir ./raw/quectel/ames_000 \
  --out_dir ./parsed/quectel
  
python3 parse_quectel.py \
  --raw_dir ./raw/quectel/ag_001 \
  --out_dir ./parsed/quectel
  
python3 parse_quectel.py \
  --raw_dir ./raw/quectel/ag_010 \
  --out_dir ./parsed/quectel
  
python3 parse_quectel.py \
  --raw_dir ./raw/quectel/cf_000 \
  --out_dir ./parsed/quectel


Parses Quectel 5G experiment sessions into 3 dataset types:

1) latency
   - Uses baseline/  (ping + quectel + weather, no iperf)
   - ~200ms native ping cadence
   - Output: parsed/quectel/latency/<session>_latency.csv

2) latency_loss_throughput
   - Uses udp_dl_*/ and udp_ul_*/  (iperf + ping + quectel + weather)
   - 1s native iperf cadence
   - Output: parsed/quectel/latency_loss_throughput/<session>_llt.csv

3) throughput
   - Uses iperf_udp_dl_*/ and iperf_udp_ul_*/  (iperf + quectel + weather, no ping)
   - 1s native iperf cadence
   - Output: parsed/quectel/throughput/<session>_throughput.csv

Timing strategy
---------------
Every source file keeps its exact original timestamps. All timestamps are
unioned into one index — no snapping, no invented grid. Gaps filled per
signal type:
  - ping RTT       : linear interpolation
  - quectel signal : forward-fill (1s cadence, valid between readings)
  - weather        : forward-fill (changes over minutes)
  - cell_change /
    handover       : 0 between known events (binary)

Output times are US Central (tz-naive), with hour_cst and dow_cst columns.

quectel.csv columns
-------------------
Raw:
  cellid   : full hex cell ID string  e.g. "00019401F"
  inner_cell: first 4 chars           e.g. "0001"   (macro cell)
  base_station: last 5 chars          e.g. "9401F"  (base station)
  band     : string  e.g. "mid-band"
  arfcn    : string  (radio channel number)
  rsrp     : int     (reference signal received power, dBm)
  rsrq     : int     (reference signal received quality, dB)
  sinr     : int     (signal to interference+noise ratio, dB)

Derived:
  cell_change : 1 when inner_cell changes from previous row, else 0
  handover    : 1 when base_station changes from previous row, else 0
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
CENTRAL_TZ = "America/Chicago"

# Ping RTT: linear interpolation
INTERP_COLS = {"ping_rtt_ms"}

# Binary event cols: absence = 0
BINARY_ZERO_COLS = {"cell_change", "handover"}

# Recently added cols where null in older sessions means 0
ZERO_FILL_COLS: set = set()

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
SCHEMA = {
    # Integers filled with -1 when missing (true unknown)
    "integers_neg1": [
        "rsrp", "rsrq", "sinr",
        "hour_cst", "dow_cst",
    ],
    # Boolean event cols: absence = 0, never -1
    "integers_bool": [
        "cell_change", "handover",
    ],
    "rounding": {
        4: ["ping_rtt_ms", "iperf_jitter_ms"],
        5: ["iperf_loss_pct"],
        3: ["iperf_throughput_mbps",
            "weather_temp_f", "weather_pressure",
            "weather_wind_speed", "weather_rain_rate",
            "weather_humidity", "weather_wind_dir", "weather_visibility"],
    },
}

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


def report_missing_cols(merged: pd.DataFrame, folder: Path,
                        include_iperf: bool, include_ping: bool) -> None:
    expected = [
        # quectel signal — always present
        "cellid", "inner_cell", "base_station",
        "cell_change", "handover",
        "band", "arfcn", "rsrp", "rsrq", "sinr",
        # weather — always present
        "weather_temp_f", "weather_humidity", "weather_pressure",
        "weather_wind_speed", "weather_wind_dir", "weather_rain_rate",
        "weather_visibility", "weather_code_nws",
    ]
    if include_iperf:
        expected += ["iperf_throughput_mbps", "iperf_jitter_ms", "iperf_loss_pct"]
    if include_ping:
        if (folder / "ping.csv").exists():
            expected.append("ping_rtt_ms")
    missing = [c for c in expected if c not in merged.columns]
    if missing:
        print(f"  [missing cols] {folder.name}: {', '.join(missing)}")


def apply_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce integer types and round floats per SCHEMA."""
    for col in SCHEMA["integers_neg1"]:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(-1)
                .astype("int64")
            )
    for col in SCHEMA["integers_bool"]:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(0)
                .astype("int64")
            )
    for decimals, cols in SCHEMA["rounding"].items():
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").round(decimals)
    return df


# ---------------------------------------------------------------------------
# File parsers  (preserve exact original timestamps)
# ---------------------------------------------------------------------------

def parse_ping_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = to_utc(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[["rtt_ms"]].rename(columns={"rtt_ms": "ping_rtt_ms"})
    df["ping_rtt_ms"] = pd.to_numeric(df["ping_rtt_ms"], errors="coerce")
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


def parse_quectel_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = to_utc(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    # Drop metadata cols
    for drop_col in ["experiment_time"]:
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])

    # Derive serving sector / base station from cellid
    #
    # Example:
    #   cellid = 00019401F
    #
    #   base_station = first 6 chars  -> 000194
    #   inner_cell   = last 3 chars   -> 01F
    #
    cellid = df["cellid"].astype(str).str.strip()

    df["cellid"] = cellid

    # Sector / cell portion used for cell_change detection
    df["inner_cell"] = cellid.str[-3:]

    # eNB/gNB/base-station identifier
    df["base_station"] = cellid.str[:6]

    # Cell change: inner_cell changed from previous row
    df["cell_change"] = (
        df["inner_cell"].ne(df["inner_cell"].shift())
        & df["inner_cell"].notna()
    ).astype(int)
    df.iloc[0, df.columns.get_loc("cell_change")] = 0

    # Handover: base_station changed from previous row
    df["handover"] = (
        df["base_station"].ne(df["base_station"].shift())
        & df["base_station"].notna()
    ).astype(int)
    df.iloc[0, df.columns.get_loc("handover")] = 0

    # Numeric signal cols
    for col in ["rsrp", "rsrq", "sinr"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep = ["cellid", "inner_cell", "base_station",
            "cell_change", "handover",
            "band", "arfcn", "rsrp", "rsrq", "sinr"]
    return df[[c for c in keep if c in df.columns]]


def parse_weather_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["timestamp"] = to_utc(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    # Common mappings shared across both weather station formats
    keep = {
        "Temperature":   "weather_temp_f",
        "Humidity":      "weather_humidity",
        "Pressure":      "weather_pressure",
        "WindSpeed":     "weather_wind_speed",
        "WindDirection": "weather_wind_dir",
        "RainRate":      "weather_rain_rate",
    }

    # Optional fields depending on station type
    optional_keep = {
        # WilsonHall format
        "MORVisibility": "weather_visibility",
        "WeatherCodeNWS": "weather_code_nws",

        # AgronomyFarm format alternatives
        "WeatherType": "weather_type",
        "PrecipitationType": "weather_precip_type",
    }

    existing = {}

    for src, dst in keep.items():
        if src in df.columns:
            existing[src] = dst

    for src, dst in optional_keep.items():
        if src in df.columns:
            existing[src] = dst

    df = df.rename(columns=existing)

    out_cols = list(existing.values())

    # Ensure expected columns exist even if missing from one station format
    for col in [
        "weather_visibility",
        "weather_code_nws",
        "weather_type",
        "weather_precip_type",
    ]:
        if col not in df.columns:
            df[col] = np.nan
            out_cols.append(col)

    return df[out_cols]


# ---------------------------------------------------------------------------
# Alignment and filling
# ---------------------------------------------------------------------------

def align_and_fill(frames: dict, folder: Path) -> pd.DataFrame | None:
    """
    Union all frame timestamps. Fill per signal type. Clip to overlap window.
    """
    if not frames:
        return None

    t_start = max(df.index.min() for df in frames.values())
    t_end   = min(df.index.max() for df in frames.values())
    if t_start >= t_end:
        print(f"  [skip] empty overlap window in {folder.name}")
        return None

    frames = {k: v.loc[t_start:t_end] for k, v in frames.items()}

    # Union of all exact timestamps — no snapping
    all_ts = sorted(set().union(*[set(df.index) for df in frames.values()]))
    union_index = pd.DatetimeIndex(all_ts)

    merged = pd.DataFrame(index=union_index)
    merged.index.name = "timestamp"

    for name, df in frames.items():
        report_nulls(df, name, str(folder))
        merged = merged.join(df, how="left")

    # Per-signal filling
    for col in merged.columns:

        if col in BINARY_ZERO_COLS:
            merged[col] = merged[col].fillna(0)

        elif col in INTERP_COLS:
            # Linear interpolation for ping RTT
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

        elif col in {"cellid", "inner_cell", "base_station", "band", "arfcn"}:
            # String cols: forward-fill
            merged[col] = merged[col].ffill().bfill()

        else:
            # Numeric signal cols (rsrp, rsrq, sinr, iperf)
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
                 include_ping: bool) -> pd.DataFrame | None:

    frames: dict = {}

    # iperf
    if include_iperf:
        iperf_path = folder / "iperf.csv"
        if iperf_path.exists():
            iperf_df = trim_edges(parse_iperf_csv(iperf_path))
            is_ul = bool(re.match(r"(iperf_udp_ul|udp_ul)_\d+",
                                  folder.name, re.IGNORECASE))
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
        ping_path = folder / "ping.csv"
        if ping_path.exists():
            frames["ping"] = trim_edges(parse_ping_csv(ping_path))

    # quectel modem telemetry
    quectel_path = folder / "quectel.csv"
    if quectel_path.exists():
        frames["quectel"] = trim_edges(parse_quectel_csv(quectel_path))
    else:
        print(f"  [warn] no quectel.csv in {folder.name}")

    # weather
    weather_files = sorted(folder.glob("weather_*.csv"))
    if weather_files:
        frames["weather"] = trim_edges(parse_weather_csv(weather_files[0]))

    if not frames:
        print(f"  [skip] no parseable files in {folder.name}")
        return None

    merged = align_and_fill(frames, folder)
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
    tput_cols    = ["iperf_throughput_mbps"]
    latency_cols = ["iperf_jitter_ms", "iperf_loss_pct", "ping_rtt_ms"]
    cell_cols    = ["cellid", "inner_cell", "base_station",
                    "cell_change", "handover",
                    "band", "arfcn", "rsrp", "rsrq", "sinr"]
    weather_cols = ["weather_temp_f", "weather_humidity", "weather_pressure",
                    "weather_wind_speed", "weather_wind_dir", "weather_rain_rate",
                    "weather_visibility", "weather_code_nws"]

    ordered   = time_cols + tput_cols + latency_cols + cell_cols + weather_cols
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
    device_name = session_dir.parent.name
    out_path = out_dir / f"{session_dir.name}_{device_name}_latency.csv"
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

    device_name = session_dir.parent.name

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
            f"{session_dir.name}_{device_name}_{direction}_latency_loss_throughput.csv"
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

    device_name = session_dir.parent.name

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
            f"{session_dir.name}_{device_name}_{direction}_throughput.csv"
        )

        final_df.to_csv(out_path)

        print(f"  -> {out_path}  ({len(final_df)} rows)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse Quectel 5G session data into modelling-ready CSVs."
    )
    parser.add_argument("--raw_dir",  default="./data",            help="Root raw data dir")
    parser.add_argument("--out_dir",  default="./parsed/quectel",  help="Output root dir")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    session_dirs = sorted([
        d for d in raw_dir.iterdir()
        if d.is_dir() and d.name.startswith("quectel_")
    ])

    if not session_dirs:
        print(f"No quectel_* folders found in {raw_dir}")
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
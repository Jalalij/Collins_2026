
import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime, timedelta
import math
from scipy import stats
from skyfield.api import load, Topos, EarthSatellite, wgs84
import re
import requests
import json
from pathlib import Path
import shutil
from sklearn.preprocessing import OneHotEncoder

# =====================================================
# Configuration and Experiment Management
# =====================================================
class ExperimentProcessor:
    def __init__(self, data_folder, ut_coords, output_dir="./processed_data"):
        self.data_folder = data_folder
        self.ut_coords = ut_coords
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def find_ping_batches(self):
        """Find all ping experiments (baseline: experiments are anchored on ping)."""
        experiments = []
        ping_files = glob.glob(os.path.join(self.data_folder, 'ping', '*.csv'))
        print(f"[debug] Found {len(ping_files)} ping files")

        for pf in ping_files:
            try:
                df = pd.read_csv(pf)
                if 'timestamp' in df.columns and not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                    exp_start, exp_end = df['timestamp'].min(), df['timestamp'].max()
                    experiments.append({
                        'ping_file': pf,
                        'start_time': exp_start,
                        'end_time': exp_end,
                        'duration': (exp_end - exp_start).total_seconds() / 60
                    })
                    print(f"[debug] Ping batch: {pf}, {exp_start} → {exp_end}")
            except Exception as e:
                print(f"Error processing ping file {pf}: {e}")
        return experiments

    def find_matching_iperf_files(self, exp_start, exp_end, time_tolerance_sec=20):
        """Find ALL iperf files overlapping with timeframe."""
        iperf_files = glob.glob(os.path.join(self.data_folder, 'iperf', '*.csv'))
        matching_files = []
        tol = timedelta(seconds=time_tolerance_sec)

        for ipf in iperf_files:
            try:
                df_times = pd.read_csv(ipf, usecols=['timestamp'])
                df_times['timestamp'] = pd.to_datetime(df_times['timestamp'], utc=True)
                file_start, file_end = df_times['timestamp'].min(), df_times['timestamp'].max()

                if abs(file_start-exp_start)<=tol or abs(file_end-exp_end)<=tol:
                    matching_files.append(ipf)
                    print(f"[debug] Matching iperf file: {ipf}, {file_start} → {file_end}")
                    
            except Exception as e:
                print(f"Error checking iperf file {ipf}: {e}")

        return matching_files

    def find_matching_sattrace_files(self, exp_start, exp_end, time_tolerance_min=2, save_folder="sattrace_matched"):
        """Find ALL sattrace files overlapping with timeframe."""
        sattrace_files = glob.glob(os.path.join(self.data_folder, 'sattrace', '*.csv'))
        matching_files = []
        tol = timedelta(minutes=time_tolerance_min)
        #print(f"[debug] Found {len(sattrace_files)} sattrace files")

        save_dir = os.path.join(self.data_folder, save_folder)
        os.makedirs(save_dir, exist_ok=True)

        for sf in sattrace_files:
            try:
                df_times = pd.read_csv(sf, usecols=['TIMESTAMP'], sep=r",(?! ?\"\w+\"(?=:))", engine="python")
                df_times['TIMESTAMP'] = pd.to_datetime(df_times['TIMESTAMP'], utc=True)
                file_start, file_end = df_times['TIMESTAMP'].min(), df_times['TIMESTAMP'].max()

                if (file_end + tol >= exp_start) and (file_start - tol <= exp_end):
                    matching_files.append(sf)
                    # Copy to sattrace_matched folder
                    dest = os.path.join(save_dir, os.path.basename(sf))
                    shutil.copy(sf, dest)
                    print(f"[debug] Matching sattrace file: {sf}, {file_start} → {file_end}")
            except Exception as e:
                print(f"Error checking sattrace file {sf}: {e}")

        return matching_files

# =====================================================
# Satellite TLE Data Handling
# =====================================================
def load_tle_data(tle_file_path):
    """Load TLE data from file and create satellite objects"""
    satellites = {}
    try:
        with open(tle_file_path, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 3):
                if i + 2 < len(lines):
                    name = lines[i].strip()
                    line1 = lines[i+1].strip()
                    line2 = lines[i+2].strip()
                    satellites[name] = EarthSatellite(line1, line2, name)
        print(f"Loaded {len(satellites)} satellites from TLE file")
        return satellites
    except Exception as e:
        print(f"Error loading TLE file: {e}")
        return {}

def download_tle_for_timeframe(start_time, end_time, satellite_ids=None, tle_sat='oneweb', save_path="./data/tle_data.txt"):
    """Download TLEs valid for the experiment timeframe and save to file"""
    # TODO WRONG: latest only, not historical. All the TLE fiels in the tle folder are the same
    tle_groups_all = {
        'starlink': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle',
        'oneweb': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=oneweb&FORMAT=tle',
        'iridium': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium&FORMAT=tle',
        'geo': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=geo&FORMAT=tle',
        'gps': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=gps&FORMAT=tle',
    }

    # Make sure satellite_ids is a list of strings
    if satellite_ids is not None:
        satellite_ids = [str(s) for s in satellite_ids]
    
    tle_groups = {tle_sat: tle_groups_all[tle_sat]}
    tle_data = {}
    saved_tles = []
    
    for group_name, url in tle_groups.items():
        try:
            print(f"Downloading TLEs for {group_name}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            content = response.text.strip()
            
            if content:
                lines = content.split('\n')
                for i in range(0, len(lines), 3):
                    if i + 2 < len(lines):
                        name = lines[i].strip()
                        line1 = lines[i+1].strip()
                        line2 = lines[i+2].strip()
                        
                        # Check if this satellite matches our needs
                        if satellite_ids:
                            sat_match = any(sat_id in name for sat_id in satellite_ids)
                            if not sat_match:
                                continue
                        
                        saved_tles.extend([name, line1, line2])
                        sat_obj = EarthSatellite(line1, line2, name)
                        tle_data[name] = sat_obj
                            
        except Exception as e:
            print(f"Error downloading {group_name} TLEs: {e}")
    
    # Save TLEs to file
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write('\n'.join(saved_tles))
        print(f"Saved {len(saved_tles)//3} TLEs to {save_path}")
    
    return tle_data

def infer_connected_satellite_id(ut_coords, sattrace_file, tle_data, max_angle_error_deg=2.0):
    """
    Infer connected satellite IDs for each AIM log entry using TLE propagation (SGP4),
    with azimuth aligned to true north.

    Parameters
    ----------
    ut_coords : tuple
        (lon, lat, alt_m) of the UT
    sattrace_file : str
        Path to AIM tracking log CSV
    tle_data : dict
        {name: EarthSatellite} mapping of satellite identifiers to Skyfield EarthSatellite objects
    max_angle_error_deg : float
        Maximum allowed pointing error (deg) to declare a match

    Returns
    -------
    list
        Inferred satellite IDs (strings) or None, aligned with rows in the AIM log
    """
    def _wrap_deg(diff):
        """Wrap angle difference to (-180, 180] degrees."""
        return (diff + 180.0) % 360.0 - 180.0

    # Load AIM log (regex handles the JSON-like INTEGRATOR_PARAMS field)
    sattrace_df = pd.read_csv(
        sattrace_file,
        sep=r",(?! ?\"\w+\"(?=:))",
        engine="python"
    )
    sattrace_df['TIMESTAMP'] = pd.to_datetime(sattrace_df['TIMESTAMP'], utc=True)

    # Extract pointing relative to true north
    # (cast in case CSV loaded numeric columns as strings)
    az_target = (sattrace_df['A0_ACTUAL_COORD0'].astype(float) +
                 sattrace_df['A0_TRUE_NORTH'].astype(float)) % 360.0
    el_target = sattrace_df['A0_ACTUAL_COORD1'].astype(float)

    lon, lat, alt_m = ut_coords
    ts = load.timescale()
    ut = wgs84.latlon(latitude_degrees=lat, longitude_degrees=lon, elevation_m=alt_m)

    inferred = []

    for i, row in sattrace_df.iterrows():
        t = ts.utc(row["TIMESTAMP"].to_pydatetime())
        az_t = float(az_target.iloc[i])
        el_t = float(el_target.iloc[i])

        closest_sat = None
        min_err = np.inf

        for name, sat in tle_data.items():
            # Predict topocentric alt/az from UT to satellite
            alt, az, _ = sat.at(t).observe_from(ut).apparent().altaz()

            # Compute pointing error with proper azimuth wrap
            d_az = _wrap_deg(az.degrees - az_t)
            d_el = (alt.degrees - el_t)
            err = np.hypot(d_az, d_el)  # Euclidean in az/el space (deg)

            if err < min_err:
                min_err = err
                closest_sat = name
        
        print('[debug] The Oneweb Sat ID is ', closest_sat)

        inferred.append(closest_sat if min_err <= max_angle_error_deg else None)

    return inferred

def test_calculate_satellite_metrics_from_tle(sattrace_file, ut_coords):
    """
    Test function for TLE-based satellite metric calculation.
    
    Parameters:
        sattrace_file : path to one sattrace CSV file
        ut_coords     : (lon, lat, alt) tuple for UT
    """
    """
    wrong (TODO): 
    It tries to match Hughes SATELLITE_ID string to a TLE satellite name (via substring). That’s wrong — Hughes IDs don’t map 1:1 to NORAD IDs.
    """
    print(f"[test] Loading sattrace file: {sattrace_file}")

    sattrace_df = pd.read_csv(sattrace_file, sep=r",(?! ?\"\w+\"(?=:))", engine="python")
    sattrace_df['TIMESTAMP'] = pd.to_datetime(sattrace_df['TIMESTAMP'], utc=True)

    if sattrace_df.empty:
        print("[test] No valid TIMESTAMP entries found")
        return

    exp_start, exp_end = sattrace_df['TIMESTAMP'].min(), sattrace_df['TIMESTAMP'].max()
    print(f"[test] Time range in file: {exp_start} → {exp_end}")

    # Get satellite IDs
    satellite_ids = sattrace_df['SATELLITE_ID'].astype(str).unique()
    print(f"[test] Satellite IDs found: {satellite_ids}")

    # Download TLE data for timeframe
    tle_data = download_tle_for_timeframe(exp_start, exp_end, satellite_ids, tle_sat="oneweb")

    if not tle_data:
        print("[test] No TLE data retrieved")
        return

    # Compute metrics from TLEs
    metrics_df = calculate_satellite_metrics_from_tle(sattrace_df, tle_data, ut_coords)

    print("\n[test] First 5 rows with TLE-based metrics:")
    print(metrics_df[['TIMESTAMP', 'SATELLITE_ID', 
                      'tle_elevation_deg', 'tle_azimuth_deg', 'tle_slant_range_km']].head())

    return metrics_df

def calculate_satellite_metrics_from_tle(sattrace_df, tle_data, ut_coords):
    """
    Calculate satellite geometry (elevation, azimuth, slant range) using TLEs.
    
    Parameters:
        sattrace_df : pd.DataFrame with SATELLITE_ID and TIMESTAMP
        tle_data    : dict of {name: EarthSatellite} from download_tle_for_timeframe()
        ut_coords   : (lon, lat, alt) tuple for UT
        
    Returns:
        pd.DataFrame with added columns:
        - tle_elevation_deg
        - tle_azimuth_deg
        - tle_slant_range_km
    """
    df = sattrace_df.copy()
    ts = load.timescale()
    ut_location = wgs84.latlon(ut_coords[1], ut_coords[0], ut_coords[2])  # (lat, lon, alt)

    elev_list, az_list, slant_list = [], [], []

    for _, row in df.iterrows():
        try:
            t = ts.from_datetime(row['TIMESTAMP'].to_pydatetime())
            sat_id = str(row['SATELLITE_ID'])

            # Find matching satellite by ID substring
            sat_obj = None
            for name, sat in tle_data.items():
                if sat_id in name:
                    sat_obj = sat
                    break

            if sat_obj is None:
                elev_list.append(np.nan)
                az_list.append(np.nan)
                slant_list.append(np.nan)
                continue

            diff = sat_obj.at(t) - ut_location
            alt, az, distance = diff.apparent().altaz()

            elev_list.append(alt.degrees)
            az_list.append(az.degrees)
            slant_list.append(distance.km)

        except Exception:
            elev_list.append(np.nan)
            az_list.append(np.nan)
            slant_list.append(np.nan)

    df['tle_elevation_deg'] = elev_list
    df['tle_azimuth_deg'] = az_list
    df['tle_slant_range_km'] = slant_list
    return df

def compute_az_el_slant(tle_data, ut_coords, timestamps, elev_min=0):
    """
    Compute azimuth, elevation, and slant range for all satellites at given times.

    Parameters:
        tle_data   : dict {name: EarthSatellite}
        ut_coords  : (lon, lat, alt_m) of UT
        timestamps : list/Series of pd.Timestamp
        elev_min   : optional filter, only return sats above this elevation (deg)

    Returns:
        pd.DataFrame with columns:
        - TIMESTAMP
        - sat_name
        - elevation_deg
        - azimuth_deg
        - slant_range_km
    """
    ts = load.timescale()
    ut_location = wgs84.latlon(ut_coords[1], ut_coords[0], ut_coords[2])

    results = []
    for tstamp in timestamps:
        t = ts.from_datetime(tstamp.to_pydatetime())
        for name, sat in tle_data.items():
            alt, az, dist = (sat.at(t) - ut_location.at(t)).altaz()
            if alt.degrees >= elev_min:
                results.append({
                    "TIMESTAMP": tstamp,
                    "sat_name": name,
                    "elevation_deg": alt.degrees,
                    "azimuth_deg": az.degrees,
                    "slant_range_km": dist.km
                })

    return pd.DataFrame(results)

# ======================================================
# Satellite Metrics Calculation
# ======================================================

def extract_angles_from_aim(df):
    """
    Extract azimuth and elevation from Hughes AIM tracking logs.
    TODO: Need slant_range_km
    Parameters
    ----------
    df : pandas.DataFrame
        Must contain:
        - A0_ACTUAL_COORD0 (float): raw azimuth
        - A0_ACTUAL_COORD1 (float): raw elevation
        - A0_TRUE_NORTH (float): azimuth offset

    Returns
    -------
    pandas.DataFrame
        Copy of df with added columns:
        - azimuth_deg
        - elevation_deg
    """
    df = df.copy()
    
    # Corrected azimuth (wrap to [0, 360))
    df['azimuth_deg'] = (df['A0_ACTUAL_COORD0'] + df['A0_TRUE_NORTH']) % 360
    
    # Elevation directly
    df['elevation_deg'] = df['A0_ACTUAL_COORD1']
    
    return df

def calculate_handover_rates(df, window_minutes=1, debug_index=None, debug_window=10):
    """Calculate inter-beam and inter-satellite handover rates"""
    df = df.copy()
    df = df.sort_values('TIMESTAMP')

    # Set TIMESTAMP as index for time-based rolling
    df = df.set_index('TIMESTAMP')

    # Ensure IDs are integers (avoid float/string mismatches)
    df['SATELLITE_ID'] = df['SATELLITE_ID'].astype(str).str.strip().astype(int)
    df['TRACK_ID'] = df['TRACK_ID'].astype(str).str.strip().astype(int)

    # Compare with shifted values
    df["satellite_prev"] = df["SATELLITE_ID"].shift()
    df["track_prev"] = df["TRACK_ID"].shift()
    
    # Calculate changes
    df['satellite_change'] = df['SATELLITE_ID'].ne(df["satellite_prev"]).astype(int)
    df['beam_change'] = df['TRACK_ID'].ne(df["track_prev"]).astype(int)
    df.loc[df.index[0], 'satellite_change'] = 0
    df.loc[df.index[0], 'beam_change'] = 0

    # Debug print around suspicious index
    if debug_index is not None: #debug_index = 120 for exp_000
        start = max(0, debug_index - debug_window)
        end = min(len(df), debug_index + debug_window)
        print(df.iloc[start:end][[
            "SATELLITE_ID", "satellite_prev", "satellite_change",
            "TRACK_ID", "track_prev", "beam_change"
        ]])
    
    # Calculate rolling handover rates
    # Use time-based rolling window
    window_str = f'{window_minutes}min'
    df['inter_satellite_ho_rate'] = df['satellite_change'].rolling(window=window_str).sum() / (window_minutes / 60)
    df['inter_beam_ho_rate'] = df['beam_change'].rolling(window=window_str).sum() / (window_minutes / 60)
    
    return df.reset_index()  # restore TIMESTAMP as column

def trim_to_closest_time(df, time_col, start, end):
    """
    Trim a DataFrame to the rows between the closest timestamps to start and end.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with a timestamp column.
    time_col : str
        Name of the timestamp column.
    start : pd.Timestamp
        Experiment start time.
    end : pd.Timestamp
        Experiment end time.
    
    Returns
    -------
    pd.DataFrame
        Trimmed dataframe.
    """
    # Ensure datetime
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)

    # Find closest index to start
    start_idx = (df[time_col] - start).abs().idxmin()
    # Find closest index to end
    end_idx = (df[time_col] - end).abs().idxmin()

    # Slice between them (ensure proper order)
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    return df.loc[start_idx:end_idx].reset_index(drop=True)


# ======================================================
# Data Processing Functions
# ======================================================
def load_and_process_experiment(iperf_file, ping_file, sattrace_file, ut_coords, resample_freq='1s', use_tle_metrics=False, tolerance_ms=500):
    """Process a single experiment""" #250ms
    print(f"Processing experiment: {Path(ping_file).name}")
    
    # Load data
    iperf_df = pd.read_csv(iperf_file)
    ping_df = pd.read_csv(ping_file)
    sattrace_df = pd.read_csv(sattrace_file)
    
    # Convert timestamps
    iperf_df['timestamp'] = pd.to_datetime(iperf_df['timestamp'], utc=True)
    ping_df['timestamp'] = pd.to_datetime(ping_df['timestamp'], utc=True)
    sattrace_df['TIMESTAMP'] = pd.to_datetime(sattrace_df['TIMESTAMP'], utc=True)
    
    # Get experiment timeframe
    exp_start = max(iperf_df['timestamp'].min(), ping_df['timestamp'].min())
    exp_end = min(iperf_df['timestamp'].max(), ping_df['timestamp'].max())
    print('[debug] exp_start exp_end', exp_start, exp_end)

    # Range filter ping data to experiment timeframe
    ping_df = ping_df[
        (ping_df['timestamp'] >= exp_start) &
        (ping_df['timestamp'] <= exp_end)
    ]

    # Filter iperf data to experiment timeframe
    iperf_df = iperf_df[
        (iperf_df['timestamp'] >= exp_start) &
        (iperf_df['timestamp'] <= exp_end)
    ]

    # closest index method to filter sattrace data
    # iperf_df    = trim_to_closest_time(iperf_df, "timestamp", exp_start, exp_end)
    # ping_df     = trim_to_closest_time(ping_df, "timestamp", exp_start, exp_end)


    # ---- Aggregate ping stats ----
    ping_stats = (
        ping_df.groupby('timestamp')['rtt_ms']
        .agg(['mean', 'std', 'min', 'max', 'count'])
        .reset_index()
        .rename(columns={
            'mean': 'rtt_mean',
            'std': 'rtt_std',
            'min': 'rtt_min',
            'max': 'rtt_max',
            'count': 'ping_count'
        })
    )
    
    # Filter sattrace data to experiment timeframe
    sattrace_df = sattrace_df[
        (sattrace_df['TIMESTAMP'] >= exp_start) &
        (sattrace_df['TIMESTAMP'] <= exp_end)
    ] # - timedelta(seconds=30)  + timedelta(seconds=30)

    # # closest index method to filter sattrace data
    # # TODO: function test needed
    # sattrace_df = trim_to_closest_time(sattrace_df, "TIMESTAMP", exp_start, exp_end)
    
    # Process satellite data
    sattrace_processed = extract_angles_from_aim(sattrace_df) # azimuth and elevation
    sattrace_processed = calculate_handover_rates(sattrace_processed)
    sattrace_processed = sattrace_processed.rename(columns={'AVERAGE_SINR': 'sinr_db',
                                                            'TRACK_ID': 'track_id',
                                                            'SATELLITE_ID': 'satellite_id',
                                                            'TIMESTAMP':'timestamp'
                                                            })

    # closest_idx = (sattrace_processed['timestamp'] - exp_start).abs().idxmin()
    # sattrace_processed = sattrace_processed.loc[closest_idx:].reset_index(drop=True)
    
    print("[debug] First timestamps:")
    print("ping_stats:   ", ping_stats['timestamp'].min())
    print("iperf_df:    ", iperf_df['timestamp'].min())
    print("sattrace_df: ", sattrace_processed['timestamp'].min())

    # --- Reindex to unified timeline with nearest ---
    tol = pd.Timedelta(f"{tolerance_ms}ms")

    # --- Start from iperf as anchor timeline
    merged_df = ping_stats.copy()

    # --- Merge ping by nearest
    merged_df = pd.merge_asof(
        merged_df.sort_values('timestamp'),
        iperf_df.sort_values('timestamp'),
        on='timestamp',
        # direction='nearest',
        direction='forward', # Find the next available iperf measurement
        tolerance=tol
    )

    # --- Merge sattrace by nearest
    merged_df = pd.merge_asof(
        merged_df.sort_values('timestamp'),
        sattrace_processed.sort_values('timestamp'), 
        on='timestamp',
        # direction='nearest',
        direction='forward',
        tolerance=tol
    )

    int_cols = ["satellite_id", "track_id", "satellite_change", "beam_change"]
    for col in int_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(-1).astype(int)

    # Metrics from TLE (if requested)
    if use_tle_metrics:
        # Get satellite IDs for TLE download
        tle_sat = 'oneweb'
        satellite_ids = sattrace_df['SATELLITE_ID'].astype(str).unique()
        tle_filename = f"tle_{tle_sat}_{exp_start.strftime('%Y%m%d_%H%M%S')}.txt"
        save_path = os.path.join("./data/tle", tle_filename)

        tle_data = download_tle_for_timeframe(exp_start, exp_end, satellite_ids, save_path=save_path, tle_sat=tle_sat)
    

        sattrace_tle = calculate_satellite_metrics_from_tle(sattrace_df, tle_data, ut_coords)
        sattrace_tle = sattrace_tle.rename(columns={'TIMESTAMP':'timestamp'})
        merged_df = pd.merge_asof(
            merged_df.sort_values('timestamp'),
            sattrace_tle[['timestamp','tle_elevation_deg','tle_azimuth_deg','tle_slant_range_km']].sort_values('timestamp'),
            on='timestamp',
            direction='nearest',
            tolerance=tol
        )

    # Add experiment metadata
    merged_df['experiment_id'] = f"exp_{exp_start.strftime('%Y%m%d_%H%M%S')}"
    merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'], utc=True)
    merged_df['time_since_start'] = (merged_df['timestamp'] - exp_start).dt.total_seconds()
    merged_df['time_since_start'] = merged_df['time_since_start'].clip(lower=0)

    # Keep only relevant columns
    keep_cols = [
        'timestamp',
        'throughput_mbps', 'packet_loss_percent', 'jitter_ms', 'direction',
        'rtt_mean', #'rtt_std', 'rtt_min', 'rtt_max', 'ping_count',
        'sinr_db', 'elevation_deg', 'azimuth_deg', #'slant_range_km',
        'inter_satellite_ho_rate', 'inter_beam_ho_rate', 'track_id', 'beam_change', 
        'satellite_id', 'satellite_change', 'experiment_id', 'time_since_start'
    ]

    # Add TLE columns if present
    if use_tle_metrics:
        for col in ['tle_slant_range_km']:
            if col in merged_df.columns:
                keep_cols.append(col)

    merged_df = merged_df[keep_cols]
    
    return merged_df

def process_all_experiments(data_folder, ut_coords, output_dir="./processed_data"):
    """Process all experiments and save individually"""
    processor = ExperimentProcessor(data_folder, ut_coords, output_dir)
    experiments = processor.find_ping_batches()   # baseline = ping
    
    all_experiments_data = []
    experiment_info = []
    
    for i, exp in enumerate(experiments):
        try:
            ping_file = exp['ping_file']
            exp_start, exp_end = exp['start_time'], exp['end_time']
            print(f"\nping file used as an anchor: {ping_file}, {exp_start} → {exp_end}")
            
            # Find matching iperf files
            iperf_files = processor.find_matching_iperf_files(exp_start, exp_end)
            if not iperf_files:
                print(f"No matching iperf file found for experiment {i}")
                continue
            # copy and rename the ping file if matching iperf files are found
            dest_ping_path = os.path.join(output_dir, f"ping_exp_{i}.csv")
            shutil.copy(ping_file, dest_ping_path)

            # Concatenate all iperf files
            iperf_df = pd.concat([pd.read_csv(ipf) for ipf in iperf_files])
            iperf_df = iperf_df.sort_values("timestamp").reset_index(drop=True)
            tmp_iperf_path = os.path.join(output_dir, f"iperf_exp_{i}.csv")
            iperf_df.to_csv(tmp_iperf_path, index=False)

            # Find matching sattrace files
            sattrace_files = processor.find_matching_sattrace_files(exp_start, exp_end)
            if not sattrace_files:
                print(f"No matching sattrace files found for experiment {i}")
                continue

            # Concatenate all sattrace files
            sattrace_df = pd.concat([pd.read_csv(sf, sep=r",(?! ?\"\w+\"(?=:))", engine="python") for sf in sattrace_files])
            sattrace_df = sattrace_df.sort_values("TIMESTAMP").reset_index(drop=True)
            tmp_sattrace_path = os.path.join(output_dir, f"sattrace_exp_{i}.csv")
            sattrace_df.to_csv(tmp_sattrace_path, index=False)
            #print(f"[debug] Concatenated {len(sattrace_files)} sattrace files → {tmp_sattrace_path}\n")

            # Process experiment
            exp_data = load_and_process_experiment(
                tmp_iperf_path, ping_file, tmp_sattrace_path, ut_coords
            )

            exp_filename = f"aligned_experiment_{i:03d}_{exp_start.strftime('%Y%m%d_%H%M%S')}.csv"
            exp_data.to_csv(os.path.join(output_dir, exp_filename), index=False) # TODO: file check with index false
            all_experiments_data.append(exp_data)

            experiment_info.append({
                'experiment_id': exp_data['experiment_id'].iloc[0],
                'start_time': exp_start,
                'end_time': exp_end,
                'duration_minutes': exp['duration'],
                'file_path': exp_filename
            })

            print(f"Processed experiment {i}: {exp_filename}")

        except Exception as e:
            print(f"Error processing experiment {i}: {e}")
            continue

    if experiment_info:
        pd.DataFrame(experiment_info).to_csv(os.path.join(output_dir, 'all_processed_experiments_metadata.csv'), index=False)

    if all_experiments_data:
        combined_data = pd.concat(all_experiments_data)
        combined_data.to_csv(os.path.join(output_dir, 'all_processed_experiments_combined.csv'), index=False) # TODO: file check with index false
        return combined_data
    
    return None

def split_experiments_by_direction(input_folder="processed_data", output_folder='processed_data/ULDL'):
    """
    Load experiment files containing 'aligned_experiment' and split each
    into uplink (UL) and downlink (DL) based on the 'direction' column.
    Saves new files with '_UL' and '_DL' appended to the filename.
    """
    # Find all relevant files
    files = [f for f in os.listdir(input_folder) if "aligned_experiment" in f and f.endswith(".csv")]
    
    for file in files:
        file_path = os.path.join(input_folder, file)
        print(f"[debug] Processing {file_path}")
        
        # Load data
        df = pd.read_csv(file_path, parse_dates=["timestamp"])

        # Detect missing values
        if df.isnull().any().any():
            missing_rows = df[df.isnull().any(axis=1)]
            print(f"    Found {len(missing_rows)} rows with missing values")
            print(f"    Total rows in file before cleanup: {len(df)}")

            # Show which columns are missing in each row
            rows_to_drop = []
            for idx, row in missing_rows.iterrows():
                missing_cols = row.index[row.isnull()].tolist()
                #print(f"    Row {idx} at {row['timestamp']} has missing values in: {missing_cols}")
                
                # If 'direction' is missing OR it's the last row, mark for dropping
                if "direction" in missing_cols or idx == df.index[-1]:
                    rows_to_drop.append(idx)
                    print(f"    --> Marking row {idx} for removal")
                    print(f"    Row {idx} at {row['timestamp']} has missing values in: {missing_cols}")

            # Drop only selected rows
            if rows_to_drop:
                df = df.drop(rows_to_drop).reset_index(drop=True)
                print(f"    Removed {len(rows_to_drop)} rows with invalid direction or last-row cutoff")
            else:
                print("    No rows dropped (missing values kept)")    
        
        # Separate UL and DL
        for direction in ["UL", "DL"]:
            subset = df[df["direction"] == direction].copy()
            if subset.empty:
                continue  # skip if no rows for this direction
            
            # Build new filename
            name, ext = os.path.splitext(file)
            if name.endswith(f"_{direction}"):
                new_file = f"{name}{ext}"
            else:
                new_file = f"{name}_{direction}{ext}"
            new_path = os.path.join(input_folder, new_file)
            
            # Save subset
            subset.to_csv(new_path, index=False)
            print(f"    Saved {new_path} ({len(subset)} rows)")

def prepare_time_series_data(combined_data, target_variables=None, feature_cols=None, lookback=60, forecast_horizon=30):
    """Prepare data for time series forecasting"""
    if target_variables is None:
        target_variables = ['throughput_mbps', 'packet_loss_percent', 'jitter_ms', 'rtt_mean']

    if feature_cols is None:
        # Select numeric feature columns (drop timestamp + experiment_id)
        feature_cols = [
            "throughput_mbps", "packet_loss_percent", "jitter_ms",
            "rtt_mean", #"rtt_min", "rtt_max", "ping_count",
            "sinr_db", "elevation_deg", "azimuth_deg", #"slant_range_km",
            "inter_satellite_ho_rate", "inter_beam_ho_rate"
        ]
    
    # Ensure data is sorted by time
    data = combined_data.sort_values("timestamp")
    
    # Create features for time series modeling
    features = []
    targets = []
    
    for i in range(lookback, len(data) - forecast_horizon):
        # Input features (lookback window)
        X = data.iloc[i-lookback:i].values #[feature_cols]
        
        # Target values (forecast horizon)
        y = data.iloc[i:i+forecast_horizon][target_variables].values
        
        features.append(X)
        targets.append(y)
    
    return np.array(features), np.array(targets)

# ======================================================
# Main Execution
# ======================================================
if __name__ == "__main__":
    data_folder = "./data"
    ut_coords = (-77.487488, 39.043720, 0)  # Your UT coordinates
    output_dir = "./processed_data"
    
    try:
        print("Processing all experiments...")
        combined_data = process_all_experiments(data_folder, ut_coords, output_dir=output_dir)
        
        # split the files into UL files and DL files
        split_experiments_by_direction(output_dir)

        if combined_data is not None:
            print(f"Processed {len(combined_data)} total data points from all experiments")
            print("\nAvailable metrics:")
            for col in combined_data.columns:
                print(f"  - {col}")
            
            # Prepare for time series forecasting
            print("\nPreparing data for time series forecasting...")
            X, y = prepare_time_series_data(combined_data)
            
            print(f"Time series dataset shape:")
            print(f"Features: {X.shape}, Targets: {y.shape}")
            
            # Save prepared data for modeling
            # np.save(os.path.join(output_dir, 'time_series_features.npy'), X)
            # np.save(os.path.join(output_dir, 'time_series_targets.npy'), y)
            
            print("Data preparation complete! Ready for temporal dynamics modeling.")
            
        else:
            print("No experiments were successfully processed.")
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
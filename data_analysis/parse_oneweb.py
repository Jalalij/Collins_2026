#!/usr/bin/env python3

import os
import glob
import pandas as pd

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

INPUT_DIR = "."
DL_OUTPUT_DIR = "dl"
UL_OUTPUT_DIR = "ul"

# Chicago timezone
CHICAGO_TZ = "America/Chicago"

os.makedirs(DL_OUTPUT_DIR, exist_ok=True)
os.makedirs(UL_OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------------
# Process all aligned_experiment_*_DL.csv and *_UL.csv files
# -------------------------------------------------------------------

pattern = os.path.join(INPUT_DIR, "aligned_experiment_*_[DU]L.csv")

files = glob.glob(pattern)

for file_path in files:

    filename = os.path.basename(file_path)

    # Ignore files without explicit _DL.csv or _UL.csv ending
    if filename.endswith("_DL.csv"):
        direction = "dl"
        output_dir = DL_OUTPUT_DIR
    elif filename.endswith("_UL.csv"):
        direction = "ul"
        output_dir = UL_OUTPUT_DIR
    else:
        continue

    print(f"Processing: {filename}")

    # ----------------------------------------------------------------
    # Read CSV
    # ----------------------------------------------------------------

    df = pd.read_csv(file_path)

    # ----------------------------------------------------------------
    # Timestamp handling
    # ----------------------------------------------------------------

    # Parse as UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Convert to Chicago time
    df["timestamp"] = df["timestamp"].dt.tz_convert(CHICAGO_TZ)

    # Add hour of day (0-23)
    df["hour_of_day"] = df["timestamp"].dt.hour

    # Add ISO weekday (Mon=1 ... Sun=7)
    df["day_of_week"] = df["timestamp"].dt.isocalendar().day

    # ----------------------------------------------------------------
    # Detect first handover
    # ----------------------------------------------------------------

    first_satellite = df["satellite_id"].iloc[0]

    # Find first row where satellite changes
    handover_indices = df.index[df["satellite_id"] != first_satellite]

    # If no handover exists, skip file
    if len(handover_indices) == 0:
        print(f"  Skipping (no handover found): {filename}")
        continue

    first_handover_idx = handover_indices[0]

    # Remove all rows before first handover
    df = df.iloc[first_handover_idx:].copy()

    # ----------------------------------------------------------------
    # Compute time since handover
    # ----------------------------------------------------------------

    handover_start_time = None
    previous_satellite = None

    time_since_handover = []

    for _, row in df.iterrows():

        current_satellite = row["satellite_id"]
        current_timestamp = row["timestamp"]

        # New handover detected
        if current_satellite != previous_satellite:
            handover_start_time = current_timestamp

        delta = (
            current_timestamp - handover_start_time
        ).total_seconds()

        time_since_handover.append(delta)

        previous_satellite = current_satellite

    df["time_since_handover"] = time_since_handover

    # ----------------------------------------------------------------
    # Remove unwanted columns
    # ----------------------------------------------------------------

    columns_to_remove = ["direction", "experiment_id"]

    for col in columns_to_remove:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # ----------------------------------------------------------------
    # Rename output file
    # ----------------------------------------------------------------

    # aligned_experiment_069_20250906_075128_DL.csv
    # -> ex_20250906_075128_DL.csv

    parts = filename.split("_")

    # parts:
    # ['aligned', 'experiment', '069', '20250906', '075128', 'DL.csv']

    new_filename = f"ex_{parts[3]}_{parts[4]}_{direction}.csv"

    output_path = os.path.join(output_dir, new_filename)

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------

    df.to_csv(output_path, index=False)

    print(f"  Saved: {output_path}")

print("Done.")

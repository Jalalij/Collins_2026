#!/bin/bash

# Configuration
RAW_DATA_DIR="leo_raw_traces"
INTERVAL=1  # Frequency in seconds
ENDPOINT="http://localhost:9451/metrics"

# Create a directory to keep things organized
mkdir -p $RAW_DATA_DIR

echo "Starting RAW buffer collection... Press [CTRL+C] to stop."
echo "Data is being saved to the '$RAW_DATA_DIR' directory."

while true; do
    # Generate a precise timestamp for the filename
    # Format: YYYYMMDD_HHMMSS
    TS_FILE=$(date +"%Y%m%d_%H%M%S")
    TS_HUMAN=$(date +"%Y-%m-%d %H:%M:%S")

    # Capture the entire raw buffer to a uniquely named file
    # We use a temporary file first to ensure the write is atomic
    curl -s $ENDPOINT > "${RAW_DATA_DIR}/trace_${TS_FILE}.txt"

    # Also log a heartbeat to the console so you know it's working
    echo "[$TS_HUMAN] Raw buffer captured: trace_${TS_FILE}.txt"
    
    sleep $INTERVAL
done
# config.py - Shared configuration across all modules
from datetime import datetime
import os

# Directories
BASE_DATA_DIR = "data"

# Network configuration
SERVER_IP = "10.188.0.9"
QUECTEL_IP = "10.189.55.113"
IPERF_PORT = 5202

# Collector intervals
PING_INTERVAL    = 0.2   # seconds between ping packets
PING_PACKET_SIZE = 56    # bytes
IPERF3_INTERVAL  = 1.0   # seconds between iperf3 reports
WEATHER_INTERVAL = 1.0   # seconds between weather polls

# Experiment parameters
DURATION   = 600   # seconds per test block
SLEEP_TIME = 60    # seconds between blocks

# Log rotation
MAX_LOG_SIZE = 5 * 1024 * 1024   # 5 MB
BACKUP_COUNT = 10

# Timestamp format
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"

# config.py - Shared configuration across all modules
from datetime import datetime
import os

# Directories
BASE_DATA_DIR = "data"
WEATHER_DIR   = os.path.join(BASE_DATA_DIR, "weather")
QUECTEL_DIR   = os.path.join(BASE_DATA_DIR, "radio")
PING_DIR      = os.path.join(BASE_DATA_DIR, "ping")
IPERF_DIR     = os.path.join(BASE_DATA_DIR, "iperf")

os.makedirs(WEATHER_DIR, exist_ok=True)
os.makedirs(QUECTEL_DIR, exist_ok=True)
os.makedirs(PING_DIR,    exist_ok=True)
os.makedirs(IPERF_DIR,   exist_ok=True)

# Network configuration
SERVER_IP = "10.188.0.9"
TUNNEL_IP = "10.77.0.1"

# Collector intervals
PING_INTERVAL    = 0.2   # seconds between ping packets
PING_PACKET_SIZE = 56    # bytes
IPERF3_INTERVAL  = 0.2   # seconds between iperf3 reports
WEATHER_INTERVAL = 1.0   # seconds between weather polls

# Experiment parameters
DURATION   = 600   # seconds per test block
SLEEP_TIME = 60    # seconds between blocks

# Log rotation
MAX_LOG_SIZE = 5 * 1024 * 1024   # 5 MB
BACKUP_COUNT = 10

# Timestamp format
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"


def generate_ping_filename(extension="json"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(PING_DIR, f"{timestamp}_ping.{extension}")

def generate_iperf_filename(direction, target_bw, extension="json"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(IPERF_DIR, f"{timestamp}_{direction}_{target_bw}mbps_iperf3.{extension}")

def generate_weather_filename(extension="json"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(WEATHER_DIR, f"{timestamp}_weather.{extension}")

def generate_quectel_filename(extension="json"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(QUECTEL_DIR, f"{timestamp}_quectel.{extension}")
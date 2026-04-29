# config.py - Shared configuration across all modules
from datetime import datetime
import os

# Directories
BASE_DATA_DIR = "data"

# Network configuration
SERVER_IP = "34.174.61.96"
TUNNEL_IP = "10.88.0.1"

# Terminal configuration
LATITUDE  = 42.01361811507379
LONGITUDE = -93.65103247951744
ALTITUDE  = 300
MOBILE    = False
STARLINK_GRPC_ADDR_PORT = os.getenv("STARLINK_GRPC_ADDR_PORT", "192.168.100.1:9200")

# Collector intervals
PING_INTERVAL    = 0.2
PING_PACKET_SIZE = 56
IPERF3_INTERVAL  = 1
WEATHER_INTERVAL = 1.0
STARLINK_INTERVAL = 0.2

# Experiment parameters
DURATION   = 600
SLEEP_TIME = 60

# Log rotation
MAX_LOG_SIZE = 5 * 1024 * 1024
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

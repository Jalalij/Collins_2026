# config.py - Shared configuration across all modules
import json
from datetime import datetime
import os

# Experiment configuration
EXPERIMENT_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = "data/starlink"
PING_DIR = os.path.join(RESULTS_DIR, "ping")
IPERF_DIR = os.path.join(RESULTS_DIR, "iperf")
WEATHER_DIR = os.path.join(RESULTS_DIR, "weather")
STARLINK_DIR = os.path.join(RESULTS_DIR, "terminal")

# Create directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PING_DIR, exist_ok=True)
os.makedirs(IPERF_DIR, exist_ok=True)
os.makedirs(WEATHER_DIR, exist_ok=True)
os.makedirs(STARLINK_DIR, exist_ok=True)

# Network configuration
SERVER_IP = "10.188.0.9"
TUNNEL_IP = "10.77.0.1"

# Terminal configuration
LATITUDE  = 42.01361811507379
LONGITUDE = -93.65103247951744
ALTITUDE  = 300
MOBILE    = False  # set True if the dish is on a moving vehicle
STARLINK_GRPC_ADDR_PORT = os.getenv("STARLINK_GRPC_ADDR_PORT", "192.168.100.1:9200")


# ping configurations
PING_INTERVAL = 0.2  # seconds between ping packets
PING_PACKET_SIZE = 56  # bytes (MTU - 28) 1472 for throughput and MTU testing
IPERF3_INTERVAL = 0.2  # seconds between iperf3 reports
PROTOCOL = 'udp'
WEATHER_INTERVAL = 1.0
STARLINK_INTERVAL = 0.2

# Experiment parameters
DURATION = 600  # Test duration in seconds 600
SLEEP_TIME = 60  # Sleep between tests in seconds

# Bandwidth configurations for iperf3
BANDWIDTH_CONFIGS = [
    {"dl": 100, "ul": 20},
    {"dl": 80, "ul": 15},
    {"dl": 50, "ul": 10},
    {"dl": 20, "ul": 5}
]

# Test schedule (24-hour format)
TEST_SCHEDULE = [
    {"time": "02:30", "name": "off_peak", "duration": DURATION},      # 2 AM - Off-peak
    {"time": "08:00", "name": "morning_peak", "duration": DURATION},  # 8 AM - Morning peak
    {"time": "14:00", "name": "afternoon", "duration": DURATION},     # 2 PM - Afternoon
    {"time": "21:30", "name": "evening_peak", "duration": DURATION},  # 8 PM - Evening peak  
]

# Bandwidth configurations for each time slot
# Unified bandwidth configurations for all time slots
BANDWIDTH_CONFIGS_SCHEDULE = {
    "morning_peak": [
        {"dl": 100, "ul": 20},
        {"dl": 80, "ul": 15}, 
        {"dl": 50, "ul": 10},
        {"dl": 20, "ul": 5}
    ],
    "afternoon": [
        {"dl": 100, "ul": 20},
        {"dl": 80, "ul": 15},
        {"dl": 50, "ul": 10},
        {"dl": 20, "ul": 5}
    ],
    "evening_peak": [
        {"dl": 100, "ul": 20},
        {"dl": 80, "ul": 15},
        {"dl": 50, "ul": 10},
        {"dl": 20, "ul": 5}
    ],
    "off_peak": [
        {"dl": 100, "ul": 20},
        {"dl": 80, "ul": 15},
        {"dl": 50, "ul": 10},
        {"dl": 20, "ul": 5}
    ]
}

# Log rollover parameters for controller_schedule.py
MAX_LOG_SIZE = 5 * 1024 * 1024   # 5 MB
BACKUP_COUNT = 10                # keep up to 5 old logs

# Timestamp format for synchronization
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"

def generate_ping_filename(extension="json"):
    """Generate ping filenames"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(PING_DIR, f"{timestamp}_ping.{extension}")

def generate_iperf_filename(direction, target_bw, extension="json"):
    """Generate iperf3 filenames"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(IPERF_DIR, f"{timestamp}_{direction}_{target_bw}mbps_iperf3.{extension}")

def generate_weather_filename(extension="json"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(WEATHER_DIR, f"{timestamp}_weather.{extension}")

def generate_starlink_filename(extension="json"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(STARLINK_DIR, f"{timestamp}_starlink.{extension}")

def generate_satellite_tracker_filename(extension="json"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(STARLINK_DIR, f"{timestamp}_satellite_tracker.{extension}")
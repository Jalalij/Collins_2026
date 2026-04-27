#!/usr/bin/env python3
# main_controller.py - Coordinates a single experiment slot (Starlink)

import os
import time
import threading
import subprocess
from datetime import datetime
from network_experiment.starlink.config import (
    SLEEP_TIME, SERVER_IP, PING_INTERVAL, PING_PACKET_SIZE, IPERF3_INTERVAL, WEATHER_INTERVAL
)

DURATION      = 600
PROTOCOLS     = [
    {"name": "udp",   "protocol": "udp", "congestion": None},
    {"name": "bbr",   "protocol": "tcp", "congestion": "bbr"},
    {"name": "cubic", "protocol": "tcp", "congestion": "cubic"},
]
BANDWIDTH_CONFIGS = [
    {"dl": 100, "ul": 20},
    {"dl": 50,  "ul": 10},
]

BASE_DATA_DIR = "data"
COLLECTOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "application_scripts")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_run_dir(slot_dir, label):
    run_dir = os.path.join(slot_dir, label)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _collector(script):
    return os.path.join(COLLECTOR_DIR, script)


def _run(cmd, timeout=None, label=""):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            print(f"    [{label}] failed: {result.stderr.strip()}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"    [{label}] timed out")
        return False
    except Exception as e:
        print(f"    [{label}] error: {e}")
        return False


# --------------------------------------------------------------------------- #
# Collector launchers
# --------------------------------------------------------------------------- #

def _launch_ping(run_dir, duration, source=None):
    cmd = [
        "python3", _collector("ping_collector.py"),
        "--duration",    str(duration),
        "--server-ip",   SERVER_IP,
        "--interval",    str(PING_INTERVAL),
        "--packet-size", str(PING_PACKET_SIZE),
        "--raw-file",    os.path.join(run_dir, "ping.json"),
        "--parsed-file", os.path.join(run_dir, "ping.csv"),
    ]
    if source:
        cmd += ["--source", source]
    return cmd


def _launch_ping_cloudflare(run_dir, duration):
    return [
        "python3", _collector("ping_collector.py"),
        "--duration",    str(duration),
        "--server-ip",   "1.1.1.1",
        "--interval",    str(PING_INTERVAL),
        "--packet-size", str(PING_PACKET_SIZE),
        "--raw-file",    os.path.join(run_dir, "ping_1_1_1_1.json"),
        "--parsed-file", os.path.join(run_dir, "ping_1_1_1_1.csv"),
    ]


def _launch_starlink(run_dir, duration):
    return [
        "python3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "starlink_collector.py"),
        "--duration",    str(duration),
        "--raw-file",    os.path.join(run_dir, "starlink.json"),
        "--parsed-file", os.path.join(run_dir, "starlink.csv"),
    ]


def _launch_satellite(run_dir, duration):
    return [
        "python3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "satellite_tracker.py"),
        "--duration",    str(duration),
        "--parsed-file", os.path.join(run_dir, "satellite.csv"),
    ]


def _launch_weather(run_dir, duration):
    return [
        "python3", _collector("weather_collector.py"),
        "--duration",      str(duration),
        "--poll-interval", str(WEATHER_INTERVAL),
        "--raw-file",      os.path.join(run_dir, "weather.json"),
        "--parsed-file",   os.path.join(run_dir, "weather.csv"),
    ]


def _launch_iperf(run_dir, direction, bandwidth, duration, protocol, congestion=None):
    cmd = [
        "python3", _collector("iperf3_collector.py"),
        "--direction",   direction,
        "--bandwidth",   str(bandwidth),
        "--duration",    str(duration),
        "--server-ip",   SERVER_IP,
        "--interval",    str(IPERF3_INTERVAL),
        "--protocol",    protocol,
        "--raw-file",    os.path.join(run_dir, "iperf.json"),
        "--parsed-file", os.path.join(run_dir, "iperf.csv"),
    ]
    if congestion:
        cmd += ["--congestion", congestion]
    return cmd


# --------------------------------------------------------------------------- #
# Baseline block
# --------------------------------------------------------------------------- #

def run_baseline(slot_dir, duration):
    """ping + ping_cloudflare + starlink + satellite + weather with no iperf."""
    run_dir = _make_run_dir(slot_dir, "baseline")
    print(f"  Starting baseline → {run_dir}")

    cmds = {
        "ping":           _launch_ping(run_dir, duration),
        "ping_cloudflare": _launch_ping_cloudflare(run_dir, duration),
        "starlink":       _launch_starlink(run_dir, duration),
        "satellite":      _launch_satellite(run_dir, duration),
        "weather":        _launch_weather(run_dir, duration),
    }

    threads = {}
    results = {}

    def _run_collector(name, cmd):
        results[name] = _run(cmd, timeout=duration + 30, label=name)

    for name, cmd in cmds.items():
        t = threading.Thread(target=_run_collector, args=(name, cmd))
        t.start()
        threads[name] = t

    for name, t in threads.items():
        t.join()

    ok = all(results.values())
    print(f"  Baseline complete: " +
          ", ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in results.items()))
    return ok


# --------------------------------------------------------------------------- #
# Test block
# --------------------------------------------------------------------------- #

def run_test_block(slot_dir, label, direction, bandwidth, duration, protocol, congestion=None):
    """
    Run iperf, ping, ping_cloudflare, starlink, satellite, and weather
    simultaneously for duration seconds. All output saved under slot_dir/label/.
    """
    run_dir = _make_run_dir(slot_dir, label)
    print(f"  Starting block: {label} → {run_dir}")

    cmds = {
        "iperf":           _launch_iperf(run_dir, direction, bandwidth, duration, protocol, congestion),
        "ping":            _launch_ping(run_dir, duration),
        "ping_cloudflare": _launch_ping_cloudflare(run_dir, duration),
        "starlink":        _launch_starlink(run_dir, duration),
        "satellite":       _launch_satellite(run_dir, duration),
        "weather":         _launch_weather(run_dir, duration),
    }

    threads = {}
    results = {}

    def _run_collector(name, cmd):
        results[name] = _run(cmd, timeout=duration + 30, label=name)

    for name, cmd in cmds.items():
        t = threading.Thread(target=_run_collector, args=(name, cmd))
        t.start()
        threads[name] = t

    for name, t in threads.items():
        t.join()

    ok = all(results.values())
    print(f"  Block complete: {label} — " +
          ", ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in results.items()))
    return ok


# --------------------------------------------------------------------------- #
# Full experiment slot
# --------------------------------------------------------------------------- #

def run_experiment_slot(test_name="slot"):
    """
    Full sequence for one scheduled slot:
      1. baseline — ping + ping_cloudflare + starlink + satellite + weather, 10 min
      2. For each protocol x each bandwidth config:
           DL block -> 60s sleep -> UL block -> 60s sleep
    """
    slot_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    slot_dir = os.path.join(BASE_DATA_DIR, f"starlink_{slot_ts}")
    os.makedirs(slot_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"SLOT START: {test_name.upper()} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Slot directory: {slot_dir}")
    print(f"{'='*60}")

    run_baseline(slot_dir, DURATION)

    for proto in PROTOCOLS:
        proto_name = proto["name"]
        protocol   = proto["protocol"]
        congestion = proto["congestion"]

        for cfg in BANDWIDTH_CONFIGS:
            dl_bw = cfg["dl"]
            ul_bw = cfg["ul"]

            dl_label = f"{proto_name}_dl_{dl_bw}"
            run_test_block(slot_dir, dl_label, "DL", dl_bw, DURATION, protocol, congestion)
            print(f"  Sleeping {SLEEP_TIME}s...")
            time.sleep(SLEEP_TIME)

            ul_label = f"{proto_name}_ul_{ul_bw}"
            run_test_block(slot_dir, ul_label, "UL", ul_bw, DURATION, protocol, congestion)

            if not (proto == PROTOCOLS[-1] and cfg == BANDWIDTH_CONFIGS[-1]):
                print(f"  Sleeping {SLEEP_TIME}s...")
                time.sleep(SLEEP_TIME)

    print(f"\nSlot complete: {test_name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Data: {slot_dir}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run one Starlink experiment slot")
    parser.add_argument("--test-name", type=str, default="manual", help="Slot label")
    args = parser.parse_args()

    run_experiment_slot(test_name=args.test_name)

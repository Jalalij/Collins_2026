#!/usr/bin/env python3
# main_controller.py - Coordinates a single experiment slot

import os
import time
import threading
import subprocess
from datetime import datetime, timezone
from network_experiment.quectel.config import SLEEP_TIME

DURATION       = 600   # 10 minutes per test
PROTOCOLS      = [
    {"name": "udp",   "protocol": "udp",  "congestion": None},
    {"name": "bbr",   "protocol": "tcp",  "congestion": "bbr"},
    {"name": "cubic", "protocol": "tcp",  "congestion": "cubic"},
]
BANDWIDTH_CONFIGS = [
    {"dl": 100, "ul": 20},
    {"dl": 50,  "ul": 10},
]

BASE_DATA_DIR = "data"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_run_dir(label: str) -> str:
    """Create and return a timestamped directory for one test run."""
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(BASE_DATA_DIR, f"quectel_{label}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _run(cmd, timeout=None, label=""):
    """Run a subprocess, print errors, return True on success."""
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
# Individual collector launchers
# --------------------------------------------------------------------------- #

def _launch_ping(run_dir, duration, source=None):
    cmd = [
        "python3", "ping_collector.py",
        "--duration",    str(duration),
        "--raw-file",    os.path.join(run_dir, "ping.json"),
        "--parsed-file", os.path.join(run_dir, "ping.csv"),
    ]
    if source:
        cmd += ["--source", source]
    return cmd


def _launch_quectel(run_dir, duration):
    return [
        "python3", "quectel_collector.py",
        "--duration",    str(duration),
        "--raw-file",    os.path.join(run_dir, "quectel.json"),
        "--parsed-file", os.path.join(run_dir, "quectel.csv"),
    ]


def _launch_weather(run_dir, duration):
    return [
        "python3", "weather_collector.py",
        "--duration",    str(duration),
        "--raw-file",    os.path.join(run_dir, "weather.json"),
        "--parsed-file", os.path.join(run_dir, "weather.csv"),
    ]


def _launch_iperf(run_dir, direction, bandwidth, duration, protocol, congestion=None):
    cmd = [
        "python3", "iperf3_collector.py",
        "--direction",  direction,
        "--bandwidth",  str(bandwidth),
        "--duration",   str(duration),
        "--protocol",   protocol,
        "--raw-file",   os.path.join(run_dir, "iperf.json"),
        "--parsed-file", os.path.join(run_dir, "iperf.csv"),
    ]
    if congestion:
        cmd += ["--congestion", congestion]
    return cmd


# --------------------------------------------------------------------------- #
# Run one test block (iperf + ping + quectel + weather simultaneously)
# --------------------------------------------------------------------------- #

def run_test_block(label, direction, bandwidth, duration, protocol, congestion=None):
    """
    Run iperf, ping, quectel, and weather simultaneously for duration seconds.
    All output saved to a single timestamped directory named by label.
    """
    run_dir = _make_run_dir(label)
    print(f"  Starting block: {label} → {run_dir}")

    cmds = {
        "iperf":   _launch_iperf(run_dir, direction, bandwidth, duration, protocol, congestion),
        "ping":    _launch_ping(run_dir, duration),
        "quectel": _launch_quectel(run_dir, duration),
        "weather": _launch_weather(run_dir, duration),
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
# Run baseline ping block
# --------------------------------------------------------------------------- #

def run_baseline(duration):
    """Standalone ping-only baseline at the start of each slot."""
    run_dir = _make_run_dir("baseline_ping")
    print(f"  Starting baseline ping → {run_dir}")

    cmd = _launch_ping(run_dir, duration)
    ok  = _run(cmd, timeout=duration + 30, label="ping_baseline")
    print(f"  Baseline complete: {'OK' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
# Run a full experiment slot
# --------------------------------------------------------------------------- #

def run_experiment_slot(test_name="slot"):
    """
    Full sequence for one scheduled slot:
      1. 10-min ping baseline
      2. For each protocol × each bandwidth config:
           DL test block → 60s sleep → UL test block → 60s sleep
    """
    print(f"\n{'='*60}")
    print(f"SLOT START: {test_name.upper()} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # 1. Baseline
    run_baseline(DURATION)

    # 2. Protocol × config loop
    for proto in PROTOCOLS:
        proto_name = proto["name"]
        protocol   = proto["protocol"]
        congestion = proto["congestion"]

        for cfg in BANDWIDTH_CONFIGS:
            dl_bw = cfg["dl"]
            ul_bw = cfg["ul"]

            # DL block
            dl_label = f"{proto_name}_dl_{dl_bw}"
            run_test_block(dl_label, "DL", dl_bw, DURATION, protocol, congestion)
            print(f"  Sleeping {SLEEP_TIME}s...")
            time.sleep(SLEEP_TIME)

            # UL block
            ul_label = f"{proto_name}_ul_{ul_bw}"
            run_test_block(ul_label, "UL", ul_bw, DURATION, protocol, congestion)

            # Sleep after UL unless this is the very last block
            if not (proto == PROTOCOLS[-1] and cfg == BANDWIDTH_CONFIGS[-1]):
                print(f"  Sleeping {SLEEP_TIME}s...")
                time.sleep(SLEEP_TIME)

    print(f"\nSlot complete: {test_name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# --------------------------------------------------------------------------- #
# CLI — run a slot manually
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run one experiment slot")
    parser.add_argument("--test-name", type=str, default="manual", help="Slot label")
    args = parser.parse_args()

    run_experiment_slot(test_name=args.test_name)

#!/usr/bin/env python3
# main_controller.py - Coordinates a single experiment slot (Starlink)

import os
import time
import threading
import subprocess
from datetime import datetime
from network_experiment.starlink.config import (
    SLEEP_TIME, SERVER_IP, PING_INTERVAL, PING_PACKET_SIZE, IPERF3_INTERVAL, WEATHER_INTERVAL, TUNNEL_IP
)

DURATION      = 600
PROTOCOLS     = [
    {"name": "udp",   "protocol": "udp", "congestion": None},
    # {"name": "bbr",   "protocol": "tcp", "congestion": "bbr"},
    # {"name": "cubic", "protocol": "tcp", "congestion": "cubic"},
]
BANDWIDTH_CONFIGS = [
    {"dl": 100, "ul": 10},
    # {"dl": 200,  "ul": 20},
]

GRACE_PERIOD = 10

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
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            if proc.returncode != 0:
                print(f"    [{label}] failed: {stderr.strip()}")
                return False
            return True
        except subprocess.TimeoutExpired:
            print(f"    [{label}] timeout reached, sending SIGTERM...")
            proc.terminate()
            try:
                proc.communicate(timeout=GRACE_PERIOD)
                print(f"    [{label}] cleanly terminated")
            except subprocess.TimeoutExpired:
                print(f"    [{label}] did not stop cleanly, killing")
                proc.kill()
                proc.communicate()
            return False
    except Exception as e:
        print(f"    [{label}] error: {e}")
        return False


# --------------------------------------------------------------------------- #
# Collector launchers
# --------------------------------------------------------------------------- #

def _launch_ping_server(run_dir, duration):
    return [
        "python3", _collector("ping_collector.py"),
        "--duration",    str(duration),
        "--server-ip",   SERVER_IP,
        "--source",      "eth1",
        "--interval",    str(PING_INTERVAL),
        "--packet-size", str(PING_PACKET_SIZE),
        "--raw-file",    os.path.join(run_dir, "ping_server.json"),
        "--parsed-file", os.path.join(run_dir, "ping_server.csv"),
    ]
    
    
def _launch_ping_tunnel(run_dir, duration):
    return [
        "python3", _collector("ping_collector.py"),
        "--duration",    str(duration),
        "--server-ip",   TUNNEL_IP,
        "--source",      "wg1",
        "--interval",    str(PING_INTERVAL),
        "--packet-size", str(PING_PACKET_SIZE),
        "--raw-file",    os.path.join(run_dir, "ping_tunnel.json"),
        "--parsed-file", os.path.join(run_dir, "ping_tunnel.csv"),
    ]


def _launch_ping_cloudflare(run_dir, duration):
    return [
        "python3", _collector("ping_collector.py"),
        "--duration",    str(duration),
        "--server-ip",   "1.1.1.1",
        "--source",      "eth1",
        "--interval",    str(PING_INTERVAL),
        "--packet-size", str(PING_PACKET_SIZE),
        "--raw-file",    os.path.join(run_dir, "ping_dns.json"),
        "--parsed-file", os.path.join(run_dir, "ping_dns.csv"),
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
        "--server-ip",   TUNNEL_IP,
        "--interval",    str(IPERF3_INTERVAL),
        "--protocol",    protocol,
        "--raw-file",    os.path.join(run_dir, "iperf.json"),
        "--parsed-file", os.path.join(run_dir, "iperf.csv"),
    ]
    if congestion:
        cmd += ["--congestion", congestion]
    return cmd

def _start_collector_process(cmd):
    """Launch a collector as a background process (non-blocking)."""
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _wait_for_process(proc, name, timeout):
    """Wait for a Popen process to finish, with timeout."""
    try:
        proc.wait(timeout=timeout)
        if proc.returncode != 0:
            print(f"    [{name}] failed with code {proc.returncode}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"    [{name}] timed out")
        proc.kill()
        return False


SATELLITE_WARMUP = 30  # seconds to wait after launching satellite_tracker

# --------------------------------------------------------------------------- #
# Baseline block
# --------------------------------------------------------------------------- #

def run_baseline(slot_dir, duration):
    """ping + ping_cloudflare + starlink + satellite + weather with no iperf."""
    run_dir = _make_run_dir(slot_dir, "baseline")
    print(f"  Starting baseline → {run_dir}")

    # 1. Start satellite first and give it time to initialize
    sat_cmd = _launch_satellite(run_dir, duration + SATELLITE_WARMUP)
    print(f"  Launching satellite_tracker, waiting {SATELLITE_WARMUP}s for it to initialize...")
    sat_proc = _start_collector_process(sat_cmd)
    time.sleep(SATELLITE_WARMUP)

    # 2. Start everything else in parallel
    other_cmds = {
        "ping_server":     _launch_ping_server(run_dir, duration),
        "ping_tunnel":     _launch_ping_tunnel(run_dir, duration),
        "ping_cloudflare": _launch_ping_cloudflare(run_dir, duration),
        "starlink":        _launch_starlink(run_dir, duration),
        "weather":         _launch_weather(run_dir, duration),
    }

    threads = {}
    results = {}

    def _run_collector(name, cmd):
        results[name] = _run(cmd, timeout=duration + 30, label=name)

    for name, cmd in other_cmds.items():
        t = threading.Thread(target=_run_collector, args=(name, cmd))
        t.start()
        threads[name] = t

    for name, t in threads.items():
        t.join()

    # 3. Wait for satellite to finish
    results["satellite"] = _wait_for_process(sat_proc, "satellite", timeout=duration + SATELLITE_WARMUP + 30)

    ok = all(results.values())
    print(f"  Baseline complete: " +
          ", ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in results.items()))
    return ok


# --------------------------------------------------------------------------- #
# Test block
# --------------------------------------------------------------------------- #

def run_test_block(slot_dir, label, direction, bandwidth, duration, protocol, congestion=None):
    """
    Start satellite_tracker first, wait SATELLITE_WARMUP seconds, then launch
    all other collectors (iperf, ping, starlink, weather) simultaneously.
    """
    run_dir = _make_run_dir(slot_dir, label)
    print(f"  Starting block: {label} → {run_dir}")

    # 1. Start satellite first and give it time to initialize
    sat_cmd = _launch_satellite(run_dir, duration + SATELLITE_WARMUP)
    print(f"  Launching satellite_tracker, waiting {SATELLITE_WARMUP}s for it to initialize...")
    sat_proc = _start_collector_process(sat_cmd)
    time.sleep(SATELLITE_WARMUP)

    # 2. Start everything else in parallel
    other_cmds = {
        "iperf":           _launch_iperf(run_dir, direction, bandwidth, duration, protocol, congestion),
        "ping_server":     _launch_ping_server(run_dir, duration),
        "ping_tunnel":     _launch_ping_tunnel(run_dir, duration),
        "ping_cloudflare": _launch_ping_cloudflare(run_dir, duration),
        "starlink":        _launch_starlink(run_dir, duration),
        "weather":         _launch_weather(run_dir, duration),
    }

    threads = {}
    results = {}

    def _run_collector(name, cmd):
        results[name] = _run(cmd, timeout=duration + 30, label=name)

    for name, cmd in other_cmds.items():
        t = threading.Thread(target=_run_collector, args=(name, cmd))
        t.start()
        threads[name] = t

    for name, t in threads.items():
        t.join()

    # 3. Wait for satellite to finish
    results["satellite"] = _wait_for_process(sat_proc, "satellite", timeout=duration + SATELLITE_WARMUP + 30)

    ok = all(results.values())
    print(f"  Block complete: {label} — " +
          ", ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in results.items()))
    return ok



# --------------------------------------------------------------------------- #
# iPerf block
# --------------------------------------------------------------------------- #

def run_iperf_block(slot_dir, label, direction, bandwidth, duration, protocol, congestion=None):
    """
    Start satellite_tracker first, wait SATELLITE_WARMUP seconds, then launch
    all other collectors (iperf, ping, starlink, weather) simultaneously.
    """
    run_dir = _make_run_dir(slot_dir, label)
    print(f"  Starting block: {label} → {run_dir}")

    # 1. Start satellite first and give it time to initialize
    sat_cmd = _launch_satellite(run_dir, duration + SATELLITE_WARMUP)
    print(f"  Launching satellite_tracker, waiting {SATELLITE_WARMUP}s for it to initialize...")
    sat_proc = _start_collector_process(sat_cmd)
    time.sleep(SATELLITE_WARMUP)

    # 2. Start everything else in parallel
    other_cmds = {
        "iperf":           _launch_iperf(run_dir, direction, bandwidth, duration, protocol, congestion),
        "starlink":        _launch_starlink(run_dir, duration),
        "weather":         _launch_weather(run_dir, duration),
    }

    threads = {}
    results = {}

    def _run_collector(name, cmd):
        results[name] = _run(cmd, timeout=duration + 30, label=name)

    for name, cmd in other_cmds.items():
        t = threading.Thread(target=_run_collector, args=(name, cmd))
        t.start()
        threads[name] = t

    for name, t in threads.items():
        t.join()

    # 3. Wait for satellite to finish
    results["satellite"] = _wait_for_process(sat_proc, "satellite", timeout=duration + SATELLITE_WARMUP + 30)

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
                
    run_iperf_block(slot_dir, "iperf_udp_dl_500", "DL", 500, DURATION, "udp", None)
    print(f"  Sleeping {SLEEP_TIME}s...")
    time.sleep(SLEEP_TIME)
    run_iperf_block(slot_dir, "iperf_udp_ul_80", "UL", 80, DURATION, "udp", None)

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

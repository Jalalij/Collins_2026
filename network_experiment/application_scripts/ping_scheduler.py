#!/usr/bin/env python3
# ping_scheduler.py - Runs concurrent ping sessions to multiple IPs on a schedule

import os
import time
import threading
import subprocess
import json
from datetime import datetime

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Target IPs to ping simultaneously
PING_TARGETS = [
    "1.1.1.1",      # Cloudflare
    "34.48.127.45", # Ashburn
    "34.56.70.72",  # Council Bluffs
    "34.174.61.96", # Dallas
]

# Schedule — 24hr format hours to run (fires once per hour listed, every day)
# 3, 12, 20
SCHEDULE_HOURS = [2, 6, 11, 15, 19, 23]

# Ping parameters
PING_DURATION    = 600    # seconds per session
PING_INTERVAL    = 0.2    # seconds between pings
PING_PACKET_SIZE = 56     # bytes

# Output base directory
BASE_DATA_DIR    = "data/ping_sessions"

# Path to ping_collector.py
PING_COLLECTOR   = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "application_scripts", "ping_collector.py"
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _safe_ip(ip):
    """Convert IP to filesystem-safe string."""
    return ip.replace(".", "_")


def _make_session_dir():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d  = os.path.join(BASE_DATA_DIR, f"ping_{ts}")
    os.makedirs(d, exist_ok=True)
    return d


def _run_ping(target_ip, session_dir):
    """Launch ping_collector for one target IP."""
    safe   = _safe_ip(target_ip)
    cmd    = [
        "python3", PING_COLLECTOR,
        "--duration",    str(PING_DURATION),
        "--server-ip",   target_ip,
        "--interval",    str(PING_INTERVAL),
        "--packet-size", str(PING_PACKET_SIZE),
        "--raw-file",    os.path.join(session_dir, f"ping_{safe}.json"),
        "--parsed-file", os.path.join(session_dir, f"ping_{safe}.csv"),
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = proc.communicate(timeout=PING_DURATION + 30)
            if proc.returncode != 0:
                print(f"  [{target_ip}] failed: {stderr.strip()}")
            else:
                print(f"  [{target_ip}] complete")
        except subprocess.TimeoutExpired:
            print(f"  [{target_ip}] timeout, terminating...")
            proc.terminate()
            try:
                proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
    except Exception as e:
        print(f"  [{target_ip}] error: {e}")


def run_session():
    """Fire all ping targets simultaneously in threads."""
    session_dir = _make_session_dir()
    print(f"\n{'='*60}")
    print(f"PING SESSION at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Targets: {PING_TARGETS}")
    print(f"Directory: {session_dir}")
    print(f"{'='*60}")

    threads = []
    for ip in PING_TARGETS:
        t = threading.Thread(target=_run_ping, args=(ip, session_dir))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print(f"Session complete: {session_dir}")

    # Log session
    log_path = os.path.join(BASE_DATA_DIR, "sessions.log")
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "time":      datetime.now().isoformat(),
            "targets":   PING_TARGETS,
            "directory": session_dir,
        }) + "\n")


# --------------------------------------------------------------------------- #
# Scheduler loop
# --------------------------------------------------------------------------- #

def already_ran_today(hour):
    """Track which hours have already fired today to avoid double-firing."""
    return _ran_today.get(f"{datetime.now().date()}_{hour}", False)


def mark_ran_today(hour):
    today = datetime.now().date()
    # Clear yesterday's entries
    stale = [k for k in _ran_today if not k.startswith(str(today))]
    for k in stale:
        del _ran_today[k]
    _ran_today[f"{today}_{hour}"] = True


_ran_today = {}


def main():
    os.makedirs(BASE_DATA_DIR, exist_ok=True)

    print(f"Ping scheduler starting")
    print(f"Targets:        {PING_TARGETS}")
    print(f"Schedule hours: {SCHEDULE_HOURS}")
    print(f"Duration:       {PING_DURATION}s per session")
    print(f"Output:         {BASE_DATA_DIR}")
    print(f"Checking every 30s — Ctrl+C to stop\n")

    while True:
        now  = datetime.now()
        hour = now.hour

        if hour in SCHEDULE_HOURS and not already_ran_today(hour):
            mark_ran_today(hour)
            # Run in a thread so the scheduler loop stays responsive
            t = threading.Thread(target=run_session, daemon=True)
            t.start()

        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] heartbeat — next slots: "
              f"{[h for h in SCHEDULE_HOURS if not already_ran_today(h)]}")
        time.sleep(30)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutting down ping scheduler")
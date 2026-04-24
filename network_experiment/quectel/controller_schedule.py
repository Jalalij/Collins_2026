#!/usr/bin/env python3
# controller_schedule.py - Scheduler layer (top layer)

import schedule
import time
import threading
import json
import os
from datetime import datetime
from network_experiment.quectel.config import MAX_LOG_SIZE, BACKUP_COUNT

from network_experiment.quectel.main_controller import run_experiment_slot

# Scheduled slots: (time_str, slot_name)
SLOT_SCHEDULE = [
    ("03:00", "off_peak"),
    ("12:00", "midday"),
    ("20:00", "evening"),
]


class ScheduleController:
    def __init__(self):
        self.running          = False
        self.scheduler_thread = None
        self._setup_schedule()

    def _setup_schedule(self):
        for time_str, slot_name in SLOT_SCHEDULE:
            schedule.every().day.at(time_str).do(
                self._execute_slot, slot_name=slot_name
            )
            print(f"Scheduled slot '{slot_name}' at {time_str}")

    def _execute_slot(self, slot_name):
        print(f"\n{'='*60}")
        print(f"SCHEDULED SLOT: {slot_name.upper()} at {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}")

        try:
            run_experiment_slot(test_name=slot_name)
            self._log_session(slot_name, success=True)
        except Exception as e:
            print(f"Slot {slot_name} failed with error: {e}")
            self._log_session(slot_name, success=False, error=str(e))

    def _log_session(self, slot_name, success=True, error=None):
        """Append a session log entry with rotation."""
        log_path = "test_sessions.log"

        if os.path.exists(log_path) and os.path.getsize(log_path) > MAX_LOG_SIZE:
            for i in range(BACKUP_COUNT, 0, -1):
                old = f"{log_path}.{i}"
                new = f"{log_path}.{i+1}"
                if os.path.exists(old):
                    os.rename(old, new)
            os.rename(log_path, f"{log_path}.1")

        entry = {
            "slot_name":  slot_name,
            "time":       datetime.now().isoformat(),
            "success":    success,
        }
        if error:
            entry["error"] = error

        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        print(f"Logged session: {slot_name} — {'OK' if success else 'FAILED'}")

    def start(self):
        self.running = True
        print("Starting schedule controller...")
        print("Slots:", [f"{t} ({n})" for t, n in SLOT_SCHEDULE])

        def _loop():
            while self.running:
                schedule.run_pending()
                time.sleep(30)
                print(f"[{datetime.now().isoformat()}] Scheduler heartbeat")

        self.scheduler_thread = threading.Thread(target=_loop, daemon=True)
        self.scheduler_thread.start()
        return self

    def stop(self):
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
        print("Schedule controller stopped")

    def run_manual(self, slot_name):
        """Trigger a slot manually for testing."""
        print(f"Manual run: {slot_name}")
        self._execute_slot(slot_name=slot_name)


def get_schedule_status():
    return {
        "current_time": datetime.now().isoformat(),
        "next_slots":   [
            {"slot": n, "next_run": job.next_run.isoformat() if job.next_run else None}
            for job, (_, n) in zip(schedule.get_jobs(), SLOT_SCHEDULE)
        ],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quectel experiment scheduler")
    parser.add_argument("--manual", type=str, default=None,
                        help="Run a slot immediately by name (off_peak, midday, evening)")
    args = parser.parse_args()

    controller = ScheduleController()

    if args.manual:
        controller.run_manual(args.manual)
    else:
        try:
            controller.start()
            print("\nScheduler running — Ctrl+C to stop")
            print(json.dumps(get_schedule_status(), indent=2))
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("\nShutting down...")
            controller.stop()

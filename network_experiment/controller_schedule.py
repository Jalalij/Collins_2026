#!/usr/bin/env python3
# controller_schedule.py - Scheduler layer (top layer)

import schedule
import time
import threading
import json
import os
from datetime import datetime
from config import TEST_SCHEDULE, BANDWIDTH_CONFIGS_SCHEDULE, MAX_LOG_SIZE, BACKUP_COUNT

# Import from main_controller layer
from main_controller import run_experiment_set

class ScheduleController:
    def __init__(self):
        self.running = False
        self.scheduler_thread = None
        self.setup_schedule()
    
    def setup_schedule(self):
        """Setup all scheduled tests from config"""
        for schedule_config in TEST_SCHEDULE:
            time_str = schedule_config["time"]
            test_name = schedule_config["name"]
            duration = schedule_config["duration"]
            
            # Schedule each test time
            schedule.every().day.at(time_str).do(
                self.execute_scheduled_test,
                test_name=test_name,
                duration=duration
            )
            print(f"Scheduled {test_name} test at {time_str}")
    
    def execute_scheduled_test(self, test_name, duration):
        """Execute a scheduled test - calls main_controller layer"""
        print(f"\n{'='*60}")
        print(f"EXECUTING SCHEDULED TEST: {test_name.upper()} at {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}")
        
        # Get configuration for this time slot
        configs = BANDWIDTH_CONFIGS_SCHEDULE.get(test_name, [])
        
        if not configs:
            print(f"No configurations found for {test_name}")
            return
        
        # Call main_controller layer to run the actual tests
        results = run_experiment_set(bandwidth_configs=configs, duration=duration, test_name=test_name)
        
        # Log results
        self.log_test_session_roration(test_name, results, duration)
        
        print(f"\nCompleted {test_name} test: {len(results)} configurations")
    
    def log_test_session_roration(self, test_name, results, duration):
        """Log complete test session results with rotation"""
        log_path = 'test_sessions.log'

        # Rotate if too big
        if os.path.exists(log_path) and os.path.getsize(log_path) > MAX_LOG_SIZE:
            # Shift old logs: test_sessions.log.4 → .5, ... log.1 → .2
            for i in range(BACKUP_COUNT, 0, -1):
                old = f"{log_path}.{i}"
                new = f"{log_path}.{i+1}"
                if os.path.exists(old):
                    os.rename(old, new)
            # Move current log to .1
            os.rename(log_path, f"{log_path}.1")

        session_log = {
            'test_name': test_name,
            'start_time': datetime.now().isoformat(),
            'duration': duration,
            'configurations': len(results),
            'results': results,
            'success_rate': sum(1 for r in results if r['dl_success'] and r['ul_success']) / len(results) * 100
        }

        # Append new log entry
        with open(log_path, 'a') as f:
            f.write(json.dumps(session_log) + '\n')

        print(f"Logged {test_name} session: {session_log['success_rate']:.1f}% success")

    def log_test_session(self, test_name, results, duration):
        """Log complete test session results"""
        session_log = {
            'test_name': test_name,
            'start_time': datetime.now().isoformat(),
            'duration': duration,
            'configurations': len(results),
            'results': results,
            'success_rate': sum(1 for r in results if r['dl_success'] and r['ul_success']) / len(results) * 100
        }
        
        # Append to session log
        with open('test_sessions.log', 'a') as f:
            f.write(json.dumps(session_log) + '\n')
        
        print(f"Logged {test_name} session: {session_log['success_rate']:.1f}% success")
    
    def start(self):
        """Start the scheduler daemon"""
        self.running = True
        print("Starting schedule controller daemon...")
        print("Scheduled tests will run at:", [f"{t['time']} ({t['name']})" for t in TEST_SCHEDULE])
        
        def scheduler_loop():
            while self.running:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
                print(f"[{datetime.now().isoformat()}] Scheduler heartbeat")
        
        self.scheduler_thread = threading.Thread(target=scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        return self
    
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
        print("Schedule controller stopped")
    
    def run_manual_test(self, test_name, duration=600):
        """Run a test manually (for debugging)"""
        print(f"Running manual test: {test_name}")
        self.execute_scheduled_test(test_name, duration)

# Utility functions
def get_schedule_status():
    """Get current schedule status"""
    status = {
        'current_time': datetime.now().isoformat(),
        'next_tests': [],
        'active': False
    }
    
    for job in schedule.get_jobs():
        status['next_tests'].append({
            'function': job.job_func.__name__,
            'next_run': job.next_run.isoformat() if job.next_run else None
        })
    
    return status

if __name__ == "__main__":
    controller = ScheduleController()
    
    try:
        # Start the scheduler
        controller.start()
        
        print("\nSchedule Controller Running - Press Ctrl+C to stop")
        print("Current schedule status:")
        print(json.dumps(get_schedule_status(), indent=2))
        
        # Keep main thread alive
        while True:
            time.sleep(3600)
            
    except KeyboardInterrupt:
        print("\nShutting down schedule controller...")
        controller.stop()
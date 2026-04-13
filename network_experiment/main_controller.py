#!/usr/bin/env python3
# main_controller.py - Coordinates the entire experiment

import subprocess
import time
import json
import threading
from datetime import datetime
from config import DURATION, SLEEP_TIME, BANDWIDTH_CONFIGS, EXPERIMENT_TIME, RESULTS_DIR

def run_ping_test(total_duration, test_name=None):
    """Run ping test in a separate thread - measures round-trip time"""
    try:
        if test_name:
            print(f"    [{test_name}] Starting ping test (round-trip measurement)")
        else:
            print(f"    Starting ping test (round-trip measurement)")
        ping_proc = subprocess.run([
            'python3', 'ping_collector.py',
            '--duration', str(total_duration)
        ], capture_output=True, text=True, timeout=total_duration + 10)
        
        return ping_proc.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"    Ping test timed out")
        return False
    except Exception as e:
        print(f"    Ping test error: {e}")
        return False

def run_iperf3_test(direction, bandwidth, duration, test_name=None):
    """Run iperf3 test"""
    try:
        if test_name:
            print(f"    [{test_name}] Starting iperf3 {direction} test: {bandwidth}Mbps")
        else:
            print(f"    Starting iperf3 {direction} test: {bandwidth}Mbps")
        iperf_proc = subprocess.run([
            'python3', 'iperf3_collector.py', 
            '--direction', direction,
            '--bandwidth', str(bandwidth),
            '--duration', str(duration)
        ], capture_output=True, text=True, timeout=duration + 30)
        
        if iperf_proc.returncode == 0:
            print(f"    iperf3 {direction} test completed successfully")
            return True
        else:
            print(f"    iperf3 {direction} test failed: {iperf_proc.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"    iperf3 {direction} test timed out")
        return False
    except Exception as e:
        print(f"    iperf3 {direction} test error: {e}")
        return False

def run_experiment_set(bandwidth_configs=BANDWIDTH_CONFIGS, duration=DURATION, test_name=None):
    """Run all experiments with different bandwidth configurations"""
    results = []
    experiment_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for config in bandwidth_configs:
        dl_bw = config["dl"]
        ul_bw = config["ul"]
        
        if test_name:
            print(f"\n[{test_name}] Starting experiment: DL={dl_bw}Mbps, UL={ul_bw}Mbps")
        else:
            print(f"\nStarting experiment: DL={dl_bw}Mbps, UL={ul_bw}Mbps")
        
        # Calculate total duration for ping (covers both iperf3 tests)
        total_ping_duration = duration * 2  # DL test (600) + UL test (600) = 1200s
        
        # Start SINGLE ping test in background thread
        ping_success = [None]  # Use list to store result from thread
        ping_thread = threading.Thread(
            target=lambda: ping_success.__setitem__(0, run_ping_test(total_ping_duration))
        )
        ping_thread.start()
        
        # # Wait a moment for ping to start
        # time.sleep(2)
        
        # Run iperf3 tests SEQUENTIALLY
        print("  Starting iperf3 tests sequentially...")
        
        # 1. Run iperf3 download test (DURATION seconds)
        iperf_dl_success = run_iperf3_test("DL", dl_bw, duration)
        
        # 2. Run iperf3 upload test (DURATION seconds)
        iperf_ul_success = run_iperf3_test("UL", ul_bw, duration)
        
        # Wait for ping thread to complete
        print("  Waiting for ping test to complete...")
        ping_thread.join(timeout=total_ping_duration + 10)
        
        # Collect experiment results
        experiment_data = {
            'test_name': test_name,
            'dl_bandwidth': dl_bw,
            'ul_bandwidth': ul_bw,
            'experiment_endtime': datetime.now().isoformat(),
            'duration': duration,
            'ping_success': ping_success[0],
            'dl_success': iperf_dl_success,
            'ul_success': iperf_ul_success,
            'total_ping_duration': total_ping_duration
        }
        
        results.append(experiment_data)
        
        print(f"Experiment completed: DL={dl_bw}Mbps, UL={ul_bw}Mbps")
        print(f"  Ping: {'Success' if ping_success[0] else 'Failed'}")
        print(f"  iperf3 DL: {'Success' if iperf_dl_success else 'Failed'}")
        print(f"  iperf3 UL: {'Success' if iperf_ul_success else 'Failed'}")
        
        # Sleep between experiments except after the last one
        if config != bandwidth_configs[-1]:
            print(f"Sleeping for {SLEEP_TIME} seconds...")
            time.sleep(SLEEP_TIME)
    
    # Save experiment metadata
    metadata_file = f"{RESULTS_DIR}/experiment_metadata_{experiment_time}.json"
    with open(metadata_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAll experiments completed. Metadata saved to {metadata_file}")
    
    # Print summary
    print("\n=== Experiment Summary ===")
    for i, result in enumerate(results):
        print(f"Config {i+1}: DL={result['dl_bandwidth']}Mbps, UL={result['ul_bandwidth']}Mbps")
        print(f"  Ping: {result['ping_success']}")
        print(f"  iperf3: DL={result['dl_success']}, UL={result['ul_success']}")
    
    return results

if __name__ == "__main__":
    run_experiment_set()
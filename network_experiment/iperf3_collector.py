#!/usr/bin/env python3
# iperf3_collector.py - Collects iperf3 statistics with timestamps

import subprocess
import json
import csv
import argparse
from datetime import datetime, timedelta, timezone
from config import SERVER_IP, TIMESTAMP_FORMAT, generate_iperf_filename, IPERF3_INTERVAL

def run_iperf3_test(direction, bandwidth, duration):
    """Run iperf3 test and collect results with timestamps"""
    # Generate filenames
    raw_output_file = generate_iperf_filename(direction, bandwidth, "json")
    parsed_output_file = generate_iperf_filename(direction, bandwidth, "csv")
    
    # Server iperf: iperf3 –s –B 10.77.0.1
    # Client iperf uplink: sudo iperf3 -c 10.77.0.1 -B 10.189.55.12 -b 2G -i 1 -u -t 10
    # Client iperf downlink: sudo iperf3 -c 10.77.0.1 -B 10.189.55.12 -R -b 2G -i 1 -u -t 10 --reverse
    if direction == 'DL':
        cmd = [
            'iperf3', '-c', SERVER_IP, 
            '--bandwidth', f'{bandwidth}M',
            '--time', str(duration),
            '--interval', str(IPERF3_INTERVAL),
            '--json',
            '--udp',
            '--reverse',  # Server sends to client for download
            '--get-server-output' # optional
        ]
    else:  # UL
        cmd = [
            'iperf3', '-c', SERVER_IP,
            '--bandwidth', f'{bandwidth}M',
            '--time', str(duration),
            '--interval', str(IPERF3_INTERVAL),
            '--udp',
            '--json',
            '--get-server-output' # must have for receiver-side lost_percent and jitter_ms stats
        ]
    
    experiment_time = datetime.now(timezone.utc)
    experiment_timestamp = experiment_time.strftime(TIMESTAMP_FORMAT) + "Z"
    
    try:
        # Run iperf3 and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 30)
        
        # Save raw output as JSON
        raw_data = {
            "command": " ".join(cmd),
            "start_time": experiment_time.isoformat(),
            "direction": direction,
            "target_bandwidth": bandwidth,
            "duration": duration,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
        
        with open(raw_output_file, 'w') as f:
            json.dump(raw_data, f, indent=2)
        
        if result.returncode == 0:
            # Parse and process iperf3 data
            data = json.loads(result.stdout)
            iperf_data = []
            
            # Extract interval data
            for interval in data.get('intervals', []):
                for stream in interval.get('streams', []):
                    # Calculate absolute timestamp
                    interval_start = experiment_time + timedelta(seconds=stream['start'])
                    
                    iperf_data.append({
                        'timestamp': interval_start.strftime(TIMESTAMP_FORMAT) + "Z",
                        'interval_start': stream['start'],
                        'interval_end': stream['end'],
                        'throughput_mbps': stream.get('bits_per_second', 0) / 1e6,
                        'jitter_ms': stream.get('jitter_ms', 0),
                        'packet_loss_percent': stream.get('lost_percent', 0),
                        'retransmits': stream.get('retransmits', 0),
                        'direction': direction,
                        'target_bandwidth': bandwidth,
                        'experiment_time': experiment_timestamp
                    })
            
            # Save parsed data to CSV
            if iperf_data:
                with open(parsed_output_file, 'w', newline='') as f:
                    fieldnames = [
                        'timestamp', 'interval_start', 'interval_end', 
                        'throughput_mbps', 'jitter_ms', 'packet_loss_percent', 
                        'retransmits', 'direction', 'target_bandwidth', 'experiment_time'
                    ]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(iperf_data)
            
            return raw_output_file, parsed_output_file, len(iperf_data)
        else:
            print(f"iperf3 {direction} failed: {result.stderr}")
            return raw_output_file, None, 0
            
    except subprocess.TimeoutExpired:
        print(f"iperf3 {direction} timed out")
        return raw_output_file, None, 0
    except Exception as e:
        print(f"iperf3 {direction} error: {e}")
        return raw_output_file, None, 0

def main():
    parser = argparse.ArgumentParser(description="Collect iperf3 statistics")
    parser.add_argument("--direction", type=str, required=True, choices=["DL", "UL"], help="Test direction")
    parser.add_argument("--bandwidth", type=int, required=True, help="Target bandwidth in Mbps")
    parser.add_argument("--duration", type=int, default=120, help="Test duration in seconds")
    
    args = parser.parse_args()
    
    print(f"Starting iperf3 test: direction={args.direction}, bandwidth={args.bandwidth}Mbps, duration={args.duration}s")
    
    raw_file, parsed_file, interval_count = run_iperf3_test(
        args.direction, args.bandwidth, args.duration
    )
    
    if raw_file and parsed_file:
        print(f"iperf3 raw data saved to: {raw_file}")
        print(f"iperf3 parsed data saved to: {parsed_file}")
        print(f"Collected {interval_count} measurement intervals")
    else:
        print("iperf3 test failed")

if __name__ == "__main__":
    main()

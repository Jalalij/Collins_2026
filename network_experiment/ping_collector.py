#!/usr/bin/env python3
# ping_collector.py - Collects ping statistics with timestamps

import subprocess
import re
import json
import argparse
from datetime import datetime, timezone
from config import SERVER_IP, TIMESTAMP_FORMAT, generate_ping_filename, PING_INTERVAL, PING_PACKET_SIZE

def collect_ping_data(duration):
    """Collect ping data with timestamps"""
    # Generate filenames
    raw_output_file = generate_ping_filename("json")
    parsed_output_file = generate_ping_filename("csv")
    
    # Calculate number of ping packets
    packet_count = int(duration / PING_INTERVAL)
    
    # Run ping command
    cmd = [
        'ping', SERVER_IP, 
        '-D',  # Timestamp
        '-i', str(PING_INTERVAL),
        '-s', str(PING_PACKET_SIZE),
        '-c', str(packet_count)
    ]
    
    ping_data = []
    start_time = datetime.now(timezone.utc)
    experiment_timestamp = start_time.strftime(TIMESTAMP_FORMAT) + "Z"
    
    try:
        # Run ping and capture output
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        # Save raw output as JSON
        raw_data = {
            "command": " ".join(cmd),
            "start_time": start_time.isoformat(),
            "duration": duration,
            "output": stdout,
            "error": stderr,
            "return_code": process.returncode
        }
        
        with open(raw_output_file, 'w') as f:
            json.dump(raw_data, f, indent=2)
        
        # Parse the output
        pattern = r'\[(\d+\.\d+)\].*icmp_seq=(\d+).*time=(\d+\.?\d*) ms'
        
        for line in stdout.split('\n'):
            match = re.search(pattern, line)
            if match:
                timestamp = float(match.group(1))
                seq = int(match.group(2))
                rtt = float(match.group(3))
                
                ping_data.append({
                    'timestamp': datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime(TIMESTAMP_FORMAT) + "Z",
                    'sequence': seq,
                    'rtt_ms': rtt,
                    'experiment_time': experiment_timestamp
                })
        
        # Save parsed data to CSV
        if ping_data:
            with open(parsed_output_file, 'w') as f:
                # Write CSV header
                f.write("timestamp,sequence,rtt_ms,experiment_time\n")
                
                # Write data rows
                for data in ping_data:
                    f.write(f"{data['timestamp']},{data['sequence']},{data['rtt_ms']},{data['experiment_time']}\n")
        
        return raw_output_file, parsed_output_file, len(ping_data)
        
    except Exception as e:
        print(f"Ping collection error: {e}")
        return None, None, 0

def main():
    parser = argparse.ArgumentParser(description="Collect ping statistics")
    parser.add_argument("--duration", type=int, default=120, help="Test duration in seconds")
    
    args = parser.parse_args()
    
    print(f"Starting ping test: duration={args.duration}s")
    
    raw_file, parsed_file, packet_count = collect_ping_data(
        args.duration
    )
    
    if raw_file and parsed_file:
        print(f"Ping raw data saved to: {raw_file}")
        print(f"Ping parsed data saved to: {parsed_file}")
        print(f"Collected {packet_count} ping measurements")
    else:
        print("Ping test failed")

if __name__ == "__main__":
    main()
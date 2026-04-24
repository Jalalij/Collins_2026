#!/usr/bin/env python3
# ping_collector.py - Collects ping statistics with timestamps
import subprocess
import re
import json
import argparse
from datetime import datetime, timezone
from network_experiment.testing.config import SERVER_IP, TIMESTAMP_FORMAT, generate_ping_filename, PING_INTERVAL, PING_PACKET_SIZE

def collect_ping_data(duration, source=None):
    """Collect ping data with timestamps"""
    raw_output_file = generate_ping_filename("json")
    parsed_output_file = generate_ping_filename("csv")

    packet_count = int(duration / PING_INTERVAL)

    cmd = [
        'ping', SERVER_IP,
        '-D',
        '-i', str(PING_INTERVAL),
        '-s', str(PING_PACKET_SIZE),
        '-c', str(packet_count)
    ]

    # -I accepts either an interface name (e.g. eth0) or a source IP address
    if source:
        cmd += ['-I', source]

    ping_data = []
    start_time = datetime.now(timezone.utc)
    experiment_timestamp = start_time.strftime(TIMESTAMP_FORMAT) + "Z"

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        try:
            stdout, stderr = process.communicate(timeout=duration + 5)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            print(f"Ping timed out after {duration + 5}s and was killed")
            return raw_output_file, None, 0

        raw_data = {
            "command": " ".join(cmd),
            "start_time": start_time.isoformat(),
            "duration": duration,
            "source": source,
            "output": stdout,
            "error": stderr,
            "return_code": process.returncode
        }

        with open(raw_output_file, 'w') as f:
            json.dump(raw_data, f, indent=2)

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
                    'source': source or '',
                    'experiment_time': experiment_timestamp
                })

        if ping_data:
            with open(parsed_output_file, 'w') as f:
                f.write("timestamp,sequence,rtt_ms,source,experiment_time\n")
                for data in ping_data:
                    f.write(f"{data['timestamp']},{data['sequence']},{data['rtt_ms']},{data['source']},{data['experiment_time']}\n")

        return raw_output_file, parsed_output_file, len(ping_data)

    except Exception as e:
        print(f"Ping collection error: {e}")
        return None, None, 0


def main():
    parser = argparse.ArgumentParser(description="Collect ping statistics")
    parser.add_argument("--duration", type=int, default=120,
                        help="Test duration in seconds")
    parser.add_argument("--source", type=str, default=None,
                        help="Source IP address or interface name (e.g. 10.0.0.1 or eth0)")

    args = parser.parse_args()

    src_label = f", source={args.source}" if args.source else ""
    print(f"Starting ping test: duration={args.duration}s{src_label}")

    raw_file, parsed_file, packet_count = collect_ping_data(
        args.duration,
        source=args.source
    )

    if raw_file and parsed_file:
        print(f"Ping raw data saved to: {raw_file}")
        print(f"Ping parsed data saved to: {parsed_file}")
        print(f"Collected {packet_count} ping measurements")
    else:
        print("Ping test failed")

if __name__ == "__main__":
    main()
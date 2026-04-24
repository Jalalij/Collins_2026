#!/usr/bin/env python3
# iperf3_collector.py - Collects iperf3 statistics with timestamps

import subprocess
import json
import csv
import argparse
from datetime import datetime, timedelta, timezone
from network_experiment.testing.config import SERVER_IP, TIMESTAMP_FORMAT, generate_iperf_filename, IPERF3_INTERVAL

def run_iperf3_test(direction, bandwidth, duration, protocol='udp', congestion=None, port=5201):
    """Run iperf3 test and collect results with timestamps"""
    # Generate filenames
    raw_output_file = generate_iperf_filename(direction, bandwidth, "json")
    parsed_output_file = generate_iperf_filename(direction, bandwidth, "csv")

    # Base flags common to both directions
    cmd = [
        'iperf3', '-c', SERVER_IP,
        '--port', str(port),
        '--bandwidth', f'{bandwidth}M',
        '--time', str(duration),
        '--interval', str(IPERF3_INTERVAL),
        '--json',
    ]

    if protocol == 'udp':
        cmd.append('--udp')
    elif protocol == 'tcp' and congestion:
        cmd += ['--congestion', congestion]

    if direction == 'DL':
        cmd += ['--reverse', '--get-server-output']
    else:  # UL
        cmd.append('--get-server-output')

    experiment_time = datetime.now(timezone.utc)
    experiment_timestamp = experiment_time.strftime(TIMESTAMP_FORMAT) + "Z"

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 30)

        raw_data = {
            "command": " ".join(cmd),
            "start_time": experiment_time.isoformat(),
            "direction": direction,
            "protocol": protocol,
            "congestion_control": congestion,
            "port": port,
            "target_bandwidth": bandwidth,
            "duration": duration,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }

        with open(raw_output_file, 'w') as f:
            json.dump(raw_data, f, indent=2)

        if result.returncode == 0:
            data = json.loads(result.stdout)
            iperf_data = []

            for interval in data.get('intervals', []):
                for stream in interval.get('streams', []):
                    interval_start = experiment_time + timedelta(seconds=stream['start'])

                    iperf_data.append({
                        'timestamp': interval_start.strftime(TIMESTAMP_FORMAT) + "Z",
                        'interval_start': stream['start'],
                        'interval_end': stream['end'],
                        'throughput_mbps': stream.get('bits_per_second', 0) / 1e6,
                        'jitter_ms': stream.get('jitter_ms', 0),           # UDP only
                        'packet_loss_percent': stream.get('lost_percent', 0),  # UDP only
                        'retransmits': stream.get('retransmits', 0),        # TCP only
                        'rtt_ms': stream.get('rtt', 0) / 1e3,              # TCP only (us -> ms)
                        'direction': direction,
                        'protocol': protocol,
                        'congestion_control': congestion or 'N/A',
                        'port': port,
                        'target_bandwidth': bandwidth,
                        'experiment_time': experiment_timestamp
                    })

            if iperf_data:
                with open(parsed_output_file, 'w', newline='') as f:
                    fieldnames = [
                        'timestamp', 'interval_start', 'interval_end',
                        'throughput_mbps', 'jitter_ms', 'packet_loss_percent',
                        'retransmits', 'rtt_ms', 'direction', 'protocol',
                        'congestion_control', 'port', 'target_bandwidth', 'experiment_time'
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
    parser.add_argument("--direction", type=str, required=True, choices=["DL", "UL"],
                        help="Test direction")
    parser.add_argument("--bandwidth", type=int, required=True,
                        help="Target bandwidth in Mbps")
    parser.add_argument("--duration", type=int, default=120,
                        help="Test duration in seconds")
    parser.add_argument("--protocol", type=str, default="udp", choices=["udp", "tcp"],
                        help="Transport protocol (default: udp)")
    parser.add_argument("--congestion", type=str, choices=["cubic", "bbr"],
                        help="TCP congestion control algorithm (tcp only; requires root on Linux)")
    parser.add_argument("--port", type=int, default=5201,
                        help="iperf3 server port (default: 5201)")

    args = parser.parse_args()

    if args.congestion and args.protocol != 'tcp':
        parser.error("--congestion requires --protocol tcp")

    proto_label = args.protocol.upper()
    cc_label = f" [{args.congestion}]" if args.congestion else ""
    print(f"Starting iperf3 test: direction={args.direction}, "
          f"protocol={proto_label}{cc_label}, "
          f"bandwidth={args.bandwidth}Mbps, duration={args.duration}s, port={args.port}")

    raw_file, parsed_file, interval_count = run_iperf3_test(
        args.direction, args.bandwidth, args.duration,
        protocol=args.protocol,
        congestion=args.congestion,
        port=args.port
    )

    if raw_file and parsed_file:
        print(f"iperf3 raw data saved to: {raw_file}")
        print(f"iperf3 parsed data saved to: {parsed_file}")
        print(f"Collected {interval_count} measurement intervals")
    else:
        print("iperf3 test failed")

if __name__ == "__main__":
    main()
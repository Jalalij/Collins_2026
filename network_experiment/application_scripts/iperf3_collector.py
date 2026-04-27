#!/usr/bin/env python3
# iperf3_collector.py - Collects iperf3 statistics with timestamps

import subprocess
import json
import csv
import argparse
from datetime import datetime, timedelta, timezone

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"


def run_iperf3_test(direction, bandwidth, duration, server_ip, raw_output_file, parsed_output_file,
                    protocol='udp', congestion=None, port=5201, interval=0.2):
    """Run iperf3 test and collect results with timestamps."""
    cmd = [
        'iperf3', '-c', server_ip,
        '--port',      str(port),
        '--bandwidth', f'{bandwidth}M',
        '--time',      str(duration),
        '--interval',  str(interval),
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

    experiment_time      = datetime.now(timezone.utc)
    experiment_timestamp = experiment_time.strftime(TIMESTAMP_FORMAT) + "Z"

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 30)

        # Raw JSON
        with open(raw_output_file, 'w') as f:
            json.dump({
                "command":            " ".join(cmd),
                "start_time":         experiment_time.isoformat(),
                "direction":          direction,
                "protocol":           protocol,
                "congestion_control": congestion,
                "port":               port,
                "server_ip":          server_ip,
                "target_bandwidth":   bandwidth,
                "duration":           duration,
                "interval":           interval,
                "stdout":             result.stdout,
                "stderr":             result.stderr,
                "return_code":        result.returncode,
            }, f, indent=2)

        if result.returncode == 0:
            data       = json.loads(result.stdout)
            iperf_data = []

            if direction == 'DL':
                # cwnd/retr come from server (sender) intervals
                client_intervals = data.get('intervals', [])
                server_intervals = data.get('server_output', {}).get('intervals', [])

                # build a retr/cwnd lookup by interval index from server side
                server_lookup = {}
                for i, iv in enumerate(server_intervals):
                    for stream in iv.get('streams', []):
                        server_lookup[i] = {
                            'retransmits': stream.get('retransmits', 0),
                            'cwnd_bytes':  stream.get('snd_cwnd', 0),
                        }

                for i, interval_block in enumerate(client_intervals):
                    for stream in interval_block.get('streams', []):
                        interval_start = experiment_time + timedelta(seconds=stream['start'])
                        server_stats   = server_lookup.get(i, {'retransmits': None, 'cwnd_bytes': None})
                        iperf_data.append({
                            'timestamp':           interval_start.strftime(TIMESTAMP_FORMAT) + "Z",
                            'interval_start':      stream['start'],
                            'interval_end':        stream['end'],
                            'throughput_mbps':     stream.get('bits_per_second', 0) / 1e6,
                            'jitter_ms':           stream.get('jitter_ms', 0),
                            'packet_loss_percent': stream.get('lost_percent', 0),
                            'retransmits':         server_stats['retransmits'],
                            'cwnd_bytes':          server_stats['cwnd_bytes'],
                            'rtt_ms':              stream.get('rtt', 0) / 1e3,
                            'direction':           direction,
                            'protocol':            protocol,
                            'congestion_control':  congestion or 'N/A',
                            'port':                port,
                            'target_bandwidth':    bandwidth,
                            'experiment_time':     experiment_timestamp,
                        })

            else:  # UL — cwnd/retr come from client (sender)
                for interval_block in data.get('intervals', []):
                    for stream in interval_block.get('streams', []):
                        interval_start = experiment_time + timedelta(seconds=stream['start'])
                        iperf_data.append({
                            'timestamp':           interval_start.strftime(TIMESTAMP_FORMAT) + "Z",
                            'interval_start':      stream['start'],
                            'interval_end':        stream['end'],
                            'throughput_mbps':     stream.get('bits_per_second', 0) / 1e6,
                            'jitter_ms':           stream.get('jitter_ms', 0),
                            'packet_loss_percent': stream.get('lost_percent', 0),
                            'retransmits':         stream.get('retransmits', 0),
                            'cwnd_bytes':          stream.get('snd_cwnd', 0),
                            'rtt_ms':              stream.get('rtt', 0) / 1e3,
                            'direction':           direction,
                            'protocol':            protocol,
                            'congestion_control':  congestion or 'N/A',
                            'port':                port,
                            'target_bandwidth':    bandwidth,
                            'experiment_time':     experiment_timestamp,
                        })

            # Save server output separately
            server_out = data.get('server_output')
            if server_out:
                server_file = raw_output_file.replace('.json', '_server.json')
                with open(server_file, 'w') as f:
                    json.dump(server_out, f, indent=2)

            # Parsed CSV
            if iperf_data:
                with open(parsed_output_file, 'w', newline='') as f:
                    fieldnames = [
                        'timestamp', 'interval_start', 'interval_end',
                        'throughput_mbps', 'jitter_ms', 'packet_loss_percent',
                        'retransmits', 'cwnd_bytes', 'rtt_ms',
                        'direction', 'protocol', 'congestion_control',
                        'port', 'target_bandwidth', 'experiment_time',
                    ]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(iperf_data)

            return len(iperf_data)
        else:
            print(f"iperf3 {direction} failed: {result.stderr}")
            return 0

    except subprocess.TimeoutExpired:
        print(f"iperf3 {direction} timed out")
        return 0
    except Exception as e:
        print(f"iperf3 {direction} error: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Collect iperf3 statistics")
    parser.add_argument("--direction",   type=str,   required=True, choices=["DL", "UL"],
                        help="Test direction")
    parser.add_argument("--bandwidth",   type=int,   required=True, help="Target bandwidth in Mbps")
    parser.add_argument("--duration",    type=int,   required=True, help="Test duration in seconds")
    parser.add_argument("--server-ip",   type=str,   required=True, help="iperf3 server IP")
    parser.add_argument("--raw-file",    required=True,             help="Output path for raw JSON")
    parser.add_argument("--parsed-file", required=True,             help="Output path for parsed CSV")
    parser.add_argument("--protocol",    type=str,   default="udp", choices=["udp", "tcp"],
                        help="Transport protocol (default: udp)")
    parser.add_argument("--congestion",  type=str,   choices=["cubic", "bbr"],
                        help="TCP congestion control (tcp only)")
    parser.add_argument("--port",        type=int,   default=5201,  help="iperf3 server port (default: 5201)")
    parser.add_argument("--interval",    type=float, default=0.2,   help="Reporting interval in seconds (default: 0.2)")
    args = parser.parse_args()

    if args.congestion and args.protocol != 'tcp':
        parser.error("--congestion requires --protocol tcp")

    proto_label = args.protocol.upper()
    cc_label    = f" [{args.congestion}]" if args.congestion else ""
    print(f"Starting iperf3: direction={args.direction}, protocol={proto_label}{cc_label}, "
          f"bandwidth={args.bandwidth}Mbps, duration={args.duration}s, server={args.server_ip}")

    count = run_iperf3_test(
        args.direction, args.bandwidth, args.duration, args.server_ip,
        args.raw_file, args.parsed_file,
        protocol=args.protocol,
        congestion=args.congestion,
        port=args.port,
        interval=args.interval,
    )

    print(f"iperf3 complete: {count} intervals")
    print(f"  Raw:    {args.raw_file}")
    print(f"  Parsed: {args.parsed_file}")


if __name__ == "__main__":
    main()
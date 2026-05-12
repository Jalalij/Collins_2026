#!/usr/bin/env python3
# ping_collector.py - Collects ping statistics with timestamps

import subprocess
import re
import json
import csv
import argparse
from datetime import datetime, timezone

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"


def collect_ping_data(duration, server_ip, raw_output_file, parsed_output_file,
                      interval=0.2, packet_size=56, source=None):
    """Run ping for duration seconds and collect per-packet RTT with timestamps."""

    start_time           = datetime.now(timezone.utc)
    experiment_timestamp = start_time.strftime(TIMESTAMP_FORMAT) + "Z"

    packet_count = int(duration / interval)

    cmd = [
        'ping', server_ip,
        '-D',
        '-i', str(interval),
        '-s', str(packet_size),
        '-c', str(packet_count),
    ]

    if source:
        cmd += ['-I', source]

    samples = []

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        try:
            stdout, stderr = process.communicate(timeout=duration + 5)

        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()

            print(f"Ping timed out after {duration + 5}s and was killed")

        # ------------------------------------------------------------------ #
        # Save raw JSON ALWAYS
        # ------------------------------------------------------------------ #

        with open(raw_output_file, 'w') as f:
            json.dump({
                "command":      " ".join(cmd),
                "start_time":   start_time.isoformat(),
                "duration":     duration,
                "server_ip":    server_ip,
                "interval":     interval,
                "packet_size":  packet_size,
                "source":       source,
                "output":       stdout,
                "error":        stderr,
                "return_code":  process.returncode,
            }, f, indent=2)

        # ------------------------------------------------------------------ #
        # Parse successful ping replies
        # ------------------------------------------------------------------ #

        pattern = r'\[(\d+\.\d+)\].*icmp_seq=(\d+).*time=(\d+\.?\d*) ms'

        for line in stdout.split('\n'):

            match = re.search(pattern, line)

            if match:
                samples.append({
                    'timestamp': datetime.fromtimestamp(
                        float(match.group(1)),
                        tz=timezone.utc
                    ).strftime(TIMESTAMP_FORMAT) + "Z",

                    'sequence': int(match.group(2)),
                    'rtt_ms': float(match.group(3)),
                    'source': source or '',
                    'experiment_time': experiment_timestamp,
                })

        # ------------------------------------------------------------------ #
        # ALWAYS create CSV, even if empty
        # ------------------------------------------------------------------ #

        with open(parsed_output_file, 'w', newline='') as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    'timestamp',
                    'sequence',
                    'rtt_ms',
                    'source',
                    'experiment_time'
                ]
            )

            writer.writeheader()

            if samples:
                writer.writerows(samples)

        # ------------------------------------------------------------------ #
        # Helpful debug output
        # ------------------------------------------------------------------ #

        transmitted_match = re.search(
            r'(\d+)\s+packets transmitted',
            stdout
        )

        received_match = re.search(
            r'(\d+)\s+received',
            stdout
        )

        if transmitted_match and received_match:
            tx = int(transmitted_match.group(1))
            rx = int(received_match.group(1))

            print(f"Ping stats: transmitted={tx} received={rx} lost={tx-rx}")

        if process.returncode != 0:
            print(f"Ping exited with code {process.returncode}")

        if stderr.strip():
            print(f"Ping stderr: {stderr.strip()}")

        return len(samples)

    except Exception as e:
        print(f"Ping collection error: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Collect ping statistics")

    parser.add_argument(
        "--duration",
        type=int,
        required=True,
        help="Test duration in seconds"
    )

    parser.add_argument(
        "--server-ip",
        type=str,
        required=True,
        help="Target server IP"
    )

    parser.add_argument(
        "--raw-file",
        required=True,
        help="Output path for raw JSON"
    )

    parser.add_argument(
        "--parsed-file",
        required=True,
        help="Output path for parsed CSV"
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="Seconds between pings (default: 0.2)"
    )

    parser.add_argument(
        "--packet-size",
        type=int,
        default=56,
        help="Ping packet size in bytes (default: 56)"
    )

    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Source IP or interface (e.g. 10.0.0.1 or eth0)"
    )

    args = parser.parse_args()

    src_label = f" source={args.source}" if args.source else ""

    print(
        f"Starting ping: "
        f"duration={args.duration}s "
        f"server={args.server_ip}"
        f"{src_label}"
    )

    count = collect_ping_data(
        args.duration,
        args.server_ip,
        args.raw_file,
        args.parsed_file,
        interval=args.interval,
        packet_size=args.packet_size,
        source=args.source,
    )

    print(f"Ping complete: {count} packets")
    print(f"  Raw:    {args.raw_file}")
    print(f"  Parsed: {args.parsed_file}")


if __name__ == "__main__":
    main()

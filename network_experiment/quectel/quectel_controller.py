#!/usr/bin/env python3
# quectel_collector.py - Collects 5G Quectel modem signal stats

import serial
import subprocess
import json
import csv
import time
import argparse
from datetime import datetime, timezone
from network_experiment.quectel.config import TIMESTAMP_FORMAT, generate_quectel_filename

BAUDRATE = 115200


def _find_device():
    """Locate the Quectel ttyUSB device (same logic as quectel.py)."""
    try:
        devs     = subprocess.check_output('ls /dev/ttyUSB*', shell=True).decode()
        dev_list = devs.split('\n')
        return dev_list[2]
    except Exception:
        return None


def _open_serial(dev_file):
    return serial.Serial(
        port     = dev_file,
        baudrate = BAUDRATE,
        bytesize = serial.EIGHTBITS,
        parity   = serial.PARITY_NONE,
        timeout  = 1,
        stopbits = serial.STOPBITS_ONE,
        rtscts   = False,
        dsrdtr   = False,
    )


def _read_sample(signal):
    """
    Send AT command and parse one response.
    Returns a data dict or None if no valid 5G record found.
    Mirrors the parsing logic in quectel.py measure_link().
    """
    command = 'at+qeng="servingcell"\r\n'.encode()
    signal.write(command)

    try:
        res = signal.read(1024)
        if not res:
            return None
    except Exception:
        return None

    try:
        for item in res.decode().split('\n'):
            output = item.strip()
            data   = {}

            if '\"NR5G-SA\",\"TDD\",240,80' in output and 'servingcell' in output:
                components = output.split(',')
                data = {
                    "band":   "mid-band",
                    "cellid": components[6],
                    "arfcn":  components[9],
                    "rsrp":   int(components[12]),
                    "sinr":   int(components[14]),
                    "rsrq":   int(components[13]),
                }
            elif 'NR5G-NRDC' in output:
                components = output.split(',')
                data = {
                    "band":   "mmWave",
                    "cellid": components[3],
                    "arfcn":  components[4],
                    "rsrp":   int(components[7]),
                    "sinr":   int(components[9]),
                    "rsrq":   int(components[8]),
                }

            if data:
                return data

    except (IndexError, ValueError):
        pass

    return None


def collect_quectel_data(duration):
    """
    Read Quectel modem signal stats for duration seconds.
    Produces one sample per AT command response (~1/second).
    """
    raw_output_file    = generate_quectel_filename("json")
    parsed_output_file = generate_quectel_filename("csv")

    dev_file = _find_device()
    if not dev_file:
        print("Quectel device not found")
        return None, None, 0

    start_time           = datetime.now(timezone.utc)
    experiment_timestamp = start_time.strftime(TIMESTAMP_FORMAT) + "Z"

    samples  = []
    deadline = start_time.timestamp() + duration

    try:
        signal = _open_serial(dev_file)
    except Exception as e:
        print(f"Failed to open serial port {dev_file}: {e}")
        return None, None, 0

    try:
        while time.time() < deadline:
            poll_time = datetime.now(timezone.utc)
            ts_str    = poll_time.strftime(TIMESTAMP_FORMAT) + "Z"

            sample = _read_sample(signal)
            if sample:
                sample["timestamp"]       = ts_str
                sample["experiment_time"] = experiment_timestamp
                samples.append(sample)

    finally:
        signal.close()

        # Raw JSON
        with open(raw_output_file, "w") as f:
            json.dump({
                "device":       dev_file,
                "start_time":   start_time.isoformat(),
                "duration":     duration,
                "sample_count": len(samples),
                "samples":      samples,
            }, f, indent=2)

        # Parsed CSV
        fieldnames = [
            "timestamp",
            "band",
            "cellid",
            "arfcn",
            "rsrp",
            "rsrq",
            "sinr",
            "experiment_time",
        ]
        if samples:
            with open(parsed_output_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(samples)

    return raw_output_file, parsed_output_file, len(samples)


def main():
    parser = argparse.ArgumentParser(description="Collect Quectel 5G modem signal stats")
    parser.add_argument("--duration", type=int, required=True, help="Collection duration in seconds")
    args = parser.parse_args()

    print(f"Starting Quectel: duration={args.duration}s")

    raw_file, parsed_file, sample_count = collect_quectel_data(args.duration)

    if raw_file and parsed_file:
        print(f"Quectel complete: {sample_count} samples")
        print(f"  Raw:    {raw_file}")
        print(f"  Parsed: {parsed_file}")
    else:
        print("Quectel collection failed")


if __name__ == "__main__":
    main()
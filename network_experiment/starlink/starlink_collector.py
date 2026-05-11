#!/usr/bin/env python3
# starlink_collector.py - Collects Starlink dish telemetry via grpcurl

import subprocess
import json
import csv
import time
import argparse
from datetime import datetime, timezone

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
STARLINK_INTERVAL = 0.2

DISH_ADDRESS = "192.168.100.1:9200"
DISH_SERVICE = "SpaceX.API.Device.Device/Handle"


# --------------------------------------------------------------------------- #
# grpcurl helpers
# --------------------------------------------------------------------------- #

def _grpc_call(request_key: str) -> dict | None:
    """Issue one grpcurl call and return the parsed JSON, or None on failure."""
    cmd = [
        "grpcurl", "-plaintext",
        "-d", json.dumps({request_key: {}}),
        DISH_ADDRESS,
        DISH_SERVICE,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return json.loads(result.stdout)
        return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return None


def get_status() -> dict | None:
    return _grpc_call("get_status")

def get_diagnostics() -> dict | None:
    return _grpc_call("get_diagnostics")

def get_location() -> dict | None:
    return _grpc_call("get_location")

def get_device_info() -> dict | None:
    return _grpc_call("get_device_info")


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #

def _parse_status(data: dict) -> dict:
    """Extract the fields we care about from a get_status response."""
    s     = data.get("dishGetStatus", {})
    obs   = s.get("obstructionStats", {})
    gps   = s.get("gpsStats", {})
    align = s.get("alignmentStats", {})
    return {
        "dl_throughput_bps":       s.get("downlinkThroughputBps"),
        "ul_throughput_bps":       s.get("uplinkThroughputBps"),
        "pop_ping_latency_ms":     s.get("popPingLatencyMs"),
        "obstruction_fraction": obs.get("fractionObstructed", 0),
        "time_obstructed":      obs.get("timeObstructed", 0),
        "avg_prolonged_obstruction_duration_s": obs.get("avgProlongedObstructionDurationS"),
        "avg_prolonged_obstruction_interval_s": obs.get("avgProlongedObstructionIntervalS"),
        "valid_s":                              obs.get("validS"),
        "snr_above_noise_floor":   s.get("isSnrAboveNoiseFloor"),
        "gps_valid":               gps.get("gpsValid"),
        "gps_sats":                gps.get("gpsSats"),
        "boresight_azimuth_deg":   align.get("boresightAzimuthDeg"),
        "boresight_elevation_deg": align.get("boresightElevationDeg"),
        "attitude_uncertainty_deg":             align.get("attitudeUncertaintyDeg"),
        "desired_boresight_azimuth_deg":        align.get("desiredBoresightAzimuthDeg"),
        "desired_boresight_elevation_deg":      align.get("desiredBoresightElevationDeg"),
        "tilt_angle_deg":          align.get("tiltAngleDeg"),
        "uptime_s":                s.get("deviceState", {}).get("uptimeS"),
        "software_version":        s.get("deviceInfo", {}).get("softwareVersion"),
        "disablement_code":        s.get("disablementCode"),
        "dl_bw_restricted_reason": s.get("dlBandwidthRestrictedReason"),
        "ul_bw_restricted_reason": s.get("ulBandwidthRestrictedReason"),
    }


def _parse_location(data: dict) -> dict:
    """Extract location fields from a get_location response."""
    loc = data.get("getLocation", {})
    lla = loc.get("lla", {})
    return {
        "latitude":   lla.get("lat"),
        "longitude":  lla.get("lon"),
        "altitude_m": lla.get("alt"),
        "sigma_m":    loc.get("sigmaM"),
        "loc_source": loc.get("source"),
    }


# --------------------------------------------------------------------------- #
# Main collection loop
# --------------------------------------------------------------------------- #

def collect_starlink_data(duration: int, raw_output_file: str, parsed_output_file: str):
    """Poll the Starlink dish every STARLINK_INTERVAL seconds for duration seconds."""

    start_time           = datetime.now(timezone.utc)
    experiment_timestamp = start_time.strftime(TIMESTAMP_FORMAT) + "Z"

    device_info_raw = get_device_info()
    device_info     = {}
    if device_info_raw:
        di          = device_info_raw.get("getDeviceInfo", {}).get("deviceInfo", {})
        device_info = {
            "dish_id":          di.get("id"),
            "hardware_version": di.get("hardwareVersion"),
            "software_version": di.get("softwareVersion"),
            "country_code":     di.get("countryCode"),
            "bootcount":        di.get("bootcount"),
        }

    samples  = []
    deadline = start_time.timestamp() + duration

    try:
        while time.time() < deadline:
            poll_time = datetime.now(timezone.utc)
            ts_str    = poll_time.strftime(TIMESTAMP_FORMAT) + "Z"

            status_raw   = get_status()
            location_raw = get_location()

            row = {"timestamp": ts_str, "experiment_time": experiment_timestamp}

            if status_raw:
                row.update(_parse_status(status_raw))
            if location_raw:
                row.update(_parse_location(location_raw))

            for bps_key, mbps_key in [
                ("dl_throughput_bps", "dl_throughput_mbps"),
                ("ul_throughput_bps", "ul_throughput_mbps"),
            ]:
                if bps_key in row and row[bps_key] is not None:
                    row[mbps_key] = row.pop(bps_key) / 1e6
                else:
                    row.pop(bps_key, None)
                    row[mbps_key] = None

            samples.append(row)

            elapsed = time.time() - poll_time.timestamp()
            sleep_time = STARLINK_INTERVAL - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Raw JSON
        with open(raw_output_file, "w") as f:
            json.dump({
                "start_time":    start_time.isoformat(),
                "duration":      duration,
                "poll_interval": STARLINK_INTERVAL,
                "device_info":   device_info,
                "samples":       samples,
            }, f, indent=2)

        # Parsed CSV
        fieldnames = [
            "timestamp",
            "dl_throughput_mbps",
            "ul_throughput_mbps",
            "pop_ping_latency_ms",
            "obstruction_fraction",
            "time_obstructed",
            "valid_s",
            "avg_prolonged_obstruction_duration_s",
            "avg_prolonged_obstruction_interval_s",
            "snr_above_noise_floor",
            "gps_valid",
            "gps_sats",
            "boresight_azimuth_deg",
            "boresight_elevation_deg",
            "tilt_angle_deg",
            "attitude_uncertainty_deg",
            "desired_boresight_azimuth_deg",
            "desired_boresight_elevation_deg",
            "latitude",
            "longitude",
            "altitude_m",
            "sigma_m",
            "loc_source",
            "uptime_s",
            "disablement_code",
            "dl_bw_restricted_reason",
            "ul_bw_restricted_reason",
            "experiment_time",
        ]
        if samples:
            with open(parsed_output_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(samples)

        return len(samples)

    except Exception as e:
        print(f"Starlink collection error: {e}")
        return raw_output_file, None, 0


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Starlink dish telemetry")
    parser.add_argument("--duration",    type=int, required=True, help="Collection duration in seconds")
    parser.add_argument("--raw-file",    required=True,           help="Output path for raw JSON")
    parser.add_argument("--parsed-file", required=True,           help="Output path for parsed CSV")
    args = parser.parse_args()

    print(f"Starting Starlink: duration={args.duration}s poll_interval={STARLINK_INTERVAL}s")

    count = collect_starlink_data(args.duration, args.raw_file, args.parsed_file)

    print(f"Starlink complete: {count} samples")
    print(f"  Raw:    {args.raw_file}")
    print(f"  Parsed: {args.parsed_file}")

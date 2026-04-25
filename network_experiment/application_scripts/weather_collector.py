#!/usr/bin/env python3
# weather_collector.py - Collects weather station data via araweather
import json
import csv
import time
import argparse
from datetime import datetime, timezone
from araweather import weather as araweather

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
WEATHER_STATIONS = ["WilsonHall"]
SCALAR_FIELDS = [
    "Temperature", "Humidity", "Pressure",
    "WindSpeed", "WindDirection",
    "RainRate", "RainIntensity", "RainAccumulated", "RainAbsolute",
    "SnowDepthIntensity", "MORVisibility", "RadarReflectivity",
    "KineticEnergy", "ParticlesDetected",
    "WeatherCode4677", "WeatherCode4678", "WeatherCode4680", "WeatherCodeNWS",
]

def collect_weather_data(duration, raw_output_file, parsed_output_file, poll_interval=1.0):
    """Poll weather stations every poll_interval seconds for duration seconds."""
    start_time           = datetime.now(timezone.utc)
    experiment_timestamp = start_time.strftime(TIMESTAMP_FORMAT) + "Z"

    samples     = []
    raw_samples = []
    deadline    = start_time.timestamp() + duration

    try:
        while time.time() < deadline:
            poll_time = datetime.now(timezone.utc)
            ts_str    = poll_time.strftime(TIMESTAMP_FORMAT) + "Z"

            try:
                data = araweather.get_current_weather(WEATHER_STATIONS)
            except Exception as e:
                print(f"Weather poll error: {e}")
                time.sleep(poll_interval)
                continue

            for station in WEATHER_STATIONS:
                station_data = data.get(station, {})
                raw_samples.append({
                    "timestamp":       ts_str,
                    "station":         station,
                    "experiment_time": experiment_timestamp,
                    "raw":             station_data,
                })
                row = {"timestamp": ts_str, "station": station, "experiment_time": experiment_timestamp}
                for field in SCALAR_FIELDS:
                    row[field] = station_data.get(field)
                samples.append(row)

            time.sleep(poll_interval)

        # Raw JSON
        with open(raw_output_file, "w") as f:
            json.dump({
                "start_time":    start_time.isoformat(),
                "duration":      duration,
                "poll_interval": poll_interval,
                "stations":      WEATHER_STATIONS,
                "samples":       raw_samples,
            }, f, indent=2)

        # Parsed CSV
        fieldnames = ["timestamp", "station"] + SCALAR_FIELDS + ["experiment_time"]
        if samples:
            with open(parsed_output_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(samples)

        return len(samples)

    except Exception as e:
        print(f"Weather collection error: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Collect weather station data")
    parser.add_argument("--duration",     type=int,   required=True, help="Collection duration in seconds")
    parser.add_argument("--raw-file",     required=True,             help="Output path for raw JSON")
    parser.add_argument("--parsed-file",  required=True,             help="Output path for parsed CSV")
    parser.add_argument("--poll-interval", type=float, default=1.0,  help="Seconds between polls (default: 1.0)")
    args = parser.parse_args()

    print(f"Starting weather: duration={args.duration}s stations={WEATHER_STATIONS}")

    count = collect_weather_data(
        args.duration, args.raw_file, args.parsed_file,
        poll_interval=args.poll_interval,
    )

    print(f"Weather complete: {count} samples")
    print(f"  Raw:    {args.raw_file}")
    print(f"  Parsed: {args.parsed_file}")


if __name__ == "__main__":
    main()
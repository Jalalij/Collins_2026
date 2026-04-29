#!/usr/bin/env python3
# weather_collector.py - Collects weather station data via araweather

import json
import csv
import time
import argparse
from datetime import datetime, timezone
from araweather import weather as araweather

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"

# Per-station field schemas — each station gets its own CSV
STATION_FIELDS = {
    "WilsonHall": [
        "Temperature", "Humidity", "Pressure",
        "WindSpeed", "WindDirection",
        "RainRate", "RainIntensity", "RainAccumulated", "RainAbsolute",
        "SnowDepthIntensity", "MORVisibility", "RadarReflectivity",
        "KineticEnergy", "ParticlesDetected",
        "WeatherCode4677", "WeatherCode4678", "WeatherCode4680", "WeatherCodeNWS",
    ],
    "AgronomyFarm": [
        "Temperature", "Humidity", "Pressure",
        "WindSpeed", "WindDirection",
        "RainRate", "PrecipitationIntensityHr", "PrecipitationIntensityMin",
        "PrecipitationCumulative", "PrecipitationDifferential",
        "PrecipitationType", "WeatherType",
        "TotalDrops", "DrizzleParticles", "SnowParticles", "HailParticles",
        "Drops<0.5mm", "Drops0.5-1.0mm", "Drops1.0-1.5mm", "Drops1.5-2.0mm",
        "Drops2.0-2.5mm", "Drops2.5-3.0mm", "Drops3.0-3.5mm", "Drops3.5-4.0mm",
        "Drops4.0-4.5mm", "Drops4.5-5.0mm", "Drops5.0-5.5mm", "Drops>5.5mm",
    ],
}

DEFAULT_STATIONS = ["WilsonHall"]


def collect_weather_data(duration, raw_output_file, parsed_output_file,
                         poll_interval=1.0, stations=None):
    """Poll weather stations every poll_interval seconds for duration seconds.

    Each station gets its own parsed CSV file named:
        <parsed_output_file_base>_<StationName>.csv
    A single raw JSON file contains all stations' raw data combined.
    """
    if stations is None:
        stations = DEFAULT_STATIONS

    start_time           = datetime.now(timezone.utc)
    experiment_timestamp = start_time.strftime(TIMESTAMP_FORMAT) + "Z"

    # Per-station sample lists
    samples     = {s: [] for s in stations}
    raw_samples = []
    deadline    = start_time.timestamp() + duration

    try:
        while time.time() < deadline:
            poll_time = datetime.now(timezone.utc)
            ts_str    = poll_time.strftime(TIMESTAMP_FORMAT) + "Z"

            try:
                data = araweather.get_current_weather(stations)
            except Exception as e:
                print(f"Weather poll error: {e}")
                time.sleep(poll_interval)
                continue

            for station in stations:
                station_data = data.get(station, {})

                raw_samples.append({
                    "timestamp":       ts_str,
                    "station":         station,
                    "experiment_time": experiment_timestamp,
                    "raw":             station_data,
                })

                fields = STATION_FIELDS.get(station, list(station_data.keys()))
                row    = {"timestamp": ts_str, "station": station, "experiment_time": experiment_timestamp}
                for field in fields:
                    row[field] = station_data.get(field)
                samples[station].append(row)

            time.sleep(poll_interval)

        # Raw JSON — all stations combined
        with open(raw_output_file, "w") as f:
            json.dump({
                "start_time":    start_time.isoformat(),
                "duration":      duration,
                "poll_interval": poll_interval,
                "stations":      stations,
                "samples":       raw_samples,
            }, f, indent=2)

        # Parsed CSV — one file per station
        base = parsed_output_file.replace(".csv", "")
        total = 0
        for station in stations:
            if not samples[station]:
                continue
            fields     = STATION_FIELDS.get(station, list(samples[station][0].keys()))
            fieldnames = ["timestamp", "station"] + fields + ["experiment_time"]
            out_file   = f"{base}_{station}.csv"
            with open(out_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(samples[station])
            total += len(samples[station])
            print(f"  {station}: {len(samples[station])} samples → {out_file}")

        return total

    except Exception as e:
        print(f"Weather collection error: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Collect weather station data")
    parser.add_argument("--duration",      type=int,   required=True,
                        help="Collection duration in seconds")
    parser.add_argument("--raw-file",      required=True,
                        help="Output path for raw JSON")
    parser.add_argument("--parsed-file",   required=True,
                        help="Base output path for parsed CSVs (one per station)")
    parser.add_argument("--poll-interval", type=float, default=1.0,
                        help="Seconds between polls (default: 1.0)")
    parser.add_argument("--stations",      nargs="+", default=DEFAULT_STATIONS,
                        help=f"Station names to collect (default: {DEFAULT_STATIONS})")
    args = parser.parse_args()

    unknown = [s for s in args.stations if s not in STATION_FIELDS]
    if unknown:
        print(f"Warning: unknown stations {unknown} — will use dynamic field detection")

    print(f"Starting weather: duration={args.duration}s stations={args.stations}")

    count = collect_weather_data(
        args.duration, args.raw_file, args.parsed_file,
        poll_interval=args.poll_interval,
        stations=args.stations,
    )

    print(f"Weather complete: {count} total samples")
    print(f"  Raw:    {args.raw_file}")


if __name__ == "__main__":
    main()
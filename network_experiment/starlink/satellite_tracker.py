"""
satellite_tracker.py
--------------------
Stripped-down LEOViz for Starlink satellite tracker estimation.

Usage:
    python3 satellite_tracker.py [--tle-file path] [--duration N] [--grpc addr:port]

Set LATITUDE, LONGITUDE, ALTITUDE in config.py before running.

Dependencies:
    pip install skyfield numpy pandas
    git clone https://github.com/sparky8512/starlink-grpc-tools  (next to this file)
"""

import sys
import csv
import json
import time
import logging
import threading
import subprocess
import urllib.request

from io import StringIO
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Any, Optional, Dict

import argparse
import numpy as np
import pandas as pd
from skyfield.api import load, wgs84

import config

sys.path.insert(0, str(Path("./starlink-grpc-tools").resolve()))
import starlink_grpc

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GRPC_TIMEOUT     = 10
TLE_URL          = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
TLE_DIR          = Path("TLE")


# ══════════════════════════════════════════════════════════════════════════════
# TimeslotManager  — original, unchanged
# ══════════════════════════════════════════════════════════════════════════════

class TimeslotManager:
    TIMESLOT_DURATION  = 14
    TIMESLOT_INTERVALS = [(12, 27), (27, 42), (42, 57), (57, 12)]

    @staticmethod
    def wait_until_target_time(last_timeslot_second: int) -> int:
        now = datetime.now(timezone.utc)
        if last_timeslot_second == 12:
            next_timeslot_second = 27
        elif last_timeslot_second == 27:
            next_timeslot_second = 42
        elif last_timeslot_second == 42:
            next_timeslot_second = 57
        elif last_timeslot_second == 57:
            next_timeslot_second = 12
            if now.second >= 57:
                now = now + timedelta(minutes=1)

        target_time = now.replace(microsecond=0).replace(second=next_timeslot_second)
        while datetime.now(timezone.utc) < target_time:
            time.sleep(0.1)

        return next_timeslot_second


# ══════════════════════════════════════════════════════════════════════════════
# LocationProvider  — original, unchanged
# ══════════════════════════════════════════════════════════════════════════════

class LocationProvider:
    def get_observer_location(self, df_gps_diagnostics: Optional[pd.DataFrame] = None) -> Optional[Any]:
        try:
            if config.MOBILE:
                if df_gps_diagnostics is None or df_gps_diagnostics.empty:
                    logger.error("GPS diagnostics data is required for mobile installations")
                    return None
                latest_gps = df_gps_diagnostics.iloc[-1]
                latitude   = latest_gps["lat"]
                longitude  = latest_gps["lon"]
                altitude   = latest_gps["alt"]
            else:
                latitude  = config.LATITUDE
                longitude = config.LONGITUDE
                altitude  = config.ALTITUDE
            return wgs84.latlon(latitude, longitude, altitude)
        except Exception as e:
            logger.error(f"Error getting observer location: {str(e)}", exc_info=True)
            return None


# ══════════════════════════════════════════════════════════════════════════════
# ObstructionMap  — original process_timeslot logic, CSV writer removed
#                   (we return the rows as a DataFrame instead of writing CSV)
# ══════════════════════════════════════════════════════════════════════════════

class ObstructionMap:
    def __init__(self):
        self.map_size = 123

    def process_timeslot(self, timeslot_df: pd.DataFrame, writer: csv.writer) -> None:
        """Original logic: XOR consecutive maps, write (timestamp, Y, X) to CSV writer."""
        previous_obstruction_map = timeslot_df.iloc[0]["obstruction_map"]
        previous_obstruction_map = previous_obstruction_map.reshape(self.map_size, self.map_size)

        hold_coord = None
        white_pixel_coords = []
        for _, row in timeslot_df.iterrows():
            timestamp_dt    = pd.to_datetime(row["timestamp"], unit="s")
            obstruction_map = row["obstruction_map"].reshape(self.map_size, self.map_size)
            xor_map         = np.bitwise_xor(previous_obstruction_map, obstruction_map)
            coords          = np.argwhere(xor_map == 1)

            if coords.size > 0:
                coord      = coords[-1]
                hold_coord = coord
            elif hold_coord is not None:
                coord = hold_coord
            else:
                continue

            white_pixel_coords.append((timestamp_dt, tuple(coord)))
            previous_obstruction_map = obstruction_map

        for coord in white_pixel_coords:
            writer.writerow([
                coord[0].strftime("%Y-%m-%d %H:%M:%S"),
                coord[1][0],   # Y
                coord[1][1],   # X
            ])


# ══════════════════════════════════════════════════════════════════════════════
# DataFeatureExtraction  — original, unchanged
# ══════════════════════════════════════════════════════════════════════════════

class DataFeatureExtraction:
    def __init__(self):
        self.status_columns = [
            "timestamp", "hardwareVersion", "sinr", "popPingLatencyMs",
            "downlinkThroughputBps", "uplinkThroughputBps",
            "tiltAngleDeg", "boresightAzimuthDeg", "boresightElevationDeg",
            "attitudeEstimationState", "attitudeUncertaintyDeg",
            "desiredBoresightAzimuthDeg", "desiredBoresightElevationDeg",
            "quaternion_qScalar", "quaternion_qX", "quaternion_qY", "quaternion_qZ",
        ]
        self.location_columns = [
            "gps_time", "timestamp", "lat", "lon", "alt",
            "uncertainty_valid", "uncertainty",
        ]
        self.obstruction_map = ObstructionMap()

    def extract_status_fields(self, status: Dict[str, Any], current_time: Optional[float] = None) -> List[Any]:
        alignment  = status.get("alignmentStats", {})
        quaternion = status.get("ned2dishQuaternion", {})
        deviceInfo = status.get("deviceInfo", {})
        return [
            current_time if current_time is not None else time.time(),
            deviceInfo.get("hardwareVersion", ""),
            status.get("phyRxBeamSnrAvg", 0),
            status.get("popPingLatencyMs", 0),
            status.get("downlinkThroughputBps", 0),
            status.get("uplinkThroughputBps", 0),
            alignment.get("tiltAngleDeg", 0),
            alignment.get("boresightAzimuthDeg", 0),
            alignment.get("boresightElevationDeg", 0),
            alignment.get("attitudeEstimationState", ""),
            alignment.get("attitudeUncertaintyDeg", 0),
            alignment.get("desiredBoresightAzimuthDeg", 0),
            alignment.get("desiredBoresightElevationDeg", 0),
            quaternion.get("qScalar", 0),
            quaternion.get("qX", 0),
            quaternion.get("qY", 0),
            quaternion.get("qZ", 0),
        ]

    def extract_location_fields(self, diagnostics: Dict[str, Any], current_time: Optional[float] = None) -> List[Any]:
        location = diagnostics.get("dishGetDiagnostics", {}).get("location", {})
        gps_time = location.get("gpsTimeS", 0)
        return [
            gps_time,
            current_time if current_time is not None else time.time(),
            location.get("latitude", 0),
            location.get("longitude", 0),
            location.get("altitudeMeters", 0),
            location.get("uncertaintyMetersValid", False),
            location.get("uncertaintyMeters", 0),
        ]

    def pre_process_observed_data_by_frame_type(self, merged_df: pd.DataFrame, frame_type: int) -> pd.DataFrame:
        """Original pixel → elevation/azimuth conversion."""
        pixel_to_degrees = 80 / 62
        elevations = []
        azimuths   = []

        for _, row in merged_df.iterrows():
            if frame_type == 1:  # FRAME_EARTH
                observer_x, observer_y = 62, 62
                dx = row["X"] - observer_x
                dy = (123 - row["Y"]) - observer_y
            elif frame_type == 2:  # FRAME_UT
                _tilt        = row["tiltAngleDeg"]
                _rotation_az = row["boresightAzimuthDeg"]
                observer_x   = 62
                observer_y   = 62 - (_tilt / (80 / 62))
                dx = row["X"] - observer_x
                dy = row["Y"] - observer_y
            else:
                elevations.append(np.nan)
                azimuths.append(np.nan)
                continue

            radius  = np.sqrt(dx**2 + dy**2) * pixel_to_degrees
            azimuth = np.degrees(np.arctan2(dx, dy))

            if frame_type == 1:
                azimuth = (azimuth + 360) % 360
            elif frame_type == 2:
                azimuth = (azimuth + _rotation_az + 360) % 360

            elevation = 90 - radius
            elevations.append(elevation)
            azimuths.append(azimuth)

        merged_df["Elevation"] = elevations
        merged_df["Azimuth"]   = azimuths
        return merged_df

    def merge_obstruction_with_status_and_location(
        self,
        obstruction_csv_str: str,
        frame_type: int,
        df_status: pd.DataFrame,
        df_location: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Original merge logic — identical to the original except obstruction data
        is passed as a CSV string (in-memory) instead of a file path, to avoid
        disk I/O while keeping every line of logic unchanged.
        """
        try:
            df_obstruction = pd.read_csv(StringIO(obstruction_csv_str))
            df_obstruction["timestamp"] = pd.to_datetime(df_obstruction["timestamp"], format="%Y-%m-%d %H:%M:%S")
            df_obstruction = df_obstruction.set_index("timestamp").resample("1s").min().reset_index()

            df_status["timestamp"] = pd.to_datetime(df_status["timestamp"], unit="s")
            df_status = df_status.drop(columns=["attitudeEstimationState"])
            df_status = df_status.set_index("timestamp").resample("1s").min().reset_index()

            alignment_cols = ["tiltAngleDeg", "boresightAzimuthDeg", "boresightElevationDeg"]
            df_status[alignment_cols] = (df_status[alignment_cols].ffill() + df_status[alignment_cols].bfill()) / 2

            merged_df = pd.merge(df_obstruction, df_status, on="timestamp", how="inner")

            if config.MOBILE and df_location is not None:
                df_location["timestamp"] = pd.to_datetime(df_location["timestamp"], unit="s")
                df_location["timestamp"] = df_location["timestamp"].dt.tz_localize(None)
                df_location = df_location.set_index("timestamp").resample("1s").min().reset_index()
                numeric_cols = df_location.select_dtypes(include=[np.number]).columns
                df_location[numeric_cols] = (df_location[numeric_cols].ffill() + df_location[numeric_cols].bfill()) / 2
                merged_df = pd.merge(merged_df, df_location, on="timestamp", how="inner")
            else:
                merged_df["lat"] = config.LATITUDE
                merged_df["lon"] = config.LONGITUDE
                merged_df["alt"] = config.ALTITUDE

            merged_df = merged_df[(merged_df["X"].notna()) & (merged_df["Y"].notna())].reset_index(drop=True)
            merged_df["timestamp"] = merged_df["timestamp"].dt.tz_localize("UTC")
            merged_df["X"] = merged_df["X"].astype(int)
            merged_df["Y"] = merged_df["Y"].astype(int)
            return merged_df

        except Exception as e:
            logger.error(f"Error merging data: {str(e)}", exc_info=True)
            return pd.DataFrame()

    def process_observed_data_from_df(
        self,
        merged_df: pd.DataFrame,
        timestamp: str,
    ) -> Optional[List[Tuple[datetime, Tuple[float, float]]]]:
        """
        Original process_observed_data logic — reads from the merged DataFrame
        instead of a saved CSV file. Everything else is identical.
        """
        try:
            df = merged_df.copy()
            if df["timestamp"].dt.tz is not None:
                df["timestamp"] = df["timestamp"].dt.tz_localize(None)

            timestamp_dt = pd.to_datetime(timestamp).tz_localize(None)
            timeslot_df  = df[
                (df["timestamp"] >= timestamp_dt) &
                (df["timestamp"] <  timestamp_dt + pd.Timedelta(seconds=15))
            ]

            if timeslot_df.empty:
                logger.error(f"No data found for timestamp {timestamp}")
                return None

            observed_positions = []
            for _, row in timeslot_df.iterrows():
                ts        = pd.to_datetime(row["timestamp"])
                elevation = 90 - row["Elevation"]
                azimuth   = row["Azimuth"] % 360
                observed_positions.append((ts, (elevation, azimuth)))

            return observed_positions

        except Exception as e:
            logger.error(f"Error processing observed data: {str(e)}", exc_info=True)
            return None


# ══════════════════════════════════════════════════════════════════════════════
# GrpcCommand  — original, unchanged
# ══════════════════════════════════════════════════════════════════════════════

class GrpcCommand:
    def __init__(self):
        self.status_cmd = [
            "grpcurl", "-plaintext", "-d", '{"get_status":{}}',
            config.STARLINK_GRPC_ADDR_PORT, "SpaceX.API.Device.Device/Handle",
        ]
        self.diagnostics_cmd = [
            "grpcurl", "-plaintext", "-d", '{"get_diagnostics":{}}',
            config.STARLINK_GRPC_ADDR_PORT, "SpaceX.API.Device.Device/Handle",
        ]
        self.reset_obstruction_cmd = [
            "grpcurl", "-plaintext", "-d", '{"dish_clear_obstruction_map":{}}',
            config.STARLINK_GRPC_ADDR_PORT, "SpaceX.API.Device.Device/Handle",
        ]
        self.data_extracter = DataFeatureExtraction()

    def execute(self, cmd: List[str]) -> Optional[Dict[str, Any]]:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=GRPC_TIMEOUT)
            if result.returncode != 0:
                logger.error(f"Command failed with error: {result.stderr}")
                return None
            return json.loads(result.stdout)
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {GRPC_TIMEOUT} seconds")
            return None
        except json.JSONDecodeError:
            logger.error("Failed to parse command output as JSON")
            return None
        except Exception as e:
            logger.error(f"Error executing command: {str(e)}")
            return None

    def reset_obstruction_map(self) -> None:
        try:
            result = self.execute(self.reset_obstruction_cmd)
            if result is None:
                raise Exception("Failed to reset obstruction map")
            # logger.info("Resetting dish obstruction map")
        except Exception as e:
            logger.error(f"Failed resetting obstruction map: {str(e)}")
            raise

    def status(self, current_time: float) -> Optional[List[Any]]:
        try:
            status_data = self.execute(self.status_cmd)
            if not status_data:
                return None
            dish_status = status_data.get("dishGetStatus")
            if not dish_status or "alignmentStats" not in dish_status:
                logger.warning("Missing dishGetStatus or alignmentStats in status data")
                return None
            return self.data_extracter.extract_status_fields(dish_status, current_time)
        except Exception as e:
            logger.error(f"Error getting status data: {str(e)}")
            return None

    def gps_diagnostics(self, current_time: float) -> Optional[List[Any]]:
        if not config.MOBILE:
            return None
        try:
            diagnostics_data = self.execute(self.diagnostics_cmd)
            if not diagnostics_data:
                return None
            return self.data_extracter.extract_location_fields(diagnostics_data, current_time)
        except Exception as e:
            logger.error(f"Error getting GPS diagnostics data: {str(e)}")
            return None

    def get_obstruction_map_frame_type(self) -> Tuple[int, str]:
        try:
            context    = starlink_grpc.ChannelContext(target=config.STARLINK_GRPC_ADDR_PORT)
            map        = starlink_grpc.get_obstruction_map(context)
            frame_type = {0: "UNKNOWN", 1: "FRAME_EARTH", 2: "FRAME_UT"}.get(map.map_reference_frame, "UNKNOWN")
            return map.map_reference_frame, frame_type
        except Exception as e:
            logger.error(f"Error getting obstruction map frame type: {str(e)}")
            return 0, "UNKNOWN"

    def get_obstruction_data(self) -> Optional[Tuple[float, np.ndarray]]:
        """Original: creates a new context per call."""
        try:
            context          = starlink_grpc.ChannelContext(target=config.STARLINK_GRPC_ADDR_PORT)
            obstruction_data = np.array(starlink_grpc.obstruction_map(context), dtype=int)
            obstruction_data[obstruction_data == -1] = 0
            obstruction_data = obstruction_data.flatten()
            return time.time(), obstruction_data
        except Exception as e:
            logger.error(f"Error getting obstruction data: {str(e)}")
            return None

    def collect_timeslot_data(self, timeslot_start: float) -> Optional[Dict[str, List[Any]]]:
        """Original _collect_timeslot_data from jobs.py."""
        try:
            obstruction_data_array = []
            timestamp_array        = []

            while time.time() < timeslot_start + TimeslotManager.TIMESLOT_DURATION:
                result = self.get_obstruction_data()
                if result:
                    timestamp, obstruction_data = result
                    timestamp_array.append(timestamp)
                    obstruction_data_array.append(obstruction_data)
                time.sleep(0.5)

            if not timestamp_array:
                return None

            return {
                "timestamp":      timestamp_array,
                "obstruction_map": obstruction_data_array,
            }
        except Exception as e:
            logger.error(f"Error collecting timeslot data: {str(e)}")
            return None


# ══════════════════════════════════════════════════════════════════════════════
# SatelliteProcessor  — original, unchanged
# ══════════════════════════════════════════════════════════════════════════════

class SatelliteProcessor:
    def __init__(self):
        self.data_extracter    = DataFeatureExtraction()
        self.location_provider = LocationProvider()

    def find_matching_satellites(
        self,
        satellites:                          List[Any],
        observer_location:                   Any,
        observed_positions_with_timestamps:  List[Tuple[datetime, Tuple[float, float]]],
        frame_type:                          int,
    ) -> List[str]:
        best_match               = None
        closest_total_difference = float("inf")
        ts                       = load.timescale()

        for satellite in satellites:
            satellite_positions = []
            valid_positions     = True

            for observed_time, observed_data in observed_positions_with_timestamps:
                difference  = satellite - observer_location
                topocentric = difference.at(
                    ts.utc(
                        observed_time.year, observed_time.month, observed_time.day,
                        observed_time.hour, observed_time.minute, observed_time.second,
                    )
                )
                alt, az, _ = topocentric.altaz()

                if alt.degrees <= 20:
                    valid_positions = False
                    break

                satellite_positions.append((alt.degrees, az.degrees))

            if valid_positions:
                if frame_type == 1:
                    total_difference = self.calculate_total_difference(
                        [(90 - data[0], data[1]) for _, data in observed_positions_with_timestamps],
                        satellite_positions,
                    )
                elif frame_type == 2:
                    total_difference = self.calculate_trajectory_distance_frame_ut(
                        [(90 - data[0], data[1]) for _, data in observed_positions_with_timestamps],
                        satellite_positions,
                    )
                else:
                    continue

                if total_difference < closest_total_difference:
                    closest_total_difference = total_difference
                    best_match               = satellite.name

        return [best_match] if best_match else []

    def calculate_distance_for_best_match(
        self, satellite: Any, observer_location: Any, start_time: datetime, interval_seconds: int
    ) -> List[float]:
        distances = []
        ts        = load.timescale()
        for second in range(interval_seconds + 1):
            current_time = start_time + timedelta(seconds=second)
            difference   = satellite - observer_location
            topocentric  = difference.at(ts.from_datetime(current_time))
            distances.append(topocentric.distance().km)
        return distances

    def process_feature_time_interval(
        self,
        merged_df:        pd.DataFrame,
        year:             int,
        month:            int,
        day:              int,
        hour:             int,
        minute:           int,
        second:           int,
        satellites:       List[Any],
        frame_type:       int,
        df_gps_diagnostics: Optional[pd.DataFrame] = None,
    ) -> Tuple[Optional[List], Optional[List[str]], Optional[List[float]]]:
        """Original process_feature_time_interval — reads from merged_df instead of CSV files."""
        initial_time      = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
        observer_location = self.location_provider.get_observer_location(df_gps_diagnostics)
        if observer_location is None:
            logger.error("Failed to get observer location")
            return None, None, None

        observed_positions_with_timestamps = self.data_extracter.process_observed_data_from_df(
            merged_df, initial_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        if observed_positions_with_timestamps is None:
            return None, None, None

        matching_satellites = self.find_matching_satellites(
            satellites, observer_location, observed_positions_with_timestamps, frame_type
        )
        if not matching_satellites:
            return observed_positions_with_timestamps, None, None

        best_match_satellite = next(sat for sat in satellites if sat.name == matching_satellites[0])
        distances = self.calculate_distance_for_best_match(
            best_match_satellite, observer_location, initial_time, 14
        )

        return observed_positions_with_timestamps, matching_satellites, distances

    def process_intervals(
        self,
        merged_df:        pd.DataFrame,
        start_time:       datetime,
        end_time:         datetime,
        satellites:       List[Any],
        frame_type:       int,
        df_gps_diagnostics: Optional[pd.DataFrame] = None,
    ) -> Optional[pd.DataFrame]:
        """Original process_intervals — operates on in-memory merged_df."""
        try:
            results      = []
            current_time = start_time

            while current_time <= end_time:
                # logger.info(f"Estimating connected satellites for timeslot {current_time}")

                _, matching_satellites, distances = self.process_feature_time_interval(
                    merged_df,
                    current_time.year, current_time.month, current_time.day,
                    current_time.hour, current_time.minute, current_time.second,
                    satellites, frame_type, df_gps_diagnostics,
                )

                if matching_satellites and distances:
                    for second in range(15):
                        if second < len(distances):
                            results.append({
                                "Timestamp":           current_time + timedelta(seconds=second),
                                "Connected_Satellite": matching_satellites[0],
                                "Distance":            distances[second],
                            })

                current_time += timedelta(seconds=15)

            if not results:
                logger.warning("No data points processed")
                return None

            return pd.DataFrame(results)

        except Exception as e:
            logger.error(f"Error processing intervals: {str(e)}", exc_info=True)
            return None

    # ── scoring helpers — original, unchanged ─────────────────────────────────

    def calculate_total_difference(self, observed_positions, satellite_positions):
        total_angular_separation = 0
        for i in range(len(observed_positions)):
            obs_alt, obs_az = observed_positions[i]
            sat_alt, sat_az = satellite_positions[i]
            total_angular_separation += self.angular_separation(obs_alt, obs_az, sat_alt, sat_az)
        bearing_diff = self.calculate_bearing_difference(observed_positions, satellite_positions)
        return total_angular_separation + bearing_diff

    def angular_separation(self, alt1, az1, alt2, az2):
        alt1, alt2 = np.radians(alt1), np.radians(alt2)
        az1 = (az1 + 360) % 360
        az2 = (az2 + 360) % 360
        az_diff = np.abs(az1 - az2)
        if az_diff > 180:
            az_diff = 360 - az_diff
        az_diff    = np.radians(az_diff)
        separation = np.arccos(np.clip(
            np.sin(alt1) * np.sin(alt2) + np.cos(alt1) * np.cos(alt2) * np.cos(az_diff), -1, 1
        ))
        return np.degrees(separation)

    def calculate_bearing(self, alt1, az1, alt2, az2):
        alt1, alt2 = np.radians(alt1), np.radians(alt2)
        az1,  az2  = np.radians(az1),  np.radians(az2)
        x       = np.sin(az2 - az1) * np.cos(alt2)
        y       = np.cos(alt1) * np.sin(alt2) - np.sin(alt1) * np.cos(alt2) * np.cos(az2 - az1)
        bearing = np.degrees(np.arctan2(x, y))
        return (bearing + 360) % 360

    def calculate_bearing_difference(self, observed_trajectory, satellite_trajectory):
        observed_bearing  = self.calculate_bearing(
            observed_trajectory[0][0],  observed_trajectory[0][1],
            observed_trajectory[-1][0], observed_trajectory[-1][1],
        )
        satellite_bearing = self.calculate_bearing(
            satellite_trajectory[0][0],  satellite_trajectory[0][1],
            satellite_trajectory[-1][0], satellite_trajectory[-1][1],
        )
        bearing_diff = abs(observed_bearing - satellite_bearing)
        if bearing_diff > 180:
            bearing_diff = 360 - bearing_diff
        return bearing_diff

    def calculate_trajectory_distance_frame_ut(self, observed_positions, satellite_positions):
        altitude_range  = 90.0
        azimuth_range   = 180.0
        direction_range = 2.0
        distance        = 0
        for i in range(len(observed_positions)):
            alt_deviation = abs(observed_positions[i][0] - satellite_positions[i][0]) / altitude_range
            az_deviation  = self.azimuth_difference(observed_positions[i][1], satellite_positions[i][1]) / azimuth_range
            distance     += alt_deviation + az_deviation
        obs_dir = self.calculate_direction_vector(observed_positions[0],  observed_positions[-1])
        sat_dir = self.calculate_direction_vector(satellite_positions[0], satellite_positions[-1])
        direction_diff = np.sqrt((obs_dir[0] - sat_dir[0])**2 + (obs_dir[1] - sat_dir[1])**2) / direction_range
        return distance + direction_diff

    def azimuth_difference(self, az1, az2):
        diff = abs(az1 - az2) % 360
        if diff > 180:
            diff = 360 - diff
        return diff

    def calculate_direction_vector(self, point1, point2):
        alt_diff  = point2[0] - point1[0]
        az_diff   = self.azimuth_difference(point2[1], point1[1])
        magnitude = np.sqrt(alt_diff**2 + az_diff**2)
        return (alt_diff / magnitude, az_diff / magnitude) if magnitude != 0 else (0, 0)


# ══════════════════════════════════════════════════════════════════════════════
# TLE loading
# ══════════════════════════════════════════════════════════════════════════════

def _latest_cached_tle() -> Optional[Path]:
    TLE_DIR.mkdir(exist_ok=True)
    candidates = sorted(TLE_DIR.glob("starlink-tle-*.txt"))
    return candidates[-1] if candidates else None


def load_tle(tle_file: Optional[str] = None):
    if tle_file:
        p = Path(tle_file)
        if p.exists():
            logger.info(f"Loading TLE from explicit file: {p}")
            return load.tle_file(str(p))
        logger.warning(f"Supplied TLE file not found: {p} — falling through to fetch")

    TLE_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
    save_path = TLE_DIR / f"starlink-tle-{timestamp}.txt"

    try:
        logger.info("Fetching fresh TLE data from Celestrak ...")
        req = urllib.request.Request(TLE_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            tle_bytes = response.read()
        save_path.write_bytes(tle_bytes)
        logger.info(f"Saved TLE data to {save_path}")
        sats = load.tle_file(str(save_path))
        logger.info(f"Loaded {len(sats)} satellites from Celestrak")
        return sats
    except Exception as e:
        logger.warning(f"Failed to fetch TLE from Celestrak: {e}")

    fallback = _latest_cached_tle()
    if fallback:
        logger.warning(f"Using cached TLE file: {fallback}")
        sats = load.tle_file(str(fallback))
        logger.info(f"Loaded {len(sats)} satellites from cache")
        return sats

    raise RuntimeError("No TLE data available: fetch failed and no cached files in TLE/")


# ══════════════════════════════════════════════════════════════════════════════
# Processing thread  — mirrors process_obstruction_estimate_satellites_per_timeslot
#                      from data_feature_extraction.py, minus file saving
# ══════════════════════════════════════════════════════════════════════════════

def process_timeslot_thread(
    timeslot_df:    pd.DataFrame,
    frame_type:     int,
    df_status:      pd.DataFrame,
    satellites:     List[Any],
    df_location:    Optional[pd.DataFrame],
    timeslot_start: datetime,
    samples:        list,
    experiment_timestamp: str,
):
    try:
        data_extracter   = DataFeatureExtraction()
        sat_processor    = SatelliteProcessor()
        obstruction_map  = ObstructionMap()

        # Step 1: process_timeslot → write to in-memory CSV string
        csv_buffer = StringIO()
        writer     = csv.writer(csv_buffer)
        writer.writerow(["timestamp", "Y", "X"])
        obstruction_map.process_timeslot(timeslot_df, writer)
        obstruction_csv_str = csv_buffer.getvalue()

        if not obstruction_csv_str.strip() or obstruction_csv_str.count("\n") <= 1:
            logger.warning("No trajectory data extracted from obstruction map")
            return

        # Step 2: merge obstruction + status (+ location if mobile)
        merged_df = data_extracter.merge_obstruction_with_status_and_location(
            obstruction_csv_str, frame_type, df_status.copy(),
            df_location if config.MOBILE else None,
        )
        if merged_df.empty:
            logger.warning("Merged DataFrame is empty, skipping")
            return

        # Step 3: pixel → elevation/azimuth
        merged_df = data_extracter.pre_process_observed_data_by_frame_type(merged_df, frame_type)

        # Step 4: match satellites
        end_time   = timeslot_start + timedelta(seconds=TimeslotManager.TIMESLOT_DURATION)
        result_df  = sat_processor.process_intervals(
            merged_df, timeslot_start, end_time, satellites, frame_type,
            df_location if config.MOBILE else None,
        )

        if result_df is None or result_df.empty:
            logger.warning("No satellite match found this timeslot")
            return

        for _, row in result_df.iterrows():
            samples.append({
                "timestamp":       row["Timestamp"].strftime(TIMESTAMP_FORMAT) + "Z",
                "satellite_id":    row["Connected_Satellite"],
                "distance_km":     round(row["Distance"], 1),
                "elevation_deg":   round(row.get("Elevation", 0), 3),
                "azimuth_deg":     round(row.get("Azimuth", 0), 3),
                "experiment_time": experiment_timestamp,
            })

    except Exception as e:
        logger.error(f"Error in processing thread: {str(e)}", exc_info=True)


# ══════════════════════════════════════════════════════════════════════════════
# Status collection thread  — mirrors grpc_status_job from jobs.py
# ══════════════════════════════════════════════════════════════════════════════

class StatusCollector:
    """
    Continuously collects dish status in a background thread, exactly like
    grpc_status_job. Keeps the latest window of status rows in memory so
    each processing thread can grab a snapshot.
    """
    def __init__(self, grpc: GrpcCommand):
        self._grpc  = grpc
        self._lock  = threading.Lock()
        self._rows  = []   # list of status field lists
        self._cols  = DataFeatureExtraction().status_columns
        self._stop  = threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            current_time = time.time()
            row = self._grpc.status(current_time)
            if row:
                with self._lock:
                    self._rows.append(row)
            time.sleep(0.5)

    def get_df(self) -> pd.DataFrame:
        with self._lock:
            rows = list(self._rows)
        if not rows:
            return pd.DataFrame(columns=self._cols)
        return pd.DataFrame(rows, columns=self._cols)

    def clear(self):
        with self._lock:
            self._rows.clear()


# ══════════════════════════════════════════════════════════════════════════════
# Main loop  — duration-based, mirrors ping/iperf/starlink collectors
# ══════════════════════════════════════════════════════════════════════════════

def run(tle_file: Optional[str] = None, duration: int = 120, parsed_output_file: str = None):
    if not config.MOBILE and not all([config.LATITUDE, config.LONGITUDE, config.ALTITUDE]):
        raise ValueError("Set LATITUDE, LONGITUDE, and ALTITUDE in config.py before running.")

    satellites = load_tle(tle_file)
    grpc       = GrpcCommand()

    frame_type_int, frame_type_str = grpc.get_obstruction_map_frame_type()
    logger.info(f"Obstruction map frame type: {frame_type_str}")

    status_collector = StatusCollector(grpc)
    status_collector.start()

    samples              = []
    experiment_timestamp = datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT) + "Z"
    deadline             = time.time() + duration
    last_timeslot_second = None
    thread_pool          = []

    try:
        while time.time() < deadline:
            # Sync to Starlink network timeslot boundary — original logic from jobs.py
            if last_timeslot_second is None:
                now = datetime.now(timezone.utc)
                if now.second >= 12 and now.second < 27:
                    start_time = now.replace(microsecond=0).replace(second=27)
                    last_timeslot_second = 27
                elif now.second >= 27 and now.second < 42:
                    start_time = now.replace(microsecond=0).replace(second=42)
                    last_timeslot_second = 42
                elif now.second >= 42 and now.second < 57:
                    start_time = now.replace(microsecond=0).replace(second=57)
                    last_timeslot_second = 57
                elif now.second >= 57 and now.second < 60:
                    start_time = now.replace(microsecond=0).replace(second=12) + timedelta(minutes=1)
                    last_timeslot_second = 12
                elif now.second >= 0 and now.second < 12:
                    start_time = now.replace(microsecond=0).replace(second=12)
                    last_timeslot_second = 12

                while datetime.now(timezone.utc) < start_time:
                    time.sleep(0.1)
            else:
                last_timeslot_second = TimeslotManager.wait_until_target_time(last_timeslot_second)

            grpc.reset_obstruction_map()
            timeslot_start = datetime.now(timezone.utc).replace(microsecond=0)

            timeslot_data = grpc.collect_timeslot_data(time.time())
            if not timeslot_data:
                logger.warning("No obstruction data collected this timeslot")
                continue

            timeslot_df               = pd.DataFrame(timeslot_data)
            timeslot_df["frame_type"] = frame_type_int

            df_status = status_collector.get_df()
            status_collector.clear()

            if df_status.empty:
                logger.warning("No status data available for this timeslot, skipping")
                continue

            processing_thread = threading.Thread(
                target=process_timeslot_thread,
                args=(timeslot_df, frame_type_int, df_status, satellites, None, timeslot_start, samples, experiment_timestamp),
            )
            processing_thread.start()
            thread_pool.append(processing_thread)

    finally:
        status_collector.stop()
        for t in thread_pool:
            t.join()

        if samples:
            with open(parsed_output_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp", "satellite_id", "distance_km", "elevation_deg", "azimuth_deg", "experiment_time"])
                writer.writeheader()
                writer.writerows(samples)

        print(f"Satellite complete: {len(samples)} records")
        print(f"  Parsed: {parsed_output_file}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Starlink connected-satellite tracker")
    parser.add_argument("--tle-file",    type=str, default=None,             help="Path to local TLE file (optional)")
    parser.add_argument("--duration",    type=int, default=120,              help="Collection duration in seconds")
    parser.add_argument("--parsed-file", required=True,                      help="Output path for parsed CSV")
    parser.add_argument("--grpc",        type=str, default="192.168.100.1:9200", help="Starlink gRPC address:port")
    args = parser.parse_args()

    config.STARLINK_GRPC_ADDR_PORT = args.grpc

    print(f"Starting satellite tracker: duration={args.duration}s")
    run(tle_file=args.tle_file, duration=args.duration, parsed_output_file=args.parsed_file)

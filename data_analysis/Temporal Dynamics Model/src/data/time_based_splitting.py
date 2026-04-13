"""
Time-based Data Splitting Module for SatCom Time Series Analysis
================================================================

This module handles splitting satellite communication data based on temporal characteristics
rather than random sampling. This is crucial for time series forecasting where we need to:
1. Maintain temporal order (no data leakage from future to past)
2. Split based on actual time duration (8 min train, 2 min test)
3. Handle multiple experiments within the same dataset

Key Concepts:
- Training data: First 8 minutes of each experiment
- Testing data: Last 2 minutes of each experiment
- Each experiment is split individually to maintain temporal integrity
- Fallback mechanisms for edge cases (short experiments, missing columns)

Author: SatCom TimeSeries Analysis Framework
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
from src.evaluation.standardized_evaluation import STANDARD_TARGET_COLUMNS


class TimeBasedDataSplitter:
    """
    Handles time-based data splitting for satellite communication experiments.

    This class implements a sophisticated splitting strategy that respects the temporal
    nature of satellite communication data. Instead of random splits, it uses actual
    time information to ensure realistic evaluation scenarios.

    Attributes:
        data_dir (str): Directory containing processed data files
        train_duration_minutes (float): Duration for training data (default: 8.0 minutes)
        test_duration_minutes (float): Duration for testing data (default: 2.0 minutes)
        total_duration_minutes (float): Total expected experiment duration (default: 10.0 minutes)

    Example:
        >>> splitter = TimeBasedDataSplitter("processed_data")
        >>> train_data, test_data = splitter.get_train_test_data()
        >>> print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    """

    def __init__(self, data_dir: str = "processed_data"):
        """
        Initialize the TimeBasedDataSplitter.

        Args:
            data_dir (str): Path to directory containing processed CSV files
                           Expected files: combined_DL_train.csv, combined_DL_test.csv,
                                         combined_UL_train.csv, combined_UL_test.csv

        Note:
            The splitter will re-combine and re-split the existing train/test files
            based on temporal characteristics rather than the original arbitrary split.
        """
        self.data_dir = data_dir

        # Time-based splitting configuration
        # These values define the temporal split strategy
        self.train_duration_minutes = 8.0  # First 8 minutes -> training set
        self.test_duration_minutes = 2.0  # Last 2 minutes -> test set
        self.total_duration_minutes = 10.0  # Expected total experiment duration

        print(f"TimeBasedDataSplitter initialized:")
        print(f"  - Data directory: {data_dir}")
        print(f"  - Training duration: {self.train_duration_minutes} minutes")
        print(f"  - Testing duration: {self.test_duration_minutes} minutes")
        print(f"  - Expected total duration: {self.total_duration_minutes} minutes")

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and combine all raw data files from the data directory.

        This method loads the four CSV files (DL/UL train/test) and combines them
        into a single dataset. The original train/test split is ignored because
        we need to re-split based on temporal characteristics.

        File Structure Expected:
        - data/{data_dir}/combined_DL_train.csv  (Downlink training data)
        - data/{data_dir}/combined_DL_test.csv   (Downlink test data)
        - data/{data_dir}/combined_UL_train.csv  (Uplink training data)
        - data/{data_dir}/combined_UL_test.csv   (Uplink test data)

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - First element: Combined dataset with all samples
                - Second element: Same combined dataset (for consistency with interface)

        Raises:
            FileNotFoundError: If any of the expected CSV files are missing
            pd.errors.EmptyDataError: If CSV files are empty or corrupted

        Note:
            Both return values are the same combined dataset. The original train/test
            files are merged because we need to re-split based on time, not the
            original arbitrary split.
        """
        print(f"\n=== LOADING RAW DATA ===")
        print(f"Loading raw data from {self.data_dir}/...")

        # Define expected file paths
        dl_train_path = f"data/{self.data_dir}/combined_DL_train.csv"
        dl_test_path = f"data/{self.data_dir}/combined_DL_test.csv"
        ul_train_path = f"data/{self.data_dir}/combined_UL_train.csv"
        ul_test_path = f"data/{self.data_dir}/combined_UL_test.csv"

        file_info = []

        try:
            # Load all four data files
            print("Loading individual files:")

            # Downlink data
            dl_train = pd.read_csv(dl_train_path)
            print(f"  - DL Train: {len(dl_train):,} samples from {dl_train_path}")
            file_info.append(("DL Train", len(dl_train)))

            dl_test = pd.read_csv(dl_test_path)
            print(f"  - DL Test: {len(dl_test):,} samples from {dl_test_path}")
            file_info.append(("DL Test", len(dl_test)))

            # Uplink data
            ul_train = pd.read_csv(ul_train_path)
            print(f"  - UL Train: {len(ul_train):,} samples from {ul_train_path}")
            file_info.append(("UL Train", len(ul_train)))

            ul_test = pd.read_csv(ul_test_path)
            print(f"  - UL Test: {len(ul_test):,} samples from {ul_test_path}")
            file_info.append(("UL Test", len(ul_test)))

        except FileNotFoundError as e:
            print(f"ERROR: Could not find data file: {e}")
            raise
        except pd.errors.EmptyDataError as e:
            print(f"ERROR: Empty or corrupted data file: {e}")
            raise

        # Combine DL (Downlink) data: train + test
        print("\nCombining data:")
        dl_combined = pd.concat([dl_train, dl_test], ignore_index=True)
        print(f"  - DL Combined: {len(dl_combined):,} samples")

        # Combine UL (Uplink) data: train + test
        ul_combined = pd.concat([ul_train, ul_test], ignore_index=True)
        print(f"  - UL Combined: {len(ul_combined):,} samples")

        # Final combination: DL + UL
        combined = pd.concat([dl_combined, ul_combined], ignore_index=True)
        print(f"  - Final Combined: {len(combined):,} total samples")

        # Display column information
        print(f"\nDataset Info:")
        print(f"  - Columns: {list(combined.columns)}")
        print(f"  - Shape: {combined.shape}")

        # Check for key columns needed for time-based splitting
        required_columns = ["experiment_id", "time_since_start", "timestamp"]
        available_columns = []
        for col in required_columns:
            if col in combined.columns:
                available_columns.append(col)
                print(f"  Found {col} column")
            else:
                print(f"  Missing {col} column")

        if not available_columns:
            print(
                "  WARNING: No time-related columns found! Will use fallback splitting."
            )

        print("=== RAW DATA LOADING COMPLETE ===\n")

        # Return the same combined data twice (for interface compatibility)
        return combined, combined

    def split_by_time(
        self,
        data: pd.DataFrame,
        timestamp_col: Optional[str] = None,
        train_ratio: float = 0.8,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data based on temporal characteristics using multiple fallback strategies.

        This is the core method that implements the time-based splitting logic.
        It tries multiple strategies in order of preference:

        1. PRIMARY STRATEGY: Use `time_since_start` column for precise time-based splitting
        2. SECONDARY STRATEGY: Use `timestamp` column and calculate time differences
        3. FALLBACK STRATEGY: Use train_ratio for proportional splitting when no time info

        Splitting Logic:
        - For experiments >= 10 minutes: First 8 minutes -> train, Last 2 minutes -> test
        - For shorter experiments: Use train_ratio (default 80/20) split
        - For single-row experiments: Put entire row in training set
        - Handle empty data gracefully

        Args:
            data (pd.DataFrame): Input dataset to split
            timestamp_col (Optional[str]): Name of timestamp column (auto-detected if None)
            train_ratio (float): Fallback ratio for splitting when no time info (default: 0.8)

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (training_data, testing_data)

        Time-Based Splitting Details:
            The method processes each experiment individually to maintain temporal integrity:

            Experiment Timeline (10 minutes total):
            |-------- Training (8 min) --------|-- Testing (2 min) --|
            0min                              8min                 10min


        Edge Cases Handled:
        - Empty dataset -> returns empty train and test DataFrames
        - Missing experiment_id -> treats entire dataset as single experiment
        - No time columns -> uses proportional splitting
        - Short experiments (< 10 min) -> uses train_ratio split
        - Single row -> goes to training set

        Example:
            >>> splitter = TimeBasedDataSplitter()
            >>> train, test = splitter.split_by_time(data, timestamp_col='timestamp')
            >>> print(f"Split ratio: {len(train)}/{len(test)} = {len(train)/(len(train)+len(test)):.2f}")
        """
        print(f"\n=== APPLYING TIME-BASED SPLITTING ===")
        print(
            f"Split configuration: {self.train_duration_minutes}min train, {self.test_duration_minutes}min test"
        )
        print(f"Input data: {len(data):,} samples")

        # === EDGE CASE: Empty Data ===
        if data.empty or len(data) == 0:
            print("WARNING: Empty dataset provided to splitter")
            print("Returning empty train and test DataFrames")
            return pd.DataFrame(), pd.DataFrame()

        # === PREPROCESSING: Ensure Required Columns ===
        data_copy = data.copy()  # Work with copy to avoid modifying original

        # Check/create experiment_id column
        if "experiment_id" not in data_copy.columns:
            print("WARNING: No experiment_id column found")
            print("Creating default experiment_id for entire dataset")
            data_copy["experiment_id"] = "default_experiment"

        # === GROUP BY EXPERIMENT AND PROCESS INDIVIDUALLY ===
        experiment_groups = data_copy.groupby("experiment_id")
        print(f"Found {len(experiment_groups)} unique experiments to process")

        # Collections for results
        train_data_list = []
        test_data_list = []
        split_summary = []

        # Process each experiment separately
        for exp_id, exp_data in experiment_groups:
            print(f"\n--- Processing Experiment: {exp_id} ---")
            print(f"Experiment data: {len(exp_data)} samples")

            # Split this single experiment
            exp_train, exp_test = self._split_single_experiment(
                exp_data, exp_id, timestamp_col, train_ratio
            )

            # Collect results (even if one side is empty for edge cases)
            if len(exp_train) > 0 or len(exp_test) > 0:
                train_data_list.append(exp_train)
                test_data_list.append(exp_test)

                # Record split summary
                split_info = {
                    "experiment_id": exp_id,
                    "original_samples": len(exp_data),
                    "train_samples": len(exp_train),
                    "test_samples": len(exp_test),
                    "train_ratio": (
                        len(exp_train) / len(exp_data) if len(exp_data) > 0 else 0
                    ),
                }
                split_summary.append(split_info)

                print(
                    f"Split result: {len(exp_train)} train, {len(exp_test)} test samples"
                )
                print(f"Split ratio: {split_info['train_ratio']:.2%}")
            else:
                print(f"WARNING: No valid data from experiment {exp_id}")

        # === COMBINE RESULTS FROM ALL EXPERIMENTS ===
        if train_data_list and test_data_list:
            # Concatenate all experiment results
            train_data = pd.concat(train_data_list, ignore_index=True)
            test_data = pd.concat(test_data_list, ignore_index=True)

            # Clean up temporary columns created during processing
            temp_columns = ["time_minutes", "time_since_start_temp", "temp_timestamp"]
            for temp_col in temp_columns:
                train_data = train_data.drop(temp_col, axis=1, errors="ignore")
                test_data = test_data.drop(temp_col, axis=1, errors="ignore")

            print(f"\n=== SPLITTING COMPLETE ===")
            print(f"Final Results:")
            print(f"  - Total training samples: {len(train_data):,}")
            print(f"  - Total testing samples: {len(test_data):,}")
            print(
                f"  - Overall split ratio: {len(train_data)/(len(train_data)+len(test_data)):.2%} train"
            )
            print(f"  - Experiments processed: {len(split_summary)}")

            # Display per-experiment summary
            print(f"\nPer-Experiment Summary:")
            for info in split_summary:
                print(
                    f"  {info['experiment_id']}: {info['train_samples']}/{info['test_samples']} "
                    f"({info['train_ratio']:.1%} train)"
                )

        else:
            print("ERROR: No valid train/test splits generated from any experiments")
            train_data = pd.DataFrame()
            test_data = pd.DataFrame()

        print("=== TIME-BASED SPLITTING COMPLETE ===\n")
        return train_data, test_data

    def _split_single_experiment(
        self,
        exp_data: pd.DataFrame,
        exp_id: str,
        timestamp_col: Optional[str] = None,
        train_ratio: float = 0.8,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split a single experiment's data using the best available time information.

        This method implements the core splitting logic for individual experiments.
        It uses a hierarchical strategy to determine the best splitting approach:

        STRATEGY 1: time_since_start column (PREFERRED)
        - Uses precise time measurements from experiment start
        - Splits at exactly 8.0 minutes for training/testing boundary

        STRATEGY 2: timestamp column (SECONDARY)
        - Calculates time differences from timestamp data
        - Converts to minutes since experiment start
        - Applies same 8.0 minute split threshold

        STRATEGY 3: Proportional split (FALLBACK)
        - Uses train_ratio when no time information available
        - Default 80/20 split maintains reasonable evaluation setup

        Args:
            exp_data (pd.DataFrame): Data for a single experiment
            exp_id (str): Experiment identifier for logging
            timestamp_col (Optional[str]): Specific timestamp column to use
            train_ratio (float): Fallback ratio for non-temporal splitting

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (experiment_train_data, experiment_test_data)

        Edge Cases:
        - Single row -> goes entirely to training set
        - Short experiments (< 10 min) -> uses proportional split
        - Missing time columns -> uses proportional split

        Time Calculation Examples:
            For time_since_start in seconds:
            - 0-480 seconds (0-8 min) -> Training set
            - 481-600 seconds (8-10 min) -> Testing set

            For timestamp data:
            - Calculate: minutes = (timestamp - first_timestamp) / 60
            - Apply same 8-minute threshold
        """
        print(f"    Analyzing experiment {exp_id}: {len(exp_data)} samples")

        # === EDGE CASE: Single row ===
        if len(exp_data) <= 1:
            print(f"    Single row detected -> assigning to training set")
            return exp_data.copy(), pd.DataFrame(columns=exp_data.columns)

        # Sort data to ensure proper temporal ordering
        exp_data_sorted = exp_data.copy()

        # === STRATEGY 1: Use time_since_start column (PREFERRED) ===
        if "time_since_start" in exp_data.columns:
            print(f"    Using time_since_start column for temporal splitting")

            # Sort by time_since_start to ensure chronological order
            exp_data_sorted = exp_data_sorted.sort_values(
                "time_since_start"
            ).reset_index(drop=True)

            # Convert time_since_start (assumed to be in seconds) to minutes
            exp_data_sorted["time_minutes"] = exp_data_sorted["time_since_start"] / 60.0

            # Calculate experiment duration
            min_time = exp_data_sorted["time_minutes"].min()
            max_time = exp_data_sorted["time_minutes"].max()
            duration = max_time - min_time

            print(
                f"    Experiment duration: {duration:.2f} minutes (from {min_time:.2f} to {max_time:.2f})"
            )

            # Check if experiment is long enough for time-based splitting
            if duration < self.total_duration_minutes:
                print(
                    f"    Short experiment ({duration:.1f} < {self.total_duration_minutes:.1f} min)"
                )
                print(
                    f"    Using {train_ratio:.0%}/{1-train_ratio:.0%} proportional split instead"
                )

                train_size = max(
                    1, int(train_ratio * len(exp_data_sorted))
                )  # At least 1 sample for training
                return (
                    exp_data_sorted.iloc[:train_size].copy(),
                    exp_data_sorted.iloc[train_size:].copy(),
                )

            # Apply time-based split at configured threshold
            train_threshold = min_time + self.train_duration_minutes
            print(f"    Split threshold: {train_threshold:.2f} minutes")

            # Create boolean masks for splitting
            train_mask = exp_data_sorted["time_minutes"] <= train_threshold
            test_mask = exp_data_sorted["time_minutes"] > train_threshold

            train_count = train_mask.sum()
            test_count = test_mask.sum()
            print(f"    Time-based split: {train_count} train, {test_count} test")

            return (
                exp_data_sorted[train_mask].copy(),
                exp_data_sorted[test_mask].copy(),
            )

        # === STRATEGY 2: Use timestamp column (SECONDARY) ===
        elif timestamp_col and timestamp_col in exp_data.columns:
            print(f"    Using {timestamp_col} column for temporal splitting")
            return self._split_by_timestamp(
                exp_data_sorted, exp_id, timestamp_col, train_ratio
            )

        elif "timestamp" in exp_data.columns:
            print(f"    Using timestamp column for temporal splitting")
            return self._split_by_timestamp(
                exp_data_sorted, exp_id, "timestamp", train_ratio
            )

        # === STRATEGY 3: Proportional split (FALLBACK) ===
        else:
            print(
                f"    No time columns found -> using {train_ratio:.0%}/{1-train_ratio:.0%} proportional split"
            )

            train_size = max(
                1, int(train_ratio * len(exp_data_sorted))
            )  # Ensure at least 1 training sample

            print(
                f"    Proportional split: {train_size} train, {len(exp_data_sorted)-train_size} test"
            )

            return (
                exp_data_sorted.iloc[:train_size].copy(),
                exp_data_sorted.iloc[train_size:].copy(),
            )

    def _split_by_timestamp(
        self,
        exp_data: pd.DataFrame,
        exp_id: str,
        timestamp_col: str,
        train_ratio: float,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split experiment data using timestamp information.

        This helper method handles timestamp-based splitting when time_since_start
        is not available. It converts timestamps to relative time and applies
        the same temporal splitting logic.

        Args:
            exp_data (pd.DataFrame): Experiment data with timestamp column
            exp_id (str): Experiment ID for logging
            timestamp_col (str): Name of the timestamp column
            train_ratio (float): Fallback ratio if timestamp-based splitting fails

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_data, test_data)

        Process:
        1. Convert timestamp column to datetime if needed
        2. Sort by timestamp to ensure temporal order
        3. Calculate relative time in minutes from experiment start
        4. Apply 8-minute threshold for splitting
        5. Fall back to proportional split for short experiments
        """
        try:
            # Ensure timestamp column is datetime
            if not pd.api.types.is_datetime64_any_dtype(exp_data[timestamp_col]):
                print(f"    Converting {timestamp_col} to datetime")
                exp_data[timestamp_col] = pd.to_datetime(exp_data[timestamp_col])

            # Sort by timestamp
            exp_data_sorted = exp_data.sort_values(timestamp_col).reset_index(drop=True)

            # Calculate relative time in minutes from experiment start
            start_time = exp_data_sorted[timestamp_col].iloc[0]
            exp_data_sorted["time_minutes"] = (
                exp_data_sorted[timestamp_col] - start_time
            ).dt.total_seconds() / 60.0

            # Calculate experiment duration
            duration = exp_data_sorted["time_minutes"].max()
            print(f"    Timestamp-based duration: {duration:.2f} minutes")

            # Apply time-based splitting if duration is sufficient
            if duration >= self.total_duration_minutes:
                print(
                    f"    Applying {self.train_duration_minutes}-minute temporal threshold"
                )

                train_mask = (
                    exp_data_sorted["time_minutes"] <= self.train_duration_minutes
                )
                test_mask = (
                    exp_data_sorted["time_minutes"] > self.train_duration_minutes
                )

                return (
                    exp_data_sorted[train_mask].copy(),
                    exp_data_sorted[test_mask].copy(),
                )
            else:
                print(
                    f"    Short experiment ({duration:.1f} min) -> proportional split"
                )
                train_size = max(1, int(train_ratio * len(exp_data_sorted)))
                return (
                    exp_data_sorted.iloc[:train_size].copy(),
                    exp_data_sorted.iloc[train_size:].copy(),
                )

        except Exception as e:
            print(f"    ERROR processing timestamps: {e}")
            print(f"    Falling back to proportional split")

            train_size = max(1, int(train_ratio * len(exp_data)))
            return (
                exp_data.iloc[:train_size].copy(),
                exp_data.iloc[train_size:].copy(),
            )

    def get_train_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main interface method to load raw data and perform time-based splitting.

        This is the primary method users should call to get properly split training
        and testing datasets. It orchestrates the entire pipeline:

        1. Data Loading: Load and combine all raw CSV files
        2. Time-Based Splitting: Apply sophisticated temporal splitting logic
        3. Validation: Verify split results and provide summary statistics
        4. Return: Clean training and testing DataFrames

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (training_data, testing_data)

        Process Flow:
        Raw Files -> load_raw_data() -> Combined Dataset -> split_by_time() -> Train/Test

        The method ensures:
        - Proper temporal order is maintained
        - No data leakage from future to past
        - Each experiment is split individually
        - Comprehensive logging of the process
        - Graceful handling of edge cases

        Example Usage:
            >>> splitter = TimeBasedDataSplitter("processed_data")
            >>> train_data, test_data = splitter.get_train_test_data()
            >>> print(f"Ready for training: {len(train_data)} samples")
            >>> print(f"Ready for testing: {len(test_data)} samples")

        Raises:
            FileNotFoundError: If required CSV files are missing
            pd.errors.EmptyDataError: If CSV files are empty or corrupted
            ValueError: If no valid splits can be generated
        """
        print(f"\n" + "=" * 60)
        print(f"STARTING TIME-BASED DATA SPLITTING PIPELINE")
        print(f"=" * 60)

        try:
            # === STEP 1: Load Raw Data ===
            print(f"STEP 1: Loading raw data files...")
            combined_data, _ = self.load_raw_data()

            if combined_data.empty:
                raise ValueError("No data loaded from files")

            # === STEP 2: Apply Time-Based Splitting ===
            print(f"STEP 2: Applying time-based splitting logic...")
            train_data, test_data = self.split_by_time(combined_data)

            # === STEP 3: Validation and Summary ===
            print(f"STEP 3: Validating results and generating summary...")

            if train_data.empty and test_data.empty:
                raise ValueError("No valid train/test splits generated")

            # Final processing and validation
            total_samples = len(train_data) + len(test_data)
            train_percentage = (
                (len(train_data) / total_samples * 100) if total_samples > 0 else 0
            )

            print(f"\n" + "=" * 60)
            print(f"DATA SPLITTING PIPELINE COMPLETE")
            print(f"=" * 60)
            print(f"FINAL RESULTS:")
            print(f"  Training samples: {len(train_data):,} ({train_percentage:.1f}%)")
            print(
                f"  Testing samples:  {len(test_data):,} ({100-train_percentage:.1f}%)"
            )
            print(f"  Total samples:    {total_samples:,}")

            # Data quality checks
            if len(train_data) == 0:
                print(f"  WARNING: No training data generated!")
            if len(test_data) == 0:
                print(f"  WARNING: No testing data generated!")

            # Column consistency check
            if set(train_data.columns) != set(test_data.columns):
                print(f"  WARNING: Train and test have different columns!")

            print(f"  Pipeline completed successfully")
            print(f"=" * 60 + "\n")

            return train_data, test_data

        except Exception as e:
            print(f"\nERROR in data splitting pipeline: {e}")
            print(f"Returning empty DataFrames")
            print(f"=" * 60 + "\n")

            # Return empty DataFrames with error state
            return pd.DataFrame(), pd.DataFrame()

    def get_dataset_summary(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> Dict:
        """
        Generate comprehensive summary statistics for the split dataset.

        This method provides detailed analytics about the train/test split including:
        - Sample counts and ratios
        - Experiment distribution
        - Time range coverage
        - Target variable availability
        - Data quality metrics

        Args:
            train_data (pd.DataFrame): Training dataset
            test_data (pd.DataFrame): Testing dataset

        Returns:
            Dict: Comprehensive dataset summary with the following structure:
            {
                'total_samples': int,           # Total number of samples
                'train_samples': int,           # Training samples count
                'test_samples': int,            # Testing samples count
                'train_ratio': float,           # Training proportion (0.0-1.0)
                'experiments_count': int,       # Number of unique experiments
                'target_columns': List[str],    # Available target columns
                'time_range': Dict,             # Time coverage information
                'data_quality': Dict            # Quality metrics
            }
        """
        print(f"\n=== GENERATING DATASET SUMMARY ===")

        # === Basic Count Statistics ===
        total_samples = len(train_data) + len(test_data)
        train_ratio = len(train_data) / total_samples if total_samples > 0 else 0.0

        summary = {
            "total_samples": total_samples,
            "train_samples": len(train_data),
            "test_samples": len(test_data),
            "train_ratio": train_ratio,
            "experiments_count": 0,
            "target_columns": STANDARD_TARGET_COLUMNS.copy(),
            "time_range": {},
            "data_quality": {},
        }

        print(f"Basic statistics calculated:")
        print(f"  - Total samples: {total_samples:,}")
        print(f"  - Train/test ratio: {train_ratio:.2%} / {1-train_ratio:.2%}")

        # === Experiment Count Analysis ===
        try:
            train_experiments = set()
            test_experiments = set()

            if "experiment_id" in train_data.columns and len(train_data) > 0:
                train_experiments = set(train_data["experiment_id"].unique())
                print(f"  - Training experiments: {len(train_experiments)}")

            if "experiment_id" in test_data.columns and len(test_data) > 0:
                test_experiments = set(test_data["experiment_id"].unique())
                print(f"  - Testing experiments: {len(test_experiments)}")

            # Count unique experiments across both sets
            all_experiments = train_experiments.union(test_experiments)
            summary["experiments_count"] = len(all_experiments)

            # Check for experiment overlap
            overlap = train_experiments.intersection(test_experiments)
            if overlap:
                print(
                    f"  - Experiments in both sets: {len(overlap)} (expected for time-based splits)"
                )

        except Exception as e:
            print(f"  - Could not analyze experiments: {e}")

        # === Time Range Analysis ===
        try:
            if "timestamp" in train_data.columns and len(train_data) > 0:
                summary["time_range"]["train_start"] = str(
                    train_data["timestamp"].min()
                )
                summary["time_range"]["train_end"] = str(train_data["timestamp"].max())
                print(
                    f"  - Training time range: {summary['time_range']['train_start']} to {summary['time_range']['train_end']}"
                )

            if "timestamp" in test_data.columns and len(test_data) > 0:
                summary["time_range"]["test_start"] = str(test_data["timestamp"].min())
                summary["time_range"]["test_end"] = str(test_data["timestamp"].max())
                print(
                    f"  - Testing time range: {summary['time_range']['test_start']} to {summary['time_range']['test_end']}"
                )

            # Check for temporal ordering (test should come after train)
            if (
                "train_end" in summary["time_range"]
                and "test_start" in summary["time_range"]
            ):
                train_end = pd.to_datetime(summary["time_range"]["train_end"])
                test_start = pd.to_datetime(summary["time_range"]["test_start"])
                if test_start >= train_end:
                    print(f"  Proper temporal ordering maintained")
                else:
                    print(f"  WARNING: Temporal overlap detected!")

        except Exception as e:
            print(f"  - Could not analyze time ranges: {e}")

        print(f"=== DATASET SUMMARY COMPLETE ===\n")

        return summary

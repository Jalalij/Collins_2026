import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate models on time series data"
    )

    # Paths
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../processed_data",
        help="Directory containing input CSV files",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/",
        help="Directory to save results and plots",
    )

    # Training
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    parser.add_argument(
        "--direction",
        type=str,
        default="UL",
        choices=["UL", "DL"],
        help="Direction of the data (UL/DL)",
    )

    parser.add_argument(
        "--filesplit",
        type=bool,
        default=False,
        help="Whether to split the data by file",
    )

    parser.add_argument(
        "--smoke_test",
        type=bool,
        default=False,
        help="Run a smoke test (quick test with fewer epochs)",
    )

    # Time-series windowing
    parser.add_argument(
        "--window_size",
        type=int,
        default=1,
        help="Number of timesteps in each input window",
    )
    parser.add_argument(
        "--minutes",
        type=int,
        default=8,
        help="how many minutes of data to use for training (8 min train / 2 min test split)",
    )
    parser.add_argument(
        "--target_hz",
        type=int,
        default=5,
        help="Target frequency in Hz (e.g., 5 for 5 Hz data)",
    )

    args = parser.parse_args()
    return args


args = get_args()

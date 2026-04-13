import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from .standardized_data_prep import prepare_standardized_data


class ExperimentDataset(Dataset):
    def __init__(
        self,
        files,
        direction="UL",
        mode="train",
        window_size=10,
        minutes=8,
        target_hz=None,
        filesplit=False,
    ):
        """
        Args:
            files (list[str]): List of CSV file paths (one per experiment).
            direction (str): "UL" or "DL".
            mode (str): "train" or "test".
        """
        self.samples = []
        self.X = []
        self.y = []
        files = [f for f in files if f"_{direction}.csv" in f]

        self.target_hz = target_hz
        self.filesplit = filesplit
        self.mode = mode

        for fpath in files:
            df = pd.read_csv(fpath, parse_dates=["timestamp"])
            df = df.sort_values("timestamp")

            if self.target_hz is not None:
                dt = pd.Timedelta(seconds=1 / self.target_hz)

                df = df.set_index("timestamp")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df_num = df[numeric_cols].resample(dt).mean().interpolate()

                df = df_num.reset_index()

            if self.filesplit:
                # ---- file split ----
                train_raw = df
                test_raw = df
            else:
                # ---- 8 / 2 minute split ----
                cutoff = df["timestamp"].min() + pd.Timedelta(minutes=minutes)
                train_raw = df[df["timestamp"] <= cutoff]
                test_raw = df[df["timestamp"] > cutoff]

            # ---- standardized prep ----
            result = prepare_standardized_data(train_raw, test_raw)

            data = result["train_raw"] if self.mode == "train" else result["test_raw"]

            num_features = data.shape[1]

            self.num_features = int(window_size * num_features)

            for i in range(len(data) - window_size):
                # take 128 timesteps × 7 features → flatten
                x_window = (
                    data[i : i + window_size].astype(np.float32).reshape(-1)
                )  # [896]
                y_next = data[i + window_size].astype(np.float32)

                self.X.append(torch.tensor(x_window))  # shape [896]
                self.y.append(torch.tensor(y_next))  # shape [7]

        self.transform()

    def transform(self):
        X_stack = torch.stack(self.X)  # shape [N, D]
        y_stack = torch.stack(self.y)  # shape [N, F]

        # Compute per-feature min/max across X and y combined
        data_stack = torch.cat([X_stack, y_stack], dim=0)
        self.min = data_stack.min(dim=0).values  # shape [D]
        self.max = data_stack.max(dim=0).values
        self.scale = (self.max - self.min).clamp(min=1e-8)  # avoid div-by-zero

        # Apply scaling
        self.X = [(x - self.min) / self.scale for x in self.X]
        self.y = [
            (y - self.min[: y.shape[0]]) / self.scale[: y.shape[0]] for y in self.y
        ]

        if self.filesplit:
            # If filesplit is True, convert lists to tensors for easier indexing
            if self.mode == "train":
                self.X = self.X[: int(len(self.X) * 0.8)]
                self.y = self.y[: int(len(self.y) * 0.8)]
            else:
                self.X = self.X[int(len(self.X) * 0.8) :]
                self.y = self.y[int(len(self.y) * 0.8) :]

    def inverse_transform(self, input_data):
        # Ensure min and scale are on the same device as input_data
        device = (
            input_data.device
            if isinstance(input_data, torch.Tensor)
            else torch.device("cpu")
        )
        min_ = self.min.to(device)
        scale = self.scale.to(device)

        if isinstance(input_data, torch.Tensor):
            return input_data * scale + min_
        elif isinstance(input_data, list):
            return [x.to(device) * scale + min_ for x in input_data]
        else:
            raise TypeError("input_data must be a torch.Tensor or list of tensors")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

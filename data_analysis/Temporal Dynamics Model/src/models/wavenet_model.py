#!/usr/bin/env python3
"""
WaveNet Model Implementation – matches LSTM/Transformer trainer interface

- Trainer signature:
    train_and_evaluate(train_set, test_set, evaluator, results_dir, direction="UL")
- Accepts inputs shaped [B, T, F] or flattened [B, D=T*F]; predicts next-step [B, F]
- Uses ExperimentDataset.inverse_transform for reporting in original units
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime
from typing import Dict, Any, Tuple
import warnings

warnings.filterwarnings("ignore")

from ..evaluation.standardized_evaluation import STANDARD_TARGET_COLUMNS


# ----------------------------- WaveNet -----------------------------


class WaveNet(nn.Module):
    """
    Causal 1D WaveNet block for multivariate forecasting.

    Input to forward(): x as [B, F, T]  (channels-first for Conv1d)
    Output: next-step prediction [B, F]
    """

    def __init__(
        self,
        input_channels: int,  # F
        residual_channels: int = 32,
        dilation_channels: int = 32,  # must be even (gating splits in half)
        skip_channels: int = 64,
        end_channels: int = 64,
        num_layers: int = 6,
        num_cycles: int = 1,
        kernel_size: int = 2,
        output_size: int | None = None,  # F by default
    ):
        super().__init__()
        assert (
            dilation_channels % 2 == 0
        ), "dilation_channels must be even for gated activation"

        if output_size is None:
            output_size = input_channels

        self.input_channels = input_channels
        self.output_size = output_size

        self.start = nn.Conv1d(input_channels, residual_channels, kernel_size=1)

        self.dilated = nn.ModuleList()
        self.residual = nn.ModuleList()
        self.skip = nn.ModuleList()

        total_layers = num_layers * num_cycles
        for i in range(total_layers):
            d = 2 ** (i % num_layers)
            # use causal padding = (kernel_size-1)*d left padding; implement via padding then crop
            self.dilated.append(
                nn.Conv1d(
                    residual_channels,
                    dilation_channels,
                    kernel_size,
                    dilation=d,
                    padding=(kernel_size - 1) * d,
                )
            )
            self.residual.append(
                nn.Conv1d(dilation_channels // 2, residual_channels, kernel_size=1)
            )
            self.skip.append(
                nn.Conv1d(dilation_channels // 2, skip_channels, kernel_size=1)
            )

        self.end1 = nn.Conv1d(skip_channels, end_channels, kernel_size=1)
        self.end2 = nn.Conv1d(end_channels, output_size, kernel_size=1)

    def _causal_crop(self, x: torch.Tensor, d: int, k: int) -> torch.Tensor:
        """Remove right padding to enforce causality after 'same' padding."""
        crop = (k - 1) * d
        if crop == 0:
            return x
        return x[:, :, :-crop]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, F, T]
        returns: [B, F] (next-step prediction)
        """
        B, C, T = x.shape
        h = self.start(x)  # [B, R, T]
        skip_sum = 0

        for conv, res, sk in zip(self.dilated, self.residual, self.skip):
            d = conv.dilation[0]
            k = conv.kernel_size[0]

            z = conv(h)  # [B, D, T+pad]
            z = self._causal_crop(z, d, k)  # [B, D, T]
            # Gated activation
            D = z.size(1)
            z_tanh = torch.tanh(z[:, : D // 2, :])
            z_sig = torch.sigmoid(z[:, D // 2 :, :])
            z = z_tanh * z_sig  # [B, D/2, T]

            # Skip & residual
            skip_out = sk(z)  # [B, S, T]
            skip_sum = skip_out if isinstance(skip_sum, int) else (skip_sum + skip_out)
            h = res(z) + h  # [B, R, T]

        y = F.relu(skip_sum)
        y = F.relu(self.end1(y))
        y = self.end2(y)  # [B, F, T]

        # Use the last time index as next-step prediction head
        return y[:, :, -1]  # [B, F]


# ----------------------------- Trainer -----------------------------


class WaveNetTrainer:
    """WaveNet trainer aligned with LSTM/Transformer trainers."""

    @staticmethod
    def _infer_dims_from_loader(loader: DataLoader) -> Tuple[int, int, bool]:
        """
        Return (seq_len, num_vars, is_flat) from first batch.
        Accepts xb as [B,T,F] or [B,D=T*F]; yb as [B,F]
        """
        xb, yb = next(iter(loader))

        if xb.dim() == 3:
            # [B, T, F]
            _, T, F = xb.shape
            if yb.dim() != 2 or yb.shape[1] != F:
                raise ValueError(f"Expected targets [B,F={F}], got {tuple(yb.shape)}")
            return T, F, False

        if xb.dim() == 2:
            # [B, D] → infer F
            _, D = xb.shape
            F = len(STANDARD_TARGET_COLUMNS) if len(STANDARD_TARGET_COLUMNS) > 0 else 7
            if D % F != 0:
                raise ValueError(
                    f"Flat width D={D} not divisible by F={F}; cannot reshape to [B,T,F]."
                )
            T = D // F
            if yb.dim() != 2 or yb.shape[1] != F:
                raise ValueError(f"Expected targets [B,F={F}], got {tuple(yb.shape)}")
            return T, F, True

        raise ValueError(f"Unsupported input shape {tuple(xb.shape)}")

    @staticmethod
    def _reshape_if_flat(xb: torch.Tensor, seq_len: int, num_vars: int) -> torch.Tensor:
        """Return [B, F, T] for the WaveNet (channels-first)."""
        if xb.dim() == 2:
            B, D = xb.shape
            if D != seq_len * num_vars:
                raise ValueError(f"Cannot reshape [B,{D}] to [B,{seq_len},{num_vars}]")
            xb = xb.view(B, seq_len, num_vars)  # [B, T, F]
        # now xb is [B, T, F]
        return xb.permute(0, 2, 1)  # [B, F, T]

    @staticmethod
    def train_and_evaluate(
        train_set,
        test_set,
        evaluator,
        results_dir: str,
        direction: str = "UL",
        batch_size: int = 64,
        max_epochs: int = 100,
        patience: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ) -> Tuple[Dict[str, Any], Dict, Dict]:
        print("\n" + "=" * 20 + " WAVENET MODEL " + "=" * 20)

        try:
            # DataLoaders
            train_loader = DataLoader(
                train_set, batch_size=batch_size, shuffle=True, drop_last=True
            )
            test_loader = DataLoader(
                test_set, batch_size=batch_size, shuffle=False, drop_last=False
            )

            # Infer dims
            seq_len, num_vars, _ = WaveNetTrainer._infer_dims_from_loader(train_loader)

            # Model / optim
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = WaveNet(input_channels=num_vars, output_size=num_vars).to(device)

            criterion = nn.MSELoss()
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, patience=5, factor=0.5
            )

            # Train loop
            best_val = float("inf")
            no_improve = 0
            train_losses, val_losses = [], []
            start_time = datetime.now()

            for epoch in range(max_epochs):
                model.train()
                ep_train = 0.0
                for xb, yb in train_loader:
                    xb = WaveNetTrainer._reshape_if_flat(xb, seq_len, num_vars).to(
                        device
                    )  # [B,F,T]
                    yb = yb.to(device)  # [B,F]

                    opt.zero_grad()
                    out = model(xb)  # [B,F]
                    loss = criterion(out, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    ep_train += loss.item()

                model.eval()
                ep_val = 0.0
                with torch.no_grad():
                    for xb, yb in test_loader:
                        xb = WaveNetTrainer._reshape_if_flat(xb, seq_len, num_vars).to(
                            device
                        )
                        yb = yb.to(device)
                        out = model(xb)
                        loss = criterion(out, yb)
                        ep_val += loss.item()

                avg_train = ep_train / max(1, len(train_loader))
                avg_val = ep_val / max(1, len(test_loader))
                train_losses.append(avg_train)
                val_losses.append(avg_val)
                sched.step(avg_val)

                if (epoch + 1) % 5 == 0:
                    print(
                        f"Epoch {epoch+1:3d} | train {avg_train:.6f} | val {avg_val:.6f}"
                    )

                if avg_val < best_val:
                    best_val = avg_val
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            train_secs = (datetime.now() - start_time).total_seconds()
            print(f"WaveNet training complete in {train_secs:.2f}s")

            # Predictions (inverse transform to original units)
            model.eval()
            y_pred_list, y_true_list = [], []
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb = WaveNetTrainer._reshape_if_flat(xb, seq_len, num_vars).to(
                        device
                    )
                    out = model(xb)  # [B,F]
                    # inverse using the dataset scaler
                    y_pred_list.append(test_set.inverse_transform(out).cpu().numpy())
                    y_true_list.append(test_set.inverse_transform(yb).numpy())

            y_pred = np.vstack(y_pred_list)  # [N,F]
            y_true = np.vstack(y_true_list)  # [N,F]

            F = y_true.shape[1]
            targets = (
                STANDARD_TARGET_COLUMNS[:F]
                if len(STANDARD_TARGET_COLUMNS) >= F
                else [f"var_{i}" for i in range(F)]
            )
            y_true_dict = {targets[i]: y_true[:, i] for i in range(F)}
            y_pred_dict = {targets[i]: y_pred[:, i] for i in range(F)}

            metadata = {
                "model_type": "WaveNet",
                "direction": direction,
                "training_time_sec": train_secs,
                "epochs_trained": len(train_losses),
                "final_train_loss": train_losses[-1] if train_losses else None,
                "final_val_loss": val_losses[-1] if val_losses else None,
                "sequence_length": seq_len,
                "num_features": num_vars,
                "parameters": sum(p.numel() for p in model.parameters()),
            }

            results = evaluator.evaluate_model_predictions(
                "WaveNet", y_true_dict, y_pred_dict, metadata
            )
            return results, y_true_dict, y_pred_dict

        except Exception as e:
            print(f"WaveNet training failed: {e}")
            import traceback

            traceback.print_exc()
            return None, {}, {}

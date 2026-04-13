#!/usr/bin/env python3
"""
LSTM (Long Short-Term Memory) Model Implementation
- Works with flat windows [B, D] or sequence windows [B, T, F]
- If flat, D must be T*F; F inferred from STANDARD_TARGET_COLUMNS (fallback 7)
- Predicts next-step [B, F]
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from typing import Dict, Tuple, Any
import warnings
import yaml 
import time

warnings.filterwarnings("ignore")

from ..evaluation.standardized_evaluation import STANDARD_TARGET_COLUMNS


# ----------------------------- Model -----------------------------


class SimpleLSTM(nn.Module):
    """
    LSTM encoder for next-step forecasting.
    Input:  x [B, T, F]
    Output: yhat [B, F]
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if output_size is None:
            output_size = input_size

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        if x.dim() != 3:
            raise ValueError(f"Expected x of shape [B,T,F], got {tuple(x.shape)}")

        B = x.size(0)
        h0 = x.new_zeros(self.num_layers, B, self.hidden_size)
        c0 = x.new_zeros(self.num_layers, B, self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))  # [B, T, H]
        last = out[:, -1, :]  # [B, H]
        yhat = self.fc(self.dropout(last))  # [B, F]
        return yhat


# ----------------------------- Trainer -----------------------------


class LSTMTrainer:
    """LSTM model trainer."""

    @staticmethod
    def _collect_files(csv_dir: str, direction: str) -> list[str]:
        direction = direction.upper()
        files = [
            os.path.join(csv_dir, f)
            for f in os.listdir(csv_dir)
            if f.endswith(f"_{direction}.csv")
        ]
        if not files:
            raise FileNotFoundError(
                f"No CSVs found in {csv_dir} for direction '{direction}'."
            )
        return sorted(files)

    @staticmethod
    def _infer_dims_from_loader(loader: DataLoader) -> Tuple[int, int, bool]:
        """
        Peek one batch to get (seq_len, num_vars, is_flat).
        Accepts xb as [B,T,F] or [B,D] where D=T*F.
        """
        xb, yb = next(iter(loader))

        if xb.dim() == 3:
            # [B, T, F]
            _, T, F = xb.shape
            if yb.dim() != 2 or yb.shape[1] != F:
                raise ValueError(
                    f"Expected yb to be [B,F] with F={F}, got {tuple(yb.shape)}"
                )
            return T, F, False

        if xb.dim() == 2:
            # [B, D] -> infer F from STANDARD_TARGET_COLUMNS (fallback 7)
            _, D = xb.shape
            F = len(STANDARD_TARGET_COLUMNS) if len(STANDARD_TARGET_COLUMNS) > 0 else 7
            if D % F != 0:
                raise ValueError(
                    f"Flat input width D={D} not divisible by inferred F={F}; "
                    f"cannot reshape to [B,T,F]. Adjust window_size or targets."
                )
            T = D // F
            if yb.dim() != 2 or yb.shape[1] != F:
                raise ValueError(
                    f"Expected yb to be [B,F] with F={F}, got {tuple(yb.shape)}"
                )
            return T, F, True

        raise ValueError(f"Unsupported xb shape {tuple(xb.shape)}")

    @staticmethod
    def _reshape_if_flat(xb: torch.Tensor, seq_len: int, num_vars: int) -> torch.Tensor:
        # If flat [B, D], reshape to [B, T, F]; else return as-is
        if xb.dim() == 2:
            B, D = xb.shape
            if D != seq_len * num_vars:
                raise ValueError(f"Cannot reshape [B,{D}] to [B,{seq_len},{num_vars}]")
            return xb.view(B, seq_len, num_vars)
        return xb

    @staticmethod
    def train_and_evaluate(
        train_set, test_set, evaluator, results_dir,
        direction="UL", config: dict | None = None
    ) -> Tuple[Dict[str, Any], Dict, Dict]:
        """Train and evaluate LSTM model."""
        print("\n" + "=" * 20 + " LSTM MODEL " + "=" * 20)

        try:
            # ---------------- Config ----------------
            cfg = config or {}
            batch_size  = cfg.get("batch_size", 64)
            max_epochs  = cfg.get("max_epochs", 100)
            patience    = cfg.get("patience", 10)

            # ---------------- Data ----------------
            train_loader = DataLoader(train_set, batch_size=batch_size,
                                      shuffle=True, drop_last=True)
            test_loader  = DataLoader(test_set, batch_size=batch_size,
                                      shuffle=False, drop_last=False)

            seq_len, num_vars, is_flat = LSTMTrainer._infer_dims_from_loader(train_loader)

            # ---------------- Model ----------------
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = SimpleLSTM(
                input_size=num_vars,
                hidden_size=cfg.get("hidden_size", 64),
                num_layers=cfg.get("num_layers", 2),
                output_size=num_vars,
                dropout=cfg.get("dropout", 0.2),
            ).to(device)

            # ---------------- Optimizer ----------------
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg.get("lr", 1e-3),
                weight_decay=cfg.get("weight_decay", 1e-5),
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=5, factor=0.5
            )

            # ---------------- Train loop ----------------
            print("Training LSTM...")
            start_time = datetime.now()
            best_val = float("inf")
            no_improve = 0
            train_losses, val_losses = [], []

            training_time = 0
            testing_time = 0

            for epoch in range(max_epochs):

                t0 = time.time()
                model.train()
                epoch_train_loss = 0.0

                for batch_X, batch_y in train_loader:
                    batch_X = LSTMTrainer._reshape_if_flat(batch_X, seq_len, num_vars)
                    batch_X = batch_X.to(device)  # [B,T,F]
                    batch_y = batch_y.to(device)  # [B,F]

                    optimizer.zero_grad()
                    outputs = model(batch_X)  # [B,F]
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    epoch_train_loss += loss.item()

                training_time += time.time() - t0

                t0 = time.time()
                # validation
                model.eval()
                epoch_val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X = LSTMTrainer._reshape_if_flat(
                            batch_X, seq_len, num_vars
                        )
                        batch_X = batch_X.to(device)
                        batch_y = batch_y.to(device)
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        epoch_val_loss += loss.item()

                testing_time += time.time() - t0

                avg_train = epoch_train_loss / max(1, len(train_loader))
                avg_val = epoch_val_loss / max(1, len(test_loader))
                train_losses.append(avg_train)
                val_losses.append(avg_val)
                scheduler.step(avg_val)

                if (epoch + 1) % 5 == 0:
                    print(
                        f"Epoch {epoch+1:3d} | train {avg_train:.6f} | val {avg_val:.6f}"
                    )

                if avg_val < best_val:
                    best_val = avg_val
                    no_improve = 0
                else:
                    no_improve += 1
                    # if no_improve >= patience:
                    #     print(f"Early stopping at epoch {epoch+1}")
                    #     break

            print(f"LSTM training completed in {training_time:.2f}s")

            # ---------- predictions ----------
            model.eval()
            y_pred_list, y_true_list = [], []
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X = LSTMTrainer._reshape_if_flat(batch_X, seq_len, num_vars)
                    batch_X = batch_X.to(device)
                    outputs = model(batch_X)  # [B,F]
                    y_pred_list.append(
                        test_set.inverse_transform(outputs).cpu().numpy()
                    )
                    y_true_list.append(test_set.inverse_transform(batch_y).numpy())

            y_pred_array = np.vstack(y_pred_list)  # [N,F]
            y_true_array = np.vstack(y_true_list)  # [N,F]

            # Build dicts for evaluator (match STANDARD_TARGET_COLUMNS length if possible)
            F = y_true_array.shape[1]
            targets = (
                STANDARD_TARGET_COLUMNS[:F]
                if len(STANDARD_TARGET_COLUMNS) >= F
                else [f"var_{i}" for i in range(F)]
            )
            y_true_dict = {targets[i]: y_true_array[:, i] for i in range(F)}
            y_pred_dict = {targets[i]: y_pred_array[:, i] for i in range(F)}

            # ---------- metadata & evaluate ----------
            training_metadata = {
                "model_type": "LSTM",
                "direction": direction,
                "training_time_sec": training_time,
                "testing_time_sec": testing_time,
                "epochs_trained": len(train_losses),
                "final_train_loss": train_losses[-1] if train_losses else None,
                "final_val_loss": val_losses[-1] if val_losses else None,
                "train_batches": len(train_loader),
                "test_batches": len(test_loader),
                "sequence_length": seq_len,
                "num_features": num_vars,
                "hidden_size": 64,
                "num_layers": 2,
                "parameters": sum(p.numel() for p in model.parameters()),
            }

            results = evaluator.evaluate_model_predictions(
                "LSTM", y_true_dict, y_pred_dict, training_metadata
            )
            return results, y_true_dict, y_pred_dict, training_metadata

        except Exception as e:
            print(f"LSTM training failed: {str(e)}")
            import traceback

            traceback.print_exc()
            return None, {}, {}, None

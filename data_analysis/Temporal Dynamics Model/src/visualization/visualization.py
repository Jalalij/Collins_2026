"""
Visualization module for time series model analysis.
This module contains plotting and visualization functions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from ..evaluation.standardized_evaluation import STANDARD_TARGET_COLUMNS


class ModelVisualizer:
    """Handles all visualization for model analysis."""

    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        os.makedirs(f"{results_dir}/plots", exist_ok=True)

    def create_var_plots(self, y_true_dict: Dict, y_pred_dict: Dict, results: Dict):
        """Create VAR-specific plots."""
        self._create_standard_plots(y_true_dict, y_pred_dict, results, "VAR")

    def create_lstm_plots(
        self,
        y_true_dict: Dict,
        y_pred_dict: Dict,
        results: Dict,
        train_losses: List,
        val_losses: List,
    ):
        """Create LSTM-specific plots including loss curves."""
        self._create_standard_plots(y_true_dict, y_pred_dict, results, "LSTM")

        # Training/validation loss curves
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Training Loss", color="blue")
        plt.plot(val_losses, label="Validation Loss", color="red")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("LSTM Training & Validation Loss Curves")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(train_losses, label="Training Loss", color="blue")
        plt.plot(val_losses, label="Validation Loss", color="red")
        plt.xlabel("Epoch")
        plt.ylabel("Log Loss")
        plt.yscale("log")
        plt.title("LSTM Loss Curves (Log Scale)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{self.results_dir}/plots/lstm_loss_curves.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def create_transformer_plots(
        self, y_true_dict: Dict, y_pred_dict: Dict, results: Dict
    ):
        """Create Transformer-specific plots."""
        self._create_standard_plots(y_true_dict, y_pred_dict, results, "Transformer")

    def create_varima_plots(self, y_true_dict: Dict, y_pred_dict: Dict, results: Dict):
        """Create VARIMA-specific plots."""
        self._create_standard_plots(y_true_dict, y_pred_dict, results, "VARIMA")

    def _create_standard_plots(
        self, y_true_dict: Dict, y_pred_dict: Dict, results: Dict, model_name: str
    ):
        """Create standard plots for any model."""

        # 1. Predicted vs Actual scatter plots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for i, target in enumerate(STANDARD_TARGET_COLUMNS):
            if i >= len(axes):
                break

            ax = axes[i]
            if target in y_true_dict and target in y_pred_dict:
                y_true = y_true_dict[target]
                y_pred = y_pred_dict[target]

                ax.scatter(y_true, y_pred, alpha=0.5, s=10)

                # Perfect prediction line
                min_val = min(np.min(y_true), np.min(y_pred))
                max_val = max(np.max(y_true), np.max(y_pred))
                ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8)

                # Calculate R² for title
                if "metrics" in results and target in results["metrics"]:
                    r2 = results["metrics"][target]["r2"]
                    ax.set_title(f"{target}\nR² = {r2:.3f}")
                else:
                    ax.set_title(target)

                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.grid(True, alpha=0.3)

        # Remove unused subplots
        for i in range(len(STANDARD_TARGET_COLUMNS), len(axes)):
            fig.delaxes(axes[i])

        plt.suptitle(f"{model_name} Model: Predicted vs Actual Values", fontsize=16)
        plt.tight_layout()
        plt.savefig(
            f"{self.results_dir}/plots/{model_name.lower()}_predicted_vs_actual.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # 2. Time series plots (first 100 points)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for i, target in enumerate(STANDARD_TARGET_COLUMNS[: len(axes)]):
            ax = axes[i]
            if target in y_true_dict and target in y_pred_dict:
                y_true = y_true_dict[target][:100]  # First 100 points
                y_pred = y_pred_dict[target][:100]

                ax.plot(y_true, label="Actual", color="blue", linewidth=2)
                ax.plot(
                    y_pred, label="Predicted", color="red", linewidth=1.5, alpha=0.8
                )
                ax.set_title(target)
                ax.set_xlabel("Time Steps")
                ax.set_ylabel("Value")
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.suptitle(
            f"{model_name} Model: Time Series Predictions (First 100 Steps)",
            fontsize=16,
        )
        plt.tight_layout()
        plt.savefig(
            f"{self.results_dir}/plots/{model_name.lower()}_time_series.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # 3. Residual plots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for i, target in enumerate(STANDARD_TARGET_COLUMNS[: len(axes)]):
            ax = axes[i]
            if target in y_true_dict and target in y_pred_dict:
                y_true = y_true_dict[target]
                y_pred = y_pred_dict[target]
                residuals = y_true - y_pred

                ax.scatter(range(len(residuals)), residuals, alpha=0.5, s=10)
                ax.axhline(y=0, color="r", linestyle="--", alpha=0.8)
                ax.set_title(f"{target} - Residuals")
                ax.set_xlabel("Sample Index")
                ax.set_ylabel("Residual (Actual - Predicted)")
                ax.grid(True, alpha=0.3)

        plt.suptitle(f"{model_name} Model: Residual Analysis", fontsize=16)
        plt.tight_layout()
        plt.savefig(
            f"{self.results_dir}/plots/{model_name.lower()}_residuals.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def create_visual_comparison_table(self, table_data: List, headers: List):
        """Create visual comparison table."""
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis("tight")
        ax.axis("off")

        # Create table
        table = ax.table(
            cellText=table_data, colLabels=headers, cellLoc="center", loc="center"
        )

        # Format table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.8)

        # Color headers
        for i in range(len(headers)):
            table[(0, i)].set_facecolor("#4CAF50")
            table[(0, i)].set_text_props(weight="bold", color="white")

        # Color rows alternately
        colors = ["#f0f0f0", "#ffffff"]
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                table[(i, j)].set_facecolor(colors[i % 2])

        plt.title(
            "Time Series Model Performance Comparison\n(All metrics calculated on TEST dataset only)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        plt.savefig(
            f"{self.results_dir}/plots/model_comparison_table.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def create_wavenet_plots(self, y_true_dict, y_pred_dict, results):
        """Create WaveNet-specific plots."""
        print("Creating WaveNet visualization plots...")

        # Standard plots
        self.create_prediction_plots(y_true_dict, y_pred_dict, "WaveNet")
        self.create_residual_plots(y_true_dict, y_pred_dict, "WaveNet")

        # Loss curves if available
        if (
            "training_metadata" in results
            and "loss_curves" in results["training_metadata"]
        ):
            self.create_loss_curves(
                results["training_metadata"]["loss_curves"]["train_losses"],
                results["training_metadata"]["loss_curves"]["val_losses"],
                "WaveNet",
            )

    def create_gp_plots(self, y_true_dict, y_pred_dict, results):
        """Create Gaussian Process-specific plots."""
        print("Creating Gaussian Process visualization plots...")

        # Standard plots
        self.create_prediction_plots(y_true_dict, y_pred_dict, "GaussianProcess")
        self.create_residual_plots(y_true_dict, y_pred_dict, "GaussianProcess")

        # GP-specific uncertainty plot (if uncertainty data is available)
        # For now, use standard plots

    def create_autoencoder_plots(self, y_true_dict, y_pred_dict, results, ae_type):
        """Create Predictive Autoencoder-specific plots."""
        model_name = f"PredictiveAutoencoder_{ae_type}"
        print(f"Creating {model_name} visualization plots...")

        # Standard plots
        self.create_prediction_plots(y_true_dict, y_pred_dict, model_name)
        self.create_residual_plots(y_true_dict, y_pred_dict, model_name)

        # Loss curves if available
        if (
            "training_metadata" in results
            and "loss_curves" in results["training_metadata"]
        ):
            self.create_loss_curves(
                results["training_metadata"]["loss_curves"]["train_losses"],
                results["training_metadata"]["loss_curves"]["val_losses"],
                model_name,
            )

    def create_loss_curves(self, train_losses, val_losses, model_name):
        """Create training and validation loss curves."""
        plt.figure(figsize=(10, 6))

        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
        plt.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{model_name} - Training and Validation Loss Curves")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(
            f"{self.results_dir}/plots/{model_name.lower()}_loss_curves.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

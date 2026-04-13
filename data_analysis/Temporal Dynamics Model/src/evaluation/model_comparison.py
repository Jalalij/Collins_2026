import numpy as np

"""
Model comparison module for time series analysis.
This module handles comparison tables and ranking analysis.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any
from .standardized_evaluation import STANDARD_TARGET_COLUMNS


class ModelComparison:
    """Handles model comparison and ranking analysis."""

    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.visualizer = None  # Initialize lazily to avoid circular imports

    def _get_visualizer(self):
        """Lazy initialization of visualizer to avoid circular imports."""
        if self.visualizer is None:
            from src.visualization.visualization import ModelVisualizer

            self.visualizer = ModelVisualizer(self.results_dir)
        return self.visualizer

    def create_comparison_table(
        self, results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create a comprehensive comparison table of all model results."""

        comparison_data = {
            "models": [],
            "metrics": {},
            "rankings": {},
            "best_model": None,
            "timestamp": datetime.now().isoformat(),
        }

        # Extract metrics for each model
        for model_name, model_results in results.items():
            if "test_metrics" not in model_results:
                continue

            model_metrics = model_results["test_metrics"]
            comparison_data["models"].append(model_name)

            # Initialize metric collections if needed
            for metric_name, metric_value in model_metrics.items():
                if metric_name not in comparison_data["metrics"]:
                    comparison_data["metrics"][metric_name] = {}
                comparison_data["metrics"][metric_name][model_name] = metric_value

        # Create rankings for each metric (lower is better for RMSE, MAE, MAPE)
        ranking_metrics = ["rmse", "mae", "mape"]
        ascending_metrics = ["r2"]  # Higher is better

        for metric_name, metric_values in comparison_data["metrics"].items():
            if metric_values:
                # Sort models by metric value
                if metric_name.lower() in ranking_metrics:
                    # Lower is better
                    sorted_models = sorted(metric_values.items(), key=lambda x: x[1])
                elif metric_name.lower() in ascending_metrics:
                    # Higher is better
                    sorted_models = sorted(
                        metric_values.items(), key=lambda x: x[1], reverse=True
                    )
                else:
                    # Default: lower is better
                    sorted_models = sorted(metric_values.items(), key=lambda x: x[1])

                comparison_data["rankings"][metric_name] = [
                    model for model, _ in sorted_models
                ]

        # Determine overall best model (based on RMSE if available)
        if (
            "rmse" in comparison_data["rankings"]
            and comparison_data["rankings"]["rmse"]
        ):
            comparison_data["best_model"] = comparison_data["rankings"]["rmse"][0]
        elif comparison_data["models"]:
            comparison_data["best_model"] = comparison_data["models"][0]

        return comparison_data

    def save_comparison_results(
        self, comparison_data: Dict[str, Any], filename: str = "model_comparison.json"
    ):
        """Save comparison results to JSON file."""

        os.makedirs(self.results_dir, exist_ok=True)
        filepath = os.path.join(self.results_dir, filename)

        with open(filepath, "w") as f:
            json.dump(comparison_data, f, indent=2, default=str)

        print(f"Model comparison saved to: {filepath}")

    def print_comparison_table(self, comparison_data: Dict[str, Any]):
        """Print a formatted comparison table."""

        print("\n" + "=" * 80)
        print("MODEL COMPARISON RESULTS")
        print("=" * 80)

        if not comparison_data["models"]:
            print("No model results to compare.")
            return

        # Print metrics table
        print(f"\nMETRICS COMPARISON ({len(comparison_data['models'])} models)")
        print("-" * 80)

        # Header
        model_names = comparison_data["models"]
        header = f"{'Metric':<15}"
        for model in model_names:
            header += f"{model:<15}"
        print(header)
        print("-" * 80)

        # Metrics rows
        for metric_name, metric_values in comparison_data["metrics"].items():
            row = f"{metric_name:<15}"
            for model in model_names:
                value = metric_values.get(model, "N/A")
                if isinstance(value, (int, float)):
                    row += f"{value:<15.4f}"
                else:
                    row += f"{str(value):<15}"
            print(row)

        # Print rankings
        print(f"\nRANKINGS (1=best)")
        print("-" * 80)

        for metric_name, ranked_models in comparison_data["rankings"].items():
            print(f"\n{metric_name.upper()} ranking:")
            for i, model in enumerate(ranked_models, 1):
                metric_value = comparison_data["metrics"][metric_name].get(model, "N/A")
                if isinstance(metric_value, (int, float)):
                    print(f"  {i}. {model:<25} ({metric_value:.4f})")
                else:
                    print(f"  {i}. {model:<25} ({metric_value})")

        # Print best model
        if comparison_data["best_model"]:
            print(f"\nOVERALL BEST MODEL: {comparison_data['best_model']}")

        print("=" * 80)

    def generate_comparison_plots(self, results: Dict[str, Dict[str, Any]]):
        """Generate comparison plots using visualizer."""

        visualizer = self._get_visualizer()

        # Create comparison plots
        visualizer.plot_model_comparison(results)
        visualizer.plot_metrics_comparison(results)

        print(f"Comparison plots saved to: {self.results_dir}/plots/")

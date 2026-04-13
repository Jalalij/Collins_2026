"""
Gaussian Process Model Implementation
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings("ignore")

from ..evaluation.standardized_evaluation import STANDARD_TARGET_COLUMNS


class GaussianProcessTrainer:
    """Gaussian Process model trainer using sklearn."""

    @staticmethod
    def train_and_evaluate(
        train_data, test_data, evaluator, results_dir
    ) -> Tuple[Dict[str, Any], Dict, Dict]:
        """Train and evaluate Gaussian Process model."""
        print("\n" + "=" * 20 + " GAUSSIAN PROCESS MODEL " + "=" * 20)

        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import (
                RBF,
                WhiteKernel,
                Matern,
                RationalQuadratic,
            )
            from sklearn.multioutput import MultiOutputRegressor
            from ..data.standardized_data_prep import prepare_standardized_data

            # Use standardized data preparation with smaller samples for GP (computationally expensive)
            data_prep = prepare_standardized_data(
                train_data, test_data, train_samples=5000, test_samples=1000
            )

            X_train = data_prep["train_scaled"]
            X_test = data_prep["test_scaled"]
            y_train = data_prep[
                "train_scaled"
            ]  # GP predicts same as input for time series
            y_test = data_prep["test_scaled"]

            print(
                f"Training Gaussian Process with {len(X_train):,} samples, {X_train.shape[1]} features"
            )
            print("Note: Using smaller dataset for GP due to computational complexity")

            # Define composite kernel for time series
            # RBF for smooth variations + Matern for less smooth patterns + WhiteKernel for noise
            length_scale_bounds = (1e-2, 1e2)
            kernel = (
                RBF(length_scale=1.0, length_scale_bounds=length_scale_bounds)
                + Matern(
                    length_scale=1.0, nu=2.5, length_scale_bounds=length_scale_bounds
                )
                + RationalQuadratic(
                    length_scale=1.0, alpha=1.0, length_scale_bounds=length_scale_bounds
                )
                + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1e1))
            )

            # Initialize GP with multi-output capability
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=2,  # Reduced for speed
                alpha=1e-6,  # Regularization
                normalize_y=True,
                copy_X_train=False,  # Save memory
            )

            # Use MultiOutputRegressor for multiple targets
            multi_gp = MultiOutputRegressor(gp, n_jobs=-1)

            print("Training Gaussian Process (this may take a while)...")
            start_time = datetime.now()

            # Train the model
            multi_gp.fit(X_train, y_train)

            training_time = (datetime.now() - start_time).total_seconds()
            print(f"GP training completed in {training_time:.2f} seconds")

            # Generate predictions with uncertainty
            print("Generating GP predictions...")
            y_pred = multi_gp.predict(X_test)
            y_std = None  # MultiOutputEstimator does not support return_std

            # Inverse transform predictions
            y_pred_original = data_prep["scaler"].inverse_transform(y_pred)
            y_test_original = data_prep["scaler"].inverse_transform(y_test)

            # Create prediction dictionaries for evaluator
            y_true_dict = {}
            y_pred_dict = {}

            for i, target in enumerate(STANDARD_TARGET_COLUMNS):
                y_true_dict[target] = y_test_original[:, i]
                y_pred_dict[target] = y_pred_original[:, i]

            # Calculate marginal likelihood for each estimator
            marginal_likelihoods = []
            for estimator in multi_gp.estimators_:
                try:
                    marginal_likelihoods.append(estimator.log_marginal_likelihood())
                except:
                    marginal_likelihoods.append(np.nan)

            # Training metadata
            training_metadata = {
                "model_type": "GaussianProcess",
                "training_time": training_time,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "kernel": str(kernel),
                "marginal_likelihood": np.nanmean(marginal_likelihoods),
                "uncertainty_estimates": True,
                "mean_prediction_std": np.mean(y_std),
                "std_prediction_std": np.std(y_std),
                "hyperparameters": {
                    "kernel_params": [
                        str(est.kernel_) for est in multi_gp.estimators_[:3]
                    ]  # First 3 for brevity
                },
            }

            print(
                f"GP mean prediction uncertainty: {training_metadata['mean_prediction_std']:.4f}"
            )

            # Evaluate using standardized framework
            results = evaluator.evaluate_model_predictions(
                "GaussianProcess", y_true_dict, y_pred_dict, training_metadata
            )

            return results, y_true_dict, y_pred_dict

        except Exception as e:
            print(f"Gaussian Process training failed: {str(e)}")
            import traceback

            traceback.print_exc()
            return None, {}, {}

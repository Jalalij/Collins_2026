#!/usr/bin/env python3
"""
Model Training Functions

This module provides training functions for various time series models.
"""

import numpy as np
import warnings

warnings.filterwarnings("ignore")


def train_var(data, maxlags=10):
    """
    Train a Vector Autoregression (VAR) model with robust settings.

    Args:
        data: numpy array of shape (n_samples, n_features)
        maxlags: maximum number of lags to consider

    Returns:
        Trained VAR model object
    """
    try:
        from statsmodels.tsa.vector_ar.var_model import VAR

        # Create VAR model
        model = VAR(data)

        # First try with AIC selection
        try:
            fitted_model = model.fit(maxlags=maxlags, ic="aic")
            if len(fitted_model.coefs) > 0:
                print(f"VAR fitted with {len(fitted_model.coefs)} lags (AIC selection)")
                return fitted_model
        except:
            pass

        # If AIC selection fails or gives 0 lags, try fixed lag approaches
        for fixed_lags in [1, 2, 3, min(5, maxlags)]:
            try:
                fitted_model = model.fit(maxlags=fixed_lags, ic=None)
                if len(fitted_model.coefs) > 0:
                    print(f"VAR fitted with fixed {fixed_lags} lags")
                    return fitted_model
            except Exception as e:
                print(f"Warning:  VAR with {fixed_lags} lags failed: {e}")
                continue

        # If all else fails, create mock model
        print("Warning:  All VAR fitting attempts failed, creating mock model")
        return MockVARModel(data)

    except ImportError:
        print("Warning:  statsmodels not available, creating mock VAR model")
        return MockVARModel(data)
    except Exception as e:
        print(f"Warning:  VAR training failed: {e}, creating mock model")
        return MockVARModel(data)


class MockVARModel:
    """Mock VAR model for when statsmodels is not available or fails."""

    def __init__(self, data):
        self.data = data
        self.n_features = data.shape[1] if len(data.shape) > 1 else 1

        # Calculate simple statistics for prediction
        self.mean_values = (
            np.mean(data, axis=0) if len(data.shape) > 1 else np.mean(data)
        )
        self.std_values = np.std(data, axis=0) if len(data.shape) > 1 else np.std(data)

        # Calculate simple AR(1) coefficients for each variable
        self.ar_coefs = []
        if len(data.shape) > 1:
            for i in range(data.shape[1]):
                series = data[:, i]
                if len(series) > 1:
                    # Simple AR(1): y_t = rho * y_{t-1} + noise
                    rho = np.corrcoef(series[:-1], series[1:])[0, 1]
                    self.ar_coefs.append(max(0.1, min(0.9, rho)))  # Keep stable
                else:
                    self.ar_coefs.append(0.5)
        else:
            rho = np.corrcoef(data[:-1], data[1:])[0, 1] if len(data) > 1 else 0.5
            self.ar_coefs.append(max(0.1, min(0.9, rho)))

        self.ar_coefs = np.array(self.ar_coefs)

    def forecast(self, context, steps=1):
        """
        Generate forecasts using simple AR(1) model.

        Args:
            context: Recent data points for context
            steps: Number of steps to forecast

        Returns:
            Forecasted values
        """
        np.random.seed(42)  # For reproducibility

        if len(context.shape) > 1:
            # Multiple features
            forecasts = []
            last_values = context[-1] if len(context) > 0 else self.mean_values

            for step in range(steps):
                # AR(1) forecast with small noise
                forecast = (
                    self.ar_coefs * last_values + (1 - self.ar_coefs) * self.mean_values
                )
                forecast += np.random.normal(0, self.std_values * 0.05, self.n_features)
                forecasts.append(forecast)
                last_values = forecast

            return np.array(forecasts)
        else:
            # Single feature
            forecasts = []
            last_value = context[-1] if len(context) > 0 else self.mean_values

            for step in range(steps):
                forecast = (
                    self.ar_coefs[0] * last_value
                    + (1 - self.ar_coefs[0]) * self.mean_values
                )
                forecast += np.random.normal(0, self.std_values * 0.05)
                forecasts.append(forecast)
                last_value = forecast

            return np.array(forecasts)

"""
Linear Regression Model for Time Series Analysis
===============================================

This module provides a comprehensive wrapper around scikit-learn's LinearRegression
for use in time series forecasting. It includes both a model class and a trainer
class to maintain consistency with other models in the framework.

Linear regression is often used as a baseline model for time series prediction
due to its simplicity, interpretability, and fast training. While not as sophisticated
as neural networks, it can capture linear trends and relationships effectively.

Key Features:
- Simple and interpretable baseline model
- Fast training and prediction
- Automatic data validation and reshaping
- Consistent interface with other models in the framework
- Access to model coefficients and statistics

Author: SatCom TimeSeries Analysis Framework
Date: 2024
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Optional, Union
import warnings


class LinearRegressionModel:
    """
    Linear Regression model wrapper for time series prediction.

    This class wraps scikit-learn's LinearRegression with additional functionality
    for time series analysis, including automatic data validation, consistent
    interfaces, and enhanced error handling.

    Linear Regression Model:
    The model fits a linear relationship between input features and target values:
    y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ

    Where:
    - y is the predicted target value
    - w₀ is the intercept (bias term)
    - w₁, w₂, ..., wₙ are the coefficients (weights)
    - x₁, x₂, ..., xₙ are the input features

    Advantages:
    - Fast training and prediction
    - Highly interpretable (can analyze coefficients)
    - No hyperparameter tuning required
    - Good baseline for comparison with complex models
    - Works well for linear trends and relationships

    Limitations:
    - Cannot capture non-linear relationships
    - Assumes linear relationship between features and target
    - Sensitive to outliers
    - May overfit with many features and few samples

    Attributes:
        model (LinearRegression): Underlying scikit-learn model
        is_fitted (bool): Whether the model has been trained

    Example:
        >>> model = LinearRegressionModel()
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
        >>> r2_score = model.score(X_test, y_test)
    """

    def __init__(self):
        """
        Initialize the Linear Regression model.

        Creates a new LinearRegression instance from scikit-learn with default
        parameters. The model is not fitted until the fit() method is called.

        Default Parameters:
        - fit_intercept=True: Include bias/intercept term
        - normalize=False: Do not normalize features (deprecated in newer sklearn)
        - copy_X=True: Copy input data to avoid modifying original
        - n_jobs=None: Use single thread for fitting
        """
        print("Initializing LinearRegressionModel...")

        # Create underlying scikit-learn model
        self.model = LinearRegression()

        # Track fitting state
        self.is_fitted = False

        print("LinearRegressionModel initialized successfully")
        print("  - Model type: scikit-learn LinearRegression")
        print("  - Fit intercept: True")
        print("  - Ready for training")

    def fit(
        self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]
    ) -> "LinearRegressionModel":
        """
        Fit the linear regression model to training data.

        This method trains the linear regression model by finding the optimal
        coefficients that minimize the sum of squared errors between predictions
        and actual values.

        Mathematical Foundation:
        The model minimizes the cost function:
        J(w) = (1/2m) * Σ(h_w(x_i) - y_i)²

        Where:
        - m is the number of training samples
        - h_w(x_i) is the prediction for sample i
        - y_i is the actual target value for sample i

        Args:
            X (Union[np.ndarray, list]): Training features
                                       Shape: (n_samples, n_features) or (n_samples,)
            y (Union[np.ndarray, list]): Training targets
                                       Shape: (n_samples,)

        Returns:
            LinearRegressionModel: Self reference for method chaining

        Raises:
            ValueError: If X and y have mismatched dimensions
            ValueError: If input contains NaN or infinite values

        Data Processing:
        1. Convert inputs to numpy arrays if needed
        2. Ensure X is 2D (reshape if needed for single feature)
        3. Validate data shapes and values
        4. Fit the underlying scikit-learn model
        5. Update fitted state

        Example:
            >>> X = np.array([[1], [2], [3], [4]])  # Single feature
            >>> y = np.array([2, 4, 6, 8])          # Linear relationship
            >>> model = LinearRegressionModel()
            >>> model.fit(X, y)                     # Should learn w₀=0, w₁=2
        """
        print(f"Fitting LinearRegressionModel...")

        # === STEP 1: Data Type Conversion ===
        print(f"  STEP 1: Converting input data to numpy arrays...")

        # Convert to numpy arrays if needed
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y

        print(f"    Original X shape: {X.shape}")
        print(f"    Original y shape: {y.shape}")

        # === STEP 2: Data Shape Validation ===
        print(f"  STEP 2: Validating and reshaping data...")

        # Ensure X is 2D (sklearn requirement)
        if X.ndim == 1:
            print(f"    Reshaping X from 1D to 2D (single feature case)")
            X = X.reshape(-1, 1)

        print(f"    Final X shape: {X.shape}")
        print(f"    Final y shape: {y.shape}")

        # Validate shapes
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have same number of samples: {len(X)} vs {len(y)}"
            )

        if len(X) == 0:
            raise ValueError("Cannot fit model with empty data")

        # === STEP 3: Data Quality Checks ===
        print(f"  STEP 3: Checking data quality...")

        # Check for NaN or infinite values
        if np.any(~np.isfinite(X)):
            nan_count = np.sum(~np.isfinite(X))
            warnings.warn(f"X contains {nan_count} NaN/infinite values")

        if np.any(~np.isfinite(y)):
            nan_count = np.sum(~np.isfinite(y))
            warnings.warn(f"y contains {nan_count} NaN/infinite values")

        print(f"    Training samples: {len(X)}")
        print(f"    Features: {X.shape[1]}")
        print(f"    Target range: [{np.min(y):.3f}, {np.max(y):.3f}]")

        # === STEP 4: Model Fitting ===
        print(f"  STEP 4: Training linear regression model...")

        try:
            # Fit the underlying scikit-learn model
            self.model.fit(X, y)

            # Update fitted state
            self.is_fitted = True

            print(f"    Model fitted successfully!")

            # Display learned parameters
            if hasattr(self.model, "coef_") and hasattr(self.model, "intercept_"):
                print(f"    Learned parameters:")
                print(f"      Intercept (w₀): {self.model.intercept_:.6f}")

                if X.shape[1] == 1:
                    print(f"      Coefficient (w₁): {self.model.coef_[0]:.6f}")
                else:
                    print(f"      Coefficients: {len(self.model.coef_)} features")
                    for i, coef in enumerate(self.model.coef_):
                        print(f"        w_{i+1}: {coef:.6f}")

            # Calculate training R² score for reference
            train_r2 = self.model.score(X, y)
            print(f"    Training R² score: {train_r2:.6f}")

        except Exception as e:
            print(f"    ERROR during model fitting: {e}")
            raise

        print(f"LinearRegressionModel fitting complete!")

        return self

    def predict(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Make predictions using the fitted linear regression model.

        This method applies the learned linear function to input features
        to generate predictions:
        ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ

        Args:
            X (Union[np.ndarray, list]): Input features for prediction
                                       Shape: (n_samples, n_features) or (n_samples,)

        Returns:
            np.ndarray: Predictions for input samples
                       Shape: (n_samples,)

        Raises:
            ValueError: If model has not been fitted yet
            ValueError: If input has wrong number of features

        Data Processing:
        1. Check if model is fitted
        2. Convert input to numpy array
        3. Ensure proper 2D shape
        4. Validate feature count matches training data
        5. Generate predictions

        Example:
            >>> model = LinearRegressionModel()
            >>> model.fit(X_train, y_train)
            >>> predictions = model.predict(X_test)
            >>> print(f"Predictions: {predictions}")
        """
        print(f"Making predictions with LinearRegressionModel...")

        # === STEP 1: Model State Validation ===
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        print(f"  Model is fitted and ready for prediction")

        # === STEP 2: Data Preprocessing ===
        print(f"  STEP 1: Preprocessing input data...")

        # Convert to numpy array if needed
        X = np.array(X) if not isinstance(X, np.ndarray) else X

        print(f"    Input shape: {X.shape}")

        # Ensure X is 2D (sklearn requirement)
        if X.ndim == 1:
            print(f"    Reshaping input from 1D to 2D")
            X = X.reshape(-1, 1)

        print(f"    Final input shape: {X.shape}")
        print(f"    Prediction samples: {len(X)}")

        # === STEP 3: Feature Validation ===
        print(f"  STEP 2: Validating feature consistency...")

        # Check if number of features matches training data
        expected_features = len(self.model.coef_)
        actual_features = X.shape[1]

        if actual_features != expected_features:
            raise ValueError(
                f"Input has {actual_features} features, but model was trained with {expected_features} features"
            )

        print(f"    Feature count validation passed: {actual_features} features")

        # === STEP 4: Generate Predictions ===
        print(f"  STEP 3: Generating predictions...")

        try:
            predictions = self.model.predict(X)

            print(f"    Predictions generated successfully")
            print(
                f"    Prediction range: [{np.min(predictions):.3f}, {np.max(predictions):.3f}]"
            )
            print(f"    Prediction mean: {np.mean(predictions):.3f}")

            return predictions

        except Exception as e:
            print(f"    ERROR during prediction: {e}")
            raise

    def score(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> float:
        """
        Calculate the coefficient of determination (R²) score for the model.

        R² represents the proportion of variance in the target variable that
        is predictable from the input features. It provides a measure of
        how well the linear model explains the data.

        Formula: R² = 1 - (SS_res / SS_tot)
        Where:
        - SS_res = Σ(y_true - y_pred)²  [Residual sum of squares]
        - SS_tot = Σ(y_true - y_mean)²  [Total sum of squares]

        Args:
            X (Union[np.ndarray, list]): Test features
                                       Shape: (n_samples, n_features)
            y (Union[np.ndarray, list]): True target values
                                       Shape: (n_samples,)

        Returns:
            float: R² score

        Interpretation:
            - R² = 1.0: Perfect prediction
            - R² = 0.0: Model performs as well as predicting the mean
            - R² < 0.0: Model performs worse than predicting the mean
            - Higher values indicate better performance

        Raises:
            ValueError: If model has not been fitted yet
            ValueError: If X and y have mismatched dimensions
        """
        print(f"Calculating R² score for LinearRegressionModel...")

        # === STEP 1: Model State Validation ===
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")

        print(f"  Model is fitted and ready for scoring")

        # === STEP 2: Data Preprocessing ===
        print(f"  STEP 1: Preprocessing input data...")

        # Convert to numpy arrays if needed
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y

        print(f"    X shape: {X.shape}")
        print(f"    y shape: {y.shape}")

        # Ensure X is 2D
        if X.ndim == 1:
            print(f"    Reshaping X from 1D to 2D")
            X = X.reshape(-1, 1)

        # Validate shapes
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have same number of samples: {len(X)} vs {len(y)}"
            )

        print(f"    Evaluation samples: {len(X)}")

        # === STEP 3: Calculate R² Score ===
        print(f"  STEP 2: Calculating R² score...")

        try:
            r2_score = self.model.score(X, y)

            print(f"    R² score calculated successfully: {r2_score:.6f}")

            # Provide interpretation
            if r2_score >= 0.9:
                interpretation = "Excellent fit"
            elif r2_score >= 0.7:
                interpretation = "Good fit"
            elif r2_score >= 0.5:
                interpretation = "Moderate fit"
            elif r2_score >= 0.0:
                interpretation = "Poor fit"
            else:
                interpretation = "Very poor fit (worse than predicting mean)"

            print(f"    Interpretation: {interpretation}")

            return r2_score

        except Exception as e:
            print(f"    ERROR during scoring: {e}")
            raise

    @property
    def coef_(self) -> np.ndarray:
        """
        Get the learned coefficients (weights) of the linear model.

        The coefficients represent the linear relationship between each
        input feature and the target variable. They indicate how much
        the target value is expected to change when the corresponding
        feature increases by one unit.

        Returns:
            np.ndarray: Model coefficients [w₁, w₂, ..., wₙ]
                       Shape: (n_features,)

        Raises:
            ValueError: If model has not been fitted yet

        Interpretation:
            - Positive coefficient: Feature positively correlated with target
            - Negative coefficient: Feature negatively correlated with target
            - Zero coefficient: Feature has no linear relationship with target
            - Larger magnitude: Stronger influence on target

        Example:
            >>> model.fit(X_train, y_train)
            >>> coefficients = model.coef_
            >>> print(f"Feature influences: {coefficients}")
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to access coefficients")

        return self.model.coef_

    @property
    def intercept_(self) -> float:
        """
        Get the intercept (bias term) of the linear model.

        The intercept represents the expected target value when all
        input features are zero. It's the y-intercept of the linear
        function.

        Returns:
            float: Model intercept (w₀)

        Raises:
            ValueError: If model has not been fitted yet

        Interpretation:
            - The baseline prediction when all features are zero
            - Shifts the entire linear function up or down
            - Can be positive, negative, or zero

        Example:
            >>> model.fit(X_train, y_train)
            >>> intercept = model.intercept_
            >>> print(f"Baseline prediction: {intercept}")
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to access intercept")

        return self.model.intercept_


class LinearRegressionTrainer:
    """
    Trainer class for Linear Regression model to maintain consistency with other models.

    This class provides a consistent training interface that matches other model
    trainers in the framework (like neural network trainers). It's particularly
    useful when you want to use linear regression as part of a larger pipeline
    or comparison study.

    The trainer pattern allows for:
    - Consistent interface across different model types
    - Easy integration with automated model comparison systems
    - Potential extension for more sophisticated training procedures
    - Compatibility with existing training pipelines

    Attributes:
        model (LinearRegressionModel): The trained model instance

    Example:
        >>> trainer = LinearRegressionTrainer()
        >>> trained_model = trainer.train(X_train, y_train, X_val, y_val)
        >>> predictions = trained_model.predict(X_test)
    """

    def __init__(self):
        """
        Initialize the LinearRegressionTrainer.

        Creates a new trainer instance that can be used to train linear
        regression models. The trainer starts with no trained model.
        """
        print("Initializing LinearRegressionTrainer...")

        # Model will be created during training
        self.model = None

        print("LinearRegressionTrainer initialized successfully")
        print("  - Ready to train LinearRegressionModel")
        print("  - Consistent interface with other trainers")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> LinearRegressionModel:
        """
        Train a linear regression model using the provided data.

        This method creates and trains a LinearRegressionModel instance using
        the training data. Unlike neural networks, linear regression doesn't
        require validation data for early stopping, so X_val and y_val are
        accepted for interface consistency but not used in training.

        Args:
            X_train (np.ndarray): Training features
                                Shape: (n_samples, n_features)
            y_train (np.ndarray): Training targets
                                Shape: (n_samples,)
            X_val (Optional[np.ndarray]): Validation features (unused for linear regression)
                                        Shape: (n_val_samples, n_features)
            y_val (Optional[np.ndarray]): Validation targets (unused for linear regression)
                                        Shape: (n_val_samples,)

        Returns:
            LinearRegressionModel: Trained linear regression model

        Note:
            Linear regression has a closed-form solution and doesn't require
            iterative training or validation data for early stopping. The
            validation parameters are included for interface consistency
            with other trainers but are not used in the actual training process.

        Training Process:
        1. Create new LinearRegressionModel instance
        2. Fit the model on training data
        3. Optionally evaluate on validation data for reporting
        4. Return the trained model

        Example:
            >>> trainer = LinearRegressionTrainer()
            >>> model = trainer.train(X_train, y_train)
            >>> print(f"Training R²: {model.score(X_train, y_train)}")
        """
        print(f"\n=== TRAINING LINEAR REGRESSION MODEL ===")

        # === STEP 1: Data Validation ===
        print(f"STEP 1: Validating training data...")

        print(f"  Training data shapes:")
        print(f"    X_train: {X_train.shape}")
        print(f"    y_train: {y_train.shape}")

        if X_val is not None and y_val is not None:
            print(f"  Validation data provided (will be used for evaluation only):")
            print(f"    X_val: {X_val.shape}")
            print(f"    y_val: {y_val.shape}")
        else:
            print(f"  No validation data provided")

        # === STEP 2: Model Creation and Training ===
        print(f"STEP 2: Creating and training LinearRegressionModel...")

        # Create new model instance
        self.model = LinearRegressionModel()

        # Train the model on training data
        self.model.fit(X_train, y_train)

        # === STEP 3: Training Evaluation ===
        print(f"STEP 3: Evaluating training performance...")

        # Calculate training performance
        train_r2 = self.model.score(X_train, y_train)
        print(f"  Training R² score: {train_r2:.6f}")

        # If validation data is provided, evaluate on it too
        if X_val is not None and y_val is not None:
            print(f"  Evaluating on validation data...")
            try:
                val_r2 = self.model.score(X_val, y_val)
                print(f"  Validation R² score: {val_r2:.6f}")

                # Check for significant performance difference
                r2_diff = train_r2 - val_r2
                if r2_diff > 0.1:
                    print(
                        f"  WARNING: Large train-validation gap ({r2_diff:.3f}) may indicate overfitting"
                    )
                elif r2_diff < -0.05:
                    print(
                        f"  NOTE: Validation performance better than training (difference: {r2_diff:.3f})"
                    )
                else:
                    print(
                        f"  Good train-validation consistency (difference: {r2_diff:.3f})"
                    )

            except Exception as e:
                print(f"  WARNING: Could not evaluate on validation data: {e}")

        # === STEP 4: Model Summary ===
        print(f"STEP 4: Training summary...")

        print(f"  Model parameters:")
        print(f"    Intercept: {self.model.intercept_:.6f}")

        if X_train.shape[1] == 1:
            print(f"    Coefficient: {self.model.coef_[0]:.6f}")
            print(
                f"    Linear equation: y = {self.model.intercept_:.3f} + {self.model.coef_[0]:.3f} * x"
            )
        else:
            print(f"    Coefficients: {len(self.model.coef_)} features")
            print(
                f"    Linear equation: y = {self.model.intercept_:.3f} + Σ(coef_i * x_i)"
            )

        print(f"=== LINEAR REGRESSION TRAINING COMPLETE ===")
        print(f"Trained model ready for prediction and evaluation")

        return self.model

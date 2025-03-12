"""
Ensemble model for the Energy Prediction System.

This module contains the ensemble model implementation, which combines
predictions from multiple base models for improved accuracy.
"""

import logging
import numpy as np
from typing import List, Dict, Union, Tuple, Any, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Configure logging
logger = logging.getLogger(__name__)


class EnergyPredictionEnsemble:
    """
    Ensemble model that combines predictions from multiple base models.

    This class provides methods to combine predictions from different models
    using weighted averaging, and to optimize the weights for best performance.
    """

    def __init__(self, models: List[Any], weights: Optional[List[float]] = None):
        """
        Initialize the ensemble with multiple models and optional weights.

        Args:
            models: List of model objects with predict method
            weights: Optional weights for each model, defaults to equal weights
        """
        self.models = models

        if weights is None:
            # Equal weights if not provided
            self.weights = [1/len(models)] * len(models)
        else:
            # Normalize weights to sum to 1
            total = sum(weights)
            self.weights = [w/total for w in weights]

        logger.info(f"Initialized ensemble with {len(models)} models and weights: {self.weights}")

    def predict(self, X: Any) -> np.ndarray:
        """
        Generate weighted ensemble predictions.

        Args:
            X: Input data (could be different types for different models)

        Returns:
            Array of weighted average predictions
        """
        if not self.models:
            raise ValueError("No models in the ensemble")

        # Get predictions from each model
        predictions = []
        for model in self.models:
            pred = model.predict(X)

            # Ensure predictions are 1D arrays
            if pred.ndim > 1:
                pred = pred.flatten()

            predictions.append(pred)

        # Check if all predictions have the same length
        pred_lengths = [len(p) for p in predictions]
        if len(set(pred_lengths)) > 1:
            raise ValueError(f"Models returned predictions with different lengths: {pred_lengths}")

        # Apply weighted average
        weighted_predictions = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            weighted_predictions += pred * self.weights[i]

        return weighted_predictions

    def evaluate(self, X: Any, y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate ensemble performance.

        Args:
            X: Input data
            y_true: True target values

        Returns:
            Dictionary of evaluation metrics
        """
        # Generate predictions
        y_pred = self.predict(X)

        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)

        # Handle potential division by zero in MAPE calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100

        # If mape is infinity or NaN, set to a high value
        if np.isinf(mape) or np.isnan(mape):
            mape = 999.99

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }

        logger.info(f"Ensemble evaluation metrics: {metrics}")
        return metrics

    def evaluate_individual_models(self, X: Any, y_true: np.ndarray) -> List[Dict[str, float]]:
        """
        Evaluate each individual model.

        Args:
            X: Input data
            y_true: True target values

        Returns:
            List of evaluation metrics for each model
        """
        individual_metrics = []

        for i, model in enumerate(self.models):
            # Get predictions
            pred = model.predict(X)

            # Ensure predictions are 1D arrays
            if pred.ndim > 1:
                pred = pred.flatten()

            # Calculate metrics
            mse = mean_squared_error(y_true, pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, pred)

            # Handle potential division by zero in MAPE calculation
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = mean_absolute_percentage_error(y_true, pred) * 100

            # If mape is infinity or NaN, set to a high value
            if np.isinf(mape) or np.isnan(mape):
                mape = 999.99

            metrics = {
                'model_index': i,
                'weight': self.weights[i],
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape
            }

            individual_metrics.append(metrics)

        return individual_metrics

    def optimize_weights(self, X_val: Any, y_val: np.ndarray, method: str = 'grid_search') -> List[float]:
        """
        Optimize ensemble weights using validation data.

        Args:
            X_val: Validation input data
            y_val: Validation target values
            method: Optimization method ('grid_search' or 'error_based')

        Returns:
            Optimized weights
        """
        logger.info(f"Optimizing ensemble weights using {method} method")

        if method == 'grid_search':
            # Try different weight combinations using grid search
            return self._optimize_grid_search(X_val, y_val)
        elif method == 'error_based':
            # Set weights inversely proportional to error
            return self._optimize_error_based(X_val, y_val)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

    def _optimize_grid_search(self, X_val: Any, y_val: np.ndarray) -> List[float]:
        """
        Optimize weights using grid search.

        Args:
            X_val: Validation input data
            y_val: Validation target values

        Returns:
            Optimized weights
        """
        n_models = len(self.models)

        if n_models == 1:
            # Only one model, no need to optimize
            return [1.0]

        if n_models == 2:
            # For two models, we can do a fine-grained 1D search
            best_mape = float('inf')
            best_weights = self.weights

            # Try different values for the first weight (second is 1 - first)
            for w1 in np.linspace(0, 1, 21):  # 0.0, 0.05, 0.1, ..., 1.0
                w2 = 1 - w1
                self.weights = [w1, w2]

                metrics = self.evaluate(X_val, y_val)

                if metrics['mape'] < best_mape:
                    best_mape = metrics['mape']
                    best_weights = [w1, w2]
                    logger.debug(f"New best weights: {best_weights}, MAPE: {best_mape:.4f}")

            self.weights = best_weights
            logger.info(f"Optimized weights: {best_weights}, MAPE: {best_mape:.4f}")
            return best_weights

        elif n_models == 3:
            # For three models, we can do a grid search over two parameters
            # (third weight is determined by the constraint that they sum to 1)
            best_mape = float('inf')
            best_weights = self.weights

            # Try different combinations
            for w1 in np.linspace(0, 1, 11):  # 0.0, 0.1, 0.2, ..., 1.0
                for w2 in np.linspace(0, 1 - w1, int(10 * (1 - w1)) + 1):
                    w3 = 1 - w1 - w2
                    self.weights = [w1, w2, w3]

                    metrics = self.evaluate(X_val, y_val)

                    if metrics['mape'] < best_mape:
                        best_mape = metrics['mape']
                        best_weights = [w1, w2, w3]
                        logger.debug(f"New best weights: {best_weights}, MAPE: {best_mape:.4f}")

            self.weights = best_weights
            logger.info(f"Optimized weights: {best_weights}, MAPE: {best_mape:.4f}")
            return best_weights

        else:
            # For more than three models, we'll use a simplified approach
            # Start with equal weights and then try to improve each weight
            weights = [1/n_models] * n_models
            improvement = True
            iterations = 0
            max_iterations = 10

            while improvement and iterations < max_iterations:
                improvement = False
                old_weights = weights.copy()

                for i in range(n_models):
                    # Try increasing and decreasing this weight
                    for change in [-0.1, -0.05, 0.05, 0.1]:
                        # Skip if change would make weight negative
                        if weights[i] + change <= 0:
                            continue

                        # Make change and normalize
                        test_weights = weights.copy()
                        test_weights[i] += change
                        total = sum(test_weights)
                        test_weights = [w/total for w in test_weights]

                        # Test these weights
                        self.weights = test_weights
                        metrics = self.evaluate(X_val, y_val)

                        # Update if better
                        if metrics['mape'] < best_mape:
                            best_mape = metrics['mape']
                            best_weights = test_weights.copy()
                            weights = test_weights.copy()
                            improvement = True
                            logger.debug(f"New best weights: {best_weights}, MAPE: {best_mape:.4f}")

                # Check if weights changed significantly
                weight_change = np.sum(np.abs(np.array(weights) - np.array(old_weights)))
                if weight_change < 0.01:
                    break

                iterations += 1

            self.weights = best_weights
            logger.info(f"Optimized weights after {iterations} iterations: {best_weights}, MAPE: {best_mape:.4f}")
            return best_weights

    def _optimize_error_based(self, X_val: Any, y_val: np.ndarray) -> List[float]:
        """
        Optimize weights based on individual model errors.

        This sets weights inversely proportional to each model's error.

        Args:
            X_val: Validation input data
            y_val: Validation target values

        Returns:
            Optimized weights
        """
        # Get metrics for all models
        individual_metrics = self.evaluate_individual_models(X_val, y_val)

        # Extract errors (MAPE or MSE)
        errors = [metrics['mape'] for metrics in individual_metrics]

        # Handle case where all errors are the same
        if len(set(errors)) == 1:
            weights = [1/len(self.models)] * len(self.models)
            self.weights = weights
            return weights

        # Set weights inversely proportional to error
        # Use a softmax-like normalization to emphasize differences
        inv_errors = [1/(e + 1e-10) for e in errors]  # Add small epsilon to avoid division by zero
        total = sum(inv_errors)
        weights = [e/total for e in inv_errors]

        self.weights = weights

        # Evaluate with new weights
        metrics = self.evaluate(X_val, y_val)
        logger.info(f"Error-based weights: {weights}, MAPE: {metrics['mape']:.4f}")

        return weights


# Helper classes for model adaptation
class ModelAdapter:
    """Base adapter class for standardizing model interfaces."""

    def predict(self, X):
        """Prediction method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement predict method")


class XGBoostAdapter(ModelAdapter):
    """Adapter for XGBoost models."""

    def __init__(self, model):
        """Initialize with XGBoost model."""
        self.model = model

    def predict(self, X):
        """Make predictions with XGBoost model."""
        if isinstance(X, tuple):
            # If input is a tuple (for combined input), extract appropriate part
            X_xgb = X[0]
            return self.model.predict(X_xgb)
        return self.model.predict(X)


class LSTMAdapter(ModelAdapter):
    """Adapter for LSTM models."""

    def __init__(self, model):
        """Initialize with LSTM model."""
        self.model = model

    def predict(self, X):
        """Make predictions with LSTM model."""
        if isinstance(X, tuple):
            # If input is a tuple (for combined input), extract appropriate part
            X_lstm = X[1]
            return self.model.predict(X_lstm).flatten()
        return self.model.predict(X).flatten()


def create_ensemble(xgb_model, lstm_model, weights=None):
    """
    Create an ensemble from XGBoost and LSTM models.

    Args:
        xgb_model: XGBoost model
        lstm_model: LSTM model
        weights: Optional weights for models (default is [0.6, 0.4])

    Returns:
        Ensemble model
    """
    if weights is None:
        weights = [0.6, 0.4]

    # Create adapters
    xgb_adapter = XGBoostAdapter(xgb_model)
    lstm_adapter = LSTMAdapter(lstm_model)

    # Create ensemble
    ensemble = EnergyPredictionEnsemble(
        models=[xgb_adapter, lstm_adapter],
        weights=weights
    )

    return ensemble

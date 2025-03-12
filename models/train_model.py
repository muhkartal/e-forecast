"""
Model training functions for the Energy Prediction System.

This module provides functions for training, evaluating, and optimizing
machine learning models for energy consumption prediction.
"""

import os
import logging
import yaml
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.model_selection import train_test_split

from .xgboost_model import XGBoostEnergyPredictor
from .lstm_model import EnergyPredictionLSTM, AttentionLSTM
from .ensemble import EnergyPredictionEnsemble, create_ensemble

# Configure logging
logger = logging.getLogger(__name__)


def train_xgboost_model(X_train: pd.DataFrame,
                       y_train: pd.Series,
                       X_val: pd.DataFrame,
                       y_val: pd.Series,
                       config: Dict[str, Any]) -> XGBoostEnergyPredictor:
    """
    Train XGBoost model for energy prediction.

    Args:
        X_train: Training features
        y_train: Training target values
        X_val: Validation features
        y_val: Validation target values
        config: Model configuration dictionary

    Returns:
        Trained XGBoost model
    """
    logger.info("Training XGBoost model")

    # Initialize model with configuration
    model = XGBoostEnergyPredictor(config)

    # Determine whether to use hyperparameter optimization
    use_hp_tuning = config.get('hp_tuning', {}).get('enabled', False)

    if use_hp_tuning:
        # Get number of trials
        n_trials = config.get('hp_tuning', {}).get('max_evals', 50)
        logger.info(f"Performing hyperparameter optimization with {n_trials} trials")

        # Run optimization
        model.hyperparameter_optimization(
            X_train, y_train,
            X_val, y_val,
            max_evals=n_trials
        )
    else:
        # Train with specified or default parameters
        logger.info("Training with specified parameters")
        model.train_basic(X_train, y_train)

    return model


def train_lstm_model(X_train: np.ndarray,
                    y_train: np.ndarray,
                    X_val: np.ndarray,
                    y_val: np.ndarray,
                    config: Dict[str, Any]) -> Union[EnergyPredictionLSTM, AttentionLSTM]:
    """
    Train LSTM model for energy prediction.

    Args:
        X_train: Training sequences
        y_train: Training target values
        X_val: Validation sequences
        y_val: Validation target values
        config: Model configuration dictionary

    Returns:
        Trained LSTM model
    """
    logger.info("Training LSTM model")

    # Check if attention mechanism should be used
    use_attention = config.get('attention', False)

    # Initialize appropriate model type
    if use_attention:
        logger.info("Using attention mechanism")
        model = AttentionLSTM(config)
    else:
        model = EnergyPredictionLSTM(config)

    # Get input shape from training data
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Build model architecture
    model.build_model(input_shape)

    # Create directory for model checkpoints
    checkpoint_dir = os.path.join("models", "checkpoints", "lstm")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Train the model
    model.train(X_train, y_train, X_val, y_val, model_dir=checkpoint_dir)

    return model


def create_ensemble_model(xgb_model: XGBoostEnergyPredictor,
                        lstm_model: Union[EnergyPredictionLSTM, AttentionLSTM],
                        X_val_xgb: pd.DataFrame,
                        y_val: Union[pd.Series, np.ndarray],
                        X_val_lstm: np.ndarray,
                        config: Dict[str, Any]) -> EnergyPredictionEnsemble:
    """
    Create an ensemble model combining XGBoost and LSTM.

    Args:
        xgb_model: Trained XGBoost model
        lstm_model: Trained LSTM model
        X_val_xgb: Validation data for XGBoost
        y_val: Validation target values
        X_val_lstm: Validation sequences for LSTM
        config: Ensemble configuration dictionary

    Returns:
        Ensemble model
    """
    logger.info("Creating ensemble model")

    # Get initial weights from config
    weights = config.get('weights', {'xgboost': 0.6, 'lstm': 0.4})
    initial_weights = [weights.get('xgboost', 0.6), weights.get('lstm', 0.4)]

    # Create ensemble
    ensemble = create_ensemble(
        xgb_model.model,
        lstm_model.model,
        weights=initial_weights
    )

    # Optimize weights if configured
    if config.get('weight_optimization', False):
        logger.info("Optimizing ensemble weights")

        # Create combined input for validation
        X_val_combined = (X_val_xgb, X_val_lstm)

        # Convert y_val to numpy array if it's a pandas Series
        if isinstance(y_val, pd.Series):
            y_val_np = y_val.values
        else:
            y_val_np = y_val

        # Optimize weights
        ensemble.optimize_weights(X_val_combined, y_val_np)

        logger.info(f"Optimized weights: {ensemble.weights}")

    return ensemble


def evaluate_models(xgb_model: XGBoostEnergyPredictor,
                   lstm_model: Union[EnergyPredictionLSTM, AttentionLSTM],
                   ensemble_model: EnergyPredictionEnsemble,
                   X_test_xgb: pd.DataFrame,
                   y_test: Union[pd.Series, np.ndarray],
                   X_test_lstm: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all models on test data.

    Args:
        xgb_model: Trained XGBoost model
        lstm_model: Trained LSTM model
        ensemble_model: Trained ensemble model
        X_test_xgb: Test data for XGBoost
        y_test: Test target values
        X_test_lstm: Test sequences for LSTM

    Returns:
        Dictionary of evaluation metrics for each model
    """
    logger.info("Evaluating models on test data")

    # Convert y_test to numpy array if it's a pandas Series
    if isinstance(y_test, pd.Series):
        y_test_np = y_test.values
    else:
        y_test_np = y_test

    # Evaluate XGBoost model
    xgb_metrics = xgb_model.evaluate(X_test_xgb, y_test)
    logger.info(f"XGBoost metrics: {xgb_metrics}")

    # Evaluate LSTM model
    lstm_metrics = lstm_model.evaluate(X_test_lstm, y_test_np)
    logger.info(f"LSTM metrics: {lstm_metrics}")

    # Evaluate ensemble model
    X_test_combined = (X_test_xgb, X_test_lstm)
    ensemble_metrics = ensemble_model.evaluate(X_test_combined, y_test_np)
    logger.info(f"Ensemble metrics: {ensemble_metrics}")

    # Return all metrics
    return {
        'xgboost': xgb_metrics,
        'lstm': lstm_metrics,
        'ensemble': ensemble_metrics
    }


def save_models(xgb_model: XGBoostEnergyPredictor,
               lstm_model: Union[EnergyPredictionLSTM, AttentionLSTM],
               ensemble_weights: List[float],
               metrics: Dict[str, Dict[str, float]],
               output_dir: str = "models/trained") -> Dict[str, str]:
    """
    Save trained models to disk.

    Args:
        xgb_model: Trained XGBoost model
        lstm_model: Trained LSTM model
        ensemble_weights: Ensemble model weights
        metrics: Evaluation metrics
        output_dir: Directory to save models

    Returns:
        Dictionary of saved model paths
    """
    logger.info(f"Saving models to {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save XGBoost model
    xgb_path = os.path.join(output_dir, "xgboost_model.joblib")
    xgb_model.save_model(xgb_path)
    logger.info(f"Saved XGBoost model to {xgb_path}")

    # Save LSTM model
    lstm_path = os.path.join(output_dir, "lstm_model.h5")
    lstm_model.save_model(lstm_path)
    logger.info(f"Saved LSTM model to {lstm_path}")

    # Save ensemble configuration
    ensemble_config = {
        'weights': ensemble_weights,
        'metrics': metrics['ensemble']
    }

    ensemble_path = os.path.join(output_dir, "ensemble_config.json")
    with open(ensemble_path, 'w') as f:
        json.dump(ensemble_config, f, indent=2)
    logger.info(f"Saved ensemble configuration to {ensemble_path}")

    # Save evaluation metrics
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved evaluation metrics to {metrics_path}")

    return {
        'xgboost': xgb_path,
        'lstm': lstm_path,
        'ensemble': ensemble_path,
        'metrics': metrics_path
    }


def load_models(model_dir: str = "models/trained") -> Tuple[XGBoostEnergyPredictor, Union[EnergyPredictionLSTM, Any], List[float]]:
    """
    Load trained models from disk.

    Args:
        model_dir: Directory containing saved models

    Returns:
        Tuple of (XGBoost model, LSTM model, ensemble weights)
    """
    logger.info(f"Loading models from {model_dir}")

    # Load XGBoost model
    xgb_path = os.path.join(model_dir, "xgboost_model.joblib")
    xgb_model = XGBoostEnergyPredictor({})
    xgb_model.load_model(xgb_path)
    logger.info(f"Loaded XGBoost model from {xgb_path}")

    # Load LSTM model
    lstm_path = os.path.join(model_dir, "lstm_model.h5")
    lstm_model = EnergyPredictionLSTM({})
    lstm_model.load_model(lstm_path)
    logger.info(f"Loaded LSTM model from {lstm_path}")

    # Load ensemble configuration
    ensemble_path = os.path.join(model_dir, "ensemble_config.json")
    with open(ensemble_path, 'r') as f:
        ensemble_config = json.load(f)

    ensemble_weights = ensemble_config['weights']
    logger.info(f"Loaded ensemble weights: {ensemble_weights}")

    return xgb_model, lstm_model, ensemble_weights


def plot_model_comparison(metrics: Dict[str, Dict[str, float]],
                         output_dir: str = "results/plots") -> None:
    """
    Plot model comparison based on evaluation metrics.

    Args:
        metrics: Dictionary of evaluation metrics for each model
        output_dir: Directory to save plots
    """
    logger.info("Plotting model comparison")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract metrics for plotting
    models = list(metrics.keys())
    mse_values = [metrics[model]['mse'] for model in models]
    rmse_values = [metrics[model]['rmse'] for model in models]
    mae_values = [metrics[model]['mae'] for model in models]
    mape_values = [metrics[model].get('mape', 0) for model in models]

    # Create bar plot
    plt.figure(figsize=(12, 8))

    # MSE
    plt.subplot(2, 2, 1)
    plt.bar(models, mse_values)
    plt.title('Mean Squared Error')
    plt.ylabel('MSE')
    plt.grid(axis='y', alpha=0.3)

    # RMSE
    plt.subplot(2, 2, 2)
    plt.bar(models, rmse_values)
    plt.title('Root Mean Squared Error')
    plt.ylabel('RMSE')
    plt.grid(axis='y', alpha=0.3)

    # MAE
    plt.subplot(2, 2, 3)
    plt.bar(models, mae_values)
    plt.title('Mean Absolute Error')
    plt.ylabel('MAE')
    plt.grid(axis='y', alpha=0.3)

    # MAPE
    plt.subplot(2, 2, 4)
    plt.bar(models, mape_values)
    plt.title('Mean Absolute Percentage Error')
    plt.ylabel('MAPE (%)')
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Saved model comparison plot to {plot_path}")

    # Create radar chart for comprehensive comparison
    plt.figure(figsize=(8, 8))

    # Normalize metrics for radar chart
    max_mse = max(mse_values)
    max_rmse = max(rmse_values)
    max_mae = max(mae_values)
    max_mape = max(mape_values)

    # Invert metrics for radar chart (lower is better, but radar shows larger as better)
    norm_mse = [1 - (val / max_mse) if max_mse > 0 else 0 for val in mse_values]
    norm_rmse = [1 - (val / max_rmse) if max_rmse > 0 else 0 for val in rmse_values]
    norm_mae = [1 - (val / max_mae) if max_mae > 0 else 0 for val in mae_values]
    norm_mape = [1 - (val / max_mape) if max_mape > 0 else 0 for val in mape_values]

    # Set up radar chart
    categories = ['MSE', 'RMSE', 'MAE', 'MAPE']
    N = len(categories)

    # Create angles for radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    # Set up subplot in polar coordinates
    ax = plt.subplot(111, polar=True)

    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories)

    # Draw the performance for each model
    for i, model in enumerate(models):
        values = [norm_mse[i], norm_rmse[i], norm_mae[i], norm_mape[i]]
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.1)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.title('Model Performance Comparison')

    # Save radar chart
    radar_path = os.path.join(output_dir, "model_radar_comparison.png")
    plt.savefig(radar_path, dpi=300)
    logger.info(f"Saved radar chart to {radar_path}")


def plot_feature_importance(xgb_model: XGBoostEnergyPredictor,
                           output_dir: str = "results/plots") -> None:
    """
    Plot feature importance from XGBoost model.

    Args:
        xgb_model: Trained XGBoost model
        output_dir: Directory to save plots
    """
    logger.info("Plotting feature importance")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get feature importance
    importance = xgb_model.feature_importance(plot=False)

    # Sort features by importance
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    features = [x[0] for x in sorted_features]
    values = [x[1] for x in sorted_features]

    # Limit to top 20 features if there are many
    if len(features) > 20:
        features = features[:20]
        values = values[:20]

    # Create horizontal bar plot
    plt.figure(figsize=(10, max(6, len(features) * 0.3)))
    plt.barh(range(len(features)), values, align='center')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Saved feature importance plot to {plot_path}")


def plot_predictions(xgb_model: XGBoostEnergyPredictor,
                   lstm_model: Union[EnergyPredictionLSTM, AttentionLSTM],
                   ensemble_model: EnergyPredictionEnsemble,
                   X_test_xgb: pd.DataFrame,
                   y_test: Union[pd.Series, np.ndarray],
                   X_test_lstm: np.ndarray,
                   output_dir: str = "results/plots") -> None:
    """
    Plot model predictions against actual values.

    Args:
        xgb_model: Trained XGBoost model
        lstm_model: Trained LSTM model
        ensemble_model: Trained ensemble model
        X_test_xgb: Test data for XGBoost
        y_test: Test target values
        X_test_lstm: Test sequences for LSTM
        output_dir: Directory to save plots
    """
    logger.info("Plotting model predictions")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert y_test to numpy array if it's a pandas Series
    if isinstance(y_test, pd.Series):
        y_test_np = y_test.values
    else:
        y_test_np = y_test

    # Generate predictions
    y_pred_xgb = xgb_model.predict(X_test_xgb)
    y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()
    y_pred_ensemble = ensemble_model.predict((X_test_xgb, X_test_lstm))

    # Create scatterplots of predictions vs actual
    plt.figure(figsize=(18, 6))

    # XGBoost
    plt.subplot(1, 3, 1)
    plt.scatter(y_test_np, y_pred_xgb, alpha=0.5)
    plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'r--')
    plt.title('XGBoost: Predicted vs Actual')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(alpha=0.3)

    # LSTM
    plt.subplot(1, 3, 2)
    plt.scatter(y_test_np, y_pred_lstm, alpha=0.5)
    plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'r--')
    plt.title('LSTM: Predicted vs Actual')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(alpha=0.3)

    # Ensemble
    plt.subplot(1, 3, 3)
    plt.scatter(y_test_np, y_pred_ensemble, alpha=0.5)
    plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'r--')
    plt.title('Ensemble: Predicted vs Actual')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, "predictions_vs_actual.png")
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Saved predictions plot to {plot_path}")

    # Plot time series of predictions (using a subset of data for clarity)
    sample_size = min(100, len(y_test_np))
    indices = np.arange(sample_size)

    plt.figure(figsize=(12, 6))
    plt.plot(indices, y_test_np[:sample_size], 'o-', label='Actual', alpha=0.7)
    plt.plot(indices, y_pred_xgb[:sample_size], 's--', label='XGBoost', alpha=0.7)
    plt.plot(indices, y_pred_lstm[:sample_size], '^--', label='LSTM', alpha=0.7)
    plt.plot(indices, y_pred_ensemble[:sample_size], 'D-', label='Ensemble', alpha=0.7)
    plt.title('Model Predictions Comparison')
    plt.xlabel('Sample Index')
    plt.ylabel('Energy Consumption')
    plt.legend()
    plt.grid(alpha=0.3)

    # Save time series plot
    plot_path = os.path.join(output_dir, "prediction_time_series.png")
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Saved time series plot to {plot_path}")

    # Plot error distributions
    plt.figure(figsize=(18, 6))

    # XGBoost errors
    plt.subplot(1, 3, 1)
    xgb_errors = y_test_np - y_pred_xgb
    plt.hist(xgb_errors, bins=50, alpha=0.7)
    plt.title('XGBoost: Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)

    # LSTM errors
    plt.subplot(1, 3, 2)
    lstm_errors = y_test_np - y_pred_lstm
    plt.hist(lstm_errors, bins=50, alpha=0.7)
    plt.title('LSTM: Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)

    # Ensemble errors
    plt.subplot(1, 3, 3)
    ensemble_errors = y_test_np - y_pred_ensemble
    plt.hist(ensemble_errors, bins=50, alpha=0.7)
    plt.title('Ensemble: Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)

    plt.tight_layout()

    # Save error plot
    plot_path = os.path.join(output_dir, "error_distributions.png")
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Saved error distribution plot to {plot_path}")


def cross_validate_models(X: pd.DataFrame, y: pd.Series,
                         config: Dict[str, Any],
                         n_folds: int = 5,
                         random_seed: int = 42) -> Dict[str, List[float]]:
    """
    Perform cross-validation for model evaluation.

    Args:
        X: Feature data
        y: Target values
        config: Model configuration dictionary
        n_folds: Number of cross-validation folds
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary of metrics for each model type
    """
    logger.info(f"Performing {n_folds}-fold cross-validation")

    # Initialize results dictionary
    results = {
        'xgboost_rmse': [],
        'xgboost_mape': [],
        'lstm_rmse': [],
        'lstm_mape': [],
        'ensemble_rmse': [],
        'ensemble_mape': []
    }

    # Create folds
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    # Perform cross-validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        logger.info(f"Training fold {fold+1}/{n_folds}")

        # Split data
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

        # Further split train into train and validation
        X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
            X_train_fold, y_train_fold, test_size=0.2, random_state=random_seed + fold
        )

        # Train XGBoost model
        xgb_model = train_xgboost_model(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold, config['xgboost']
        )

        # Evaluate XGBoost
        xgb_metrics = xgb_model.evaluate(X_test_fold, y_test_fold)
        results['xgboost_rmse'].append(xgb_metrics['rmse'])
        results['xgboost_mape'].append(xgb_metrics['mape'])

        # Prepare sequence data for LSTM
        # This would require preprocessing the data for sequence creation
        # For simplicity, we'll skip LSTM cross-validation here
        # In a real implementation, you would need to create sequences for each fold

        logger.info(f"Fold {fold+1} results - XGBoost RMSE: {xgb_metrics['rmse']:.4f}, MAPE: {xgb_metrics['mape']:.2f}%")

    # Calculate average metrics
    avg_results = {
        'xgboost_rmse_avg': np.mean(results['xgboost_rmse']),
        'xgboost_rmse_std': np.std(results['xgboost_rmse']),
        'xgboost_mape_avg': np.mean(results['xgboost_mape']),
        'xgboost_mape_std': np.std(results['xgboost_mape'])
    }

    logger.info(f"Cross-validation results: {avg_results}")
    return results


def get_learning_curves(X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series,
                      config: Dict[str, Any],
                      output_dir: str = "results/plots") -> None:
    """
    Generate learning curves for XGBoost model.

    Args:
        X_train: Training features
        y_train: Training target values
        X_val: Validation features
        y_val: Validation target values
        config: Model configuration dictionary
        output_dir: Directory to save plots
    """
    logger.info("Generating learning curves")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Define training sizes
    train_sizes = np.linspace(0.1, 1.0, 10)

    train_errors = []
    val_errors = []

    for size in train_sizes:
        # Calculate number of samples for this size
        n_samples = int(len(X_train) * size)

        # Subsample training data
        X_train_sub = X_train.iloc[:n_samples]
        y_train_sub = y_train.iloc[:n_samples]

        # Train model
        model = XGBoostEnergyPredictor(config['xgboost'])
        model.train_basic(X_train_sub, y_train_sub)

        # Evaluate on training and validation data
        train_metrics = model.evaluate(X_train_sub, y_train_sub)
        val_metrics = model.evaluate(X_val, y_val)

        # Record errors
        train_errors.append(train_metrics['rmse'])
        val_errors.append(val_metrics['rmse'])

        logger.debug(f"Training size {size:.2f} - Train RMSE: {train_metrics['rmse']:.4f}, Val RMSE: {val_metrics['rmse']:.4f}")

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_errors, 'o-', label='Training error')
    plt.plot(train_sizes, val_errors, 's-', label='Validation error')
    plt.title('Learning Curves (XGBoost)')
    plt.xlabel('Training Set Size (fraction)')
    plt.ylabel('Root Mean Squared Error')
    plt.legend()
    plt.grid(alpha=0.3)

    # Save plot
    plot_path = os.path.join(output_dir, "learning_curves.png")
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Saved learning curves plot to {plot_path}")

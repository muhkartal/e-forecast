"""
Visualization utilities for the Energy Prediction System.

This module provides functions for visualizing data, features, models, and results
to help understand and interpret energy consumption predictions.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
import joblib
import json

# Configure logging
logger = logging.getLogger(__name__)

# Set default style
plt.style.use('fivethirtyeight')
sns.set_palette('viridis')


def plot_feature_importance(feature_importance: Dict[str, float],
                          title: str = 'Feature Importance',
                          n_features: int = 20,
                          figsize: Tuple[int, int] = (12, 10),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot feature importance from a trained model.

    Args:
        feature_importance: Dictionary mapping feature names to importance values
        title: Plot title
        n_features: Number of top features to display
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    # Select top N features
    top_features = sorted_features[:n_features]
    features = [x[0] for x in top_features]
    importance = [x[1] for x in top_features]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create horizontal bar chart
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importance, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Importance')
    ax.set_title(title)

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {save_path}")

    return fig


def plot_predictions(y_true: np.ndarray,
                   predictions: Dict[str, np.ndarray],
                   title: str = 'Model Predictions Comparison',
                   figsize: Tuple[int, int] = (16, 10),
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot predictions from multiple models against actual values.

    Args:
        y_true: Array of actual target values
        predictions: Dictionary mapping model names to prediction arrays
        title: Plot title
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    n_models = len(predictions)

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Create scatter plots for each model
    for i, (model_name, y_pred) in enumerate(predictions.items()):
        # Calculate metrics
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)

        # Scatter plot of predictions vs actual
        ax1 = plt.subplot(2, n_models, i + 1)
        ax1.scatter(y_true, y_pred, alpha=0.5)

        # Add diagonal line for perfect predictions
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--')

        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.set_title(f'{model_name}: RMSE = {rmse:.4f}')
        ax1.grid(True, alpha=0.3)

        # Histogram of residuals
        ax2 = plt.subplot(2, n_models, i + n_models + 1)
        residuals = y_pred - y_true
        ax2.hist(residuals, bins=30, alpha=0.7)
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{model_name}: Residual Distribution')
        ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved predictions plot to {save_path}")

    return fig


def plot_time_series_prediction(timestamps: List[Union[str, pd.Timestamp]],
                              y_true: np.ndarray,
                              predictions: Dict[str, np.ndarray],
                              title: str = 'Time Series Prediction Comparison',
                              sample_size: Optional[int] = None,
                              figsize: Tuple[int, int] = (16, 8),
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot time series predictions from multiple models.

    Args:
        timestamps: List of timestamps
        y_true: Array of actual target values
        predictions: Dictionary mapping model names to prediction arrays
        title: Plot title
        sample_size: Optional number of samples to plot (for large datasets)
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    # Convert timestamps to pandas datetime if they are strings
    if isinstance(timestamps[0], str):
        timestamps = pd.to_datetime(timestamps)

    # Limit sample size if specified
    if sample_size and sample_size < len(timestamps):
        # Take the last sample_size points
        timestamps = timestamps[-sample_size:]
        y_true = y_true[-sample_size:]
        predictions = {model: pred[-sample_size:] for model, pred in predictions.items()}

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot actual values
    ax.plot(timestamps, y_true, 'o-', label='Actual', alpha=0.7, linewidth=2)

    # Plot predictions for each model
    markers = ['s', '^', 'D', 'x', '*']  # Different markers for different models
    for i, (model_name, y_pred) in enumerate(predictions.items()):
        marker = markers[i % len(markers)]
        ax.plot(timestamps, y_pred, marker + '--', label=model_name, alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Time')
    ax.set_ylabel('Energy Consumption')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Format x-axis for better readability
    fig.autofmt_xdate()

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved time series plot to {save_path}")

    return fig


def plot_correlation_matrix(df: pd.DataFrame,
                          target_col: Optional[str] = None,
                          n_features: int = 20,
                          figsize: Tuple[int, int] = (12, 10),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot correlation matrix of features.

    Args:
        df: DataFrame with features
        target_col: Optional target column to highlight correlations with
        n_features: Number of top features to include
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=np.number)

    # If target column specified, select features with highest correlation to target
    if target_col and target_col in numeric_df.columns:
        # Calculate absolute correlation with target
        correlations = numeric_df.corrwith(numeric_df[target_col]).abs()

        # Select top N features (excluding target)
        top_features = correlations.drop(target_col).nlargest(n_features).index.tolist()

        # Add target column back
        selected_columns = top_features + [target_col]

        # Filter dataframe to only include these columns
        corr_df = numeric_df[selected_columns]
    else:
        # If no target or target not in dataframe, use all columns up to n_features
        if len(numeric_df.columns) > n_features:
            corr_df = numeric_df.iloc[:, :n_features]
        else:
            corr_df = numeric_df

    # Calculate correlation matrix
    corr_matrix = corr_df.corr()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask for upper triangle
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
               square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)

    # Set title
    if target_col:
        ax.set_title(f'Correlation Matrix (Top {n_features} Features vs {target_col})')
    else:
        ax.set_title(f'Correlation Matrix (Top {n_features} Features)')

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved correlation matrix plot to {save_path}")

    return fig


def plot_energy_consumption(df: pd.DataFrame,
                          energy_col: str = 'energy_consumption',
                          groupby_col: Optional[str] = None,
                          rolling_window: Optional[int] = None,
                          figsize: Tuple[int, int] = (16, 8),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot energy consumption patterns.

    Args:
        df: DataFrame with energy consumption data
        energy_col: Column name for energy consumption
        groupby_col: Optional column to group by (e.g., 'vehicle_id', 'day_of_week')
        rolling_window: Optional window size for rolling average
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    # Check if energy column exists
    if energy_col not in df.columns:
        raise ValueError(f"Energy column '{energy_col}' not found in DataFrame")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Handle groupby case
    if groupby_col and groupby_col in df.columns:
        # Group by the specified column
        grouped = df.groupby(groupby_col)[energy_col]

        # Calculate statistics
        mean = grouped.mean()
        std = grouped.std()

        # Plot each group
        mean.plot(kind='bar', yerr=std, alpha=0.7, ax=ax)
        ax.set_ylabel(f'Mean {energy_col}')
        ax.set_title(f'Energy Consumption by {groupby_col}')

    # Handle time series case
    elif 'timestamp' in df.columns:
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort by timestamp
        df_sorted = df.sort_values('timestamp')

        # Apply rolling window if specified
        if rolling_window:
            # Calculate rolling mean
            rolling_mean = df_sorted[energy_col].rolling(window=rolling_window, center=True).mean()

            # Plot original and rolling average
            ax.plot(df_sorted['timestamp'], df_sorted[energy_col], 'o-', alpha=0.4, label='Raw')
            ax.plot(df_sorted['timestamp'], rolling_mean, 'r-', linewidth=2,
                  label=f'Rolling Avg (window={rolling_window})')
            ax.legend()
        else:
            # Plot raw values
            ax.plot(df_sorted['timestamp'], df_sorted[energy_col], 'o-', alpha=0.7)

        ax.set_xlabel('Time')
        ax.set_ylabel(energy_col)
        ax.set_title('Energy Consumption Over Time')
        fig.autofmt_xdate()

    # Handle regular case (no groupby, no timestamp)
    else:
        # Plot histogram
        ax.hist(df[energy_col], bins=30, alpha=0.7)
        ax.set_xlabel(energy_col)
        ax.set_ylabel('Frequency')
        ax.set_title('Energy Consumption Distribution')

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved energy consumption plot to {save_path}")

    return fig


def plot_model_comparison(metrics: Dict[str, Dict[str, float]],
                        figsize: Tuple[int, int] = (14, 10),
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot comparison of multiple models based on their evaluation metrics.

    Args:
        metrics: Dictionary mapping model names to dictionaries of metrics
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    # Extract model names and metrics
    models = list(metrics.keys())

    # Common metrics to compare
    metric_names = ['rmse', 'mse', 'mae', 'mape']
    available_metrics = []

    # Find metrics available in all models
    for metric in metric_names:
        if all(metric in metrics[model] for model in models):
            available_metrics.append(metric)

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Plot each metric as a bar chart
    for i, metric in enumerate(available_metrics):
        # Extract values for this metric
        values = [metrics[model][metric] for model in models]

        # Create subplot
        ax = plt.subplot(2, 2, i + 1)

        # Create bar chart
        ax.bar(range(len(models)), values, tick_label=models)

        # Add values on top of bars
        for j, v in enumerate(values):
            ax.text(j, v, f"{v:.4f}", ha='center', va='bottom')

        # Set title and labels
        ax.set_title(metric.upper())
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Model Performance Comparison', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved model comparison plot to {save_path}")

    return fig


def plot_learning_curves(train_sizes: List[float],
                        train_scores: List[float],
                        val_scores: List[float],
                        metric_name: str = 'RMSE',
                        title: str = 'Learning Curves',
                        figsize: Tuple[int, int] = (10, 6),
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot learning curves showing model performance on training and validation sets.

    Args:
        train_sizes: List of training set sizes (fractions or absolute counts)
        train_scores: List of scores on training sets
        val_scores: List of scores on validation sets
        metric_name: Name of the metric being plotted
        title: Plot title
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot learning curves
    ax.plot(train_sizes, train_scores, 'o-', label=f'Training {metric_name}')
    ax.plot(train_sizes, val_scores, 's-', label=f'Validation {metric_name}')

    # Add labels and title
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel(metric_name)
    ax.set_title(title)

    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved learning curves plot to {save_path}")

    return fig


def visualize_model_results(model_dir: str, output_dir: str = 'results/plots'):
    """
    Generate visualization plots from trained models and their evaluation metrics.

    Args:
        model_dir: Directory containing trained models and metrics
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load evaluation metrics
        metrics_path = os.path.join(model_dir, 'evaluation_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

            # Plot model comparison
            plot_model_comparison(
                metrics=metrics,
                save_path=os.path.join(output_dir, 'model_comparison.png')
            )

        # Load XGBoost model and plot feature importance
        xgb_path = os.path.join(model_dir, 'xgboost_model.joblib')
        if os.path.exists(xgb_path):
            try:
                xgb_model = joblib.load(xgb_path)

                # Check if it's a model package with 'model' key
                if isinstance(xgb_model, dict) and 'model' in xgb_model:
                    model = xgb_model['model']
                    feature_names = xgb_model.get('feature_names', None)
                else:
                    model = xgb_model
                    feature_names = getattr(model, 'feature_names_in_', None)

                # Get feature importance
                importance = model.feature_importances_

                # Map to feature names if available
                if feature_names:
                    importance_dict = dict(zip(feature_names, importance))
                else:
                    importance_dict = {f"feature_{i}": imp for i, imp in enumerate(importance)}

                # Plot feature importance
                plot_feature_importance(
                    feature_importance=importance_dict,
                    title='XGBoost Feature Importance',
                    save_path=os.path.join(output_dir, 'feature_importance.png')
                )
            except Exception as e:
                logger.error(f"Error plotting XGBoost feature importance: {str(e)}")

        # Add more visualization as needed...

        logger.info(f"Model visualization saved to {output_dir}")

    except Exception as e:
        logger.error(f"Error in model visualization: {str(e)}")


if __name__ == '__main__':
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate visualizations from trained models')
    parser.add_argument('--model-dir', type=str, default='models/trained',
                      help='Directory containing trained models and metrics')
    parser.add_argument('--output-dir', type=str, default='results/plots',
                      help='Directory to save plots')

    # Parse arguments
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Generate visualizations
    visualize_model_results(args.model_dir, args.output_dir)

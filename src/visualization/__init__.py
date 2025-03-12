"""
Visualization package for the Energy Prediction System.

This package provides modules for visualizing data, features, and model results
to gain insights into energy consumption prediction.
"""

from .visualize import (
    plot_feature_importance,
    plot_predictions,
    plot_correlation_matrix,
    plot_energy_consumption,
    plot_model_comparison
)

__all__ = [
    'plot_feature_importance',
    'plot_predictions',
    'plot_correlation_matrix',
    'plot_energy_consumption',
    'plot_model_comparison'
]

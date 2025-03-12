"""
Models package for the Energy Prediction System.

This package contains model implementations for predicting energy consumption
in electric vehicles, including gradient boosting, deep learning, and ensembles.
"""

from .xgboost_model import XGBoostEnergyPredictor
from .lstm_model import EnergyPredictionLSTM, AttentionLSTM
from .ensemble import EnergyPredictionEnsemble, create_ensemble

__all__ = [
    'XGBoostEnergyPredictor',
    'EnergyPredictionLSTM',
    'AttentionLSTM',
    'EnergyPredictionEnsemble',
    'create_ensemble'
]

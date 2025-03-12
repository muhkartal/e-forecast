"""
Configuration for pytest.
"""
import os
import sys
import pytest
import numpy as np


# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def sample_data():
    """
    Fixture providing sample data for tests.
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta

    # Create sample data
    sample_size = 100
    timestamps = [datetime(2023, 1, 1) + timedelta(minutes=i) for i in range(sample_size)]

    data = {
        'timestamp': timestamps,
        'speed': np.random.uniform(0, 120, sample_size),
        'altitude': np.random.uniform(0, 1000, sample_size),
        'temperature': np.random.uniform(-10, 40, sample_size),
        'weight': np.random.uniform(1500, 2500, sample_size),
        'drag_coefficient': np.random.uniform(0.2, 0.3, sample_size),
        'frontal_area': np.random.uniform(2.0, 3.0, sample_size),
        'road_type': np.random.choice(['highway', 'urban', 'rural'], sample_size),
        'energy_consumption': np.random.uniform(10, 30, sample_size)
    }

    return pd.DataFrame(data)

@pytest.fixture
def sample_sequence_data():
    """
    Fixture providing sample sequence data for LSTM model tests.
    """
    import numpy as np

    # Create synthetic sequence data
    n_samples = 100
    sequence_length = 10
    n_features = 5

    # Create random sequences
    X = np.random.rand(n_samples, sequence_length, n_features)
    y = np.random.rand(n_samples)

    # Split into train/val/test
    X_train = X[:80]
    y_train = y[:80]
    X_val = X[80:90]
    y_val = y[80:90]
    X_test = X[90:]
    y_test = y[90:]

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'sequence_length': sequence_length,
        'n_features': n_features
    }

@pytest.fixture
def model_config():
    """
    Fixture providing model configuration for tests.
    """
    return {
        'general': {
            'random_seed': 42,
            'target_column': 'energy_consumption',
            'test_size': 0.15,
            'validation_size': 0.15,
            'sequence_length': 10
        },
        'preprocessing': {
            'outlier_method': 'clip',
            'outlier_threshold': 3.0
        },
        'xgboost': {
            'n_estimators': 50,
            'max_depth': 3,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': 0
        },
        'lstm': {
            'lstm_units': [32, 16],
            'dense_units': [8],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'epochs': 2,
            'batch_size': 16,
            'patience': 5
        }
    }

@pytest.fixture
def api_client():
    """
    Fixture providing a TestClient for FastAPI tests.
    """
    from fastapi.testclient import TestClient
    from unittest import mock

    # Import the API
    from src.api.main import app

    # Create a test client
    client = TestClient(app)

    # Set up mocks
    mock_preprocessors = {
        'scaler': mock.MagicMock(),
        'encoder': mock.MagicMock()
    }

    mock_feature_config = {
        'xgb_features': ['speed', 'temperature', 'weight'],
        'lstm_features': ['speed', 'temperature', 'weight'],
        'numeric_features': ['speed', 'temperature', 'weight'],
        'categorical_features': ['road_type'],
        'sequence_length': 10
    }

    mock_xgb = mock.MagicMock()
    mock_xgb.predict.return_value = np.array([15.0])

    mock_lstm = mock.MagicMock()
    mock_lstm.predict.return_value = np.array([[14.0]])

    mock_models = {
        'xgboost': mock_xgb,
        'lstm': mock_lstm
    }

    mock_config = {
        'prediction': {
            'ensemble_weights': {
                'xgboost': 0.6,
                'lstm': 0.4
            },
            'sequence_length': 10
        }
    }

    # Apply mocks
    with mock.patch('src.api.main.preprocessors', mock_preprocessors), \
         mock.patch('src.api.main.feature_config', mock_feature_config), \
         mock.patch('src.api.main.models', mock_models), \
         mock.patch('src.api.main.config', mock_config):

        yield client

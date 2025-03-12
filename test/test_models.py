"""
Tests for the machine learning models.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import tempfile
from sklearn.datasets import make_regression

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.xgboost_model import XGBoostEnergyPredictor
from src.models.lstm_model import EnergyPredictionLSTM, AttentionLSTM
from src.models.ensemble import EnergyPredictionEnsemble


class TestXGBoostModel(unittest.TestCase):
    """Test cases for the XGBoost model."""

    def setUp(self):
        """Set up test data."""
        # Create synthetic regression data
        X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)

        # Convert to DataFrame for column names
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        self.X_train = pd.DataFrame(X[:400], columns=feature_names)
        self.y_train = pd.Series(y[:400])
        self.X_test = pd.DataFrame(X[400:], columns=feature_names)
        self.y_test = pd.Series(y[400:])

        # Model configuration
        self.config = {
            'n_estimators': 50,
            'max_depth': 3,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': 0
        }

        self.model = XGBoostEnergyPredictor(self.config)

    def test_train_basic(self):
        """Test basic training functionality."""
        # Train the model
        self.model.train_basic(self.X_train, self.y_train)

        # Check if model is created
        self.assertIsNotNone(self.model.model, "Model should be created")

        # Check if feature names are stored
        self.assertEqual(self.model.feature_names, self.X_train.columns.tolist(), "Feature names should be stored")

        # Make predictions
        predictions = self.model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test), "Prediction length should match test data")

    def test_hyperparameter_optimization(self):
        """Test hyperparameter optimization."""
        # Only run a few evaluations for the test
        max_evals = 2

        # Optimize hyperparameters
        model, best_params = self.model.hyperparameter_optimization(
            self.X_train, self.y_train, self.X_test, self.y_test, max_evals=max_evals
        )

        # Check if model and best parameters are returned
        self.assertIsNotNone(model, "Model should be returned")
        self.assertIsNotNone(best_params, "Best parameters should be returned")

        # Check if best parameters are stored
        self.assertEqual(self.model.best_params, best_params, "Best parameters should be stored")

        # Make predictions
        predictions = self.model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test), "Prediction length should match test data")

    def test_evaluate(self):
        """Test model evaluation."""
        # Train the model
        self.model.train_basic(self.X_train, self.y_train)

        # Evaluate model
        metrics = self.model.evaluate(self.X_test, self.y_test)

        # Check if metrics are calculated
        self.assertIn('mse', metrics, "MSE should be calculated")
        self.assertIn('rmse', metrics, "RMSE should be calculated")
        self.assertIn('mae', metrics, "MAE should be calculated")
        self.assertIn('mape', metrics, "MAPE should be calculated")

        # Check if metrics are reasonable
        self.assertGreater(metrics['mse'], 0, "MSE should be positive")
        self.assertGreater(metrics['rmse'], 0, "RMSE should be positive")
        self.assertGreater(metrics['mae'], 0, "MAE should be positive")

    def test_feature_importance(self):
        """Test feature importance calculation."""
        # Train the model
        self.model.train_basic(self.X_train, self.y_train)

        # Get feature importance
        importance = self.model.feature_importance(plot=False)

        # Check if importance is calculated for all features
        self.assertEqual(len(importance), len(self.X_train.columns), "Importance should be calculated for all features")

        # Check if importance values are non-negative
        for feature, value in importance.items():
            self.assertGreaterEqual(value, 0, f"Importance for {feature} should be non-negative")

    def test_save_load_model(self):
        """Test model saving and loading."""
        # Train the model
        self.model.train_basic(self.X_train, self.y_train)

        # Make predictions before saving
        predictions_before = self.model.predict(self.X_test)

        # Create temporary file for saving
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as temp_file:
            model_path = temp_file.name

        try:
            # Save the model
            self.model.save_model(model_path)

            # Create a new model instance
            new_model = XGBoostEnergyPredictor(self.config)

            # Load the saved model
            new_model.load_model(model_path)

            # Make predictions with loaded model
            predictions_after = new_model.predict(self.X_test)

            # Check if predictions are the same
            np.testing.assert_array_almost_equal(
                predictions_before, predictions_after, decimal=6,
                err_msg="Predictions should be the same before and after loading"
            )

            # Check if feature names are loaded
            self.assertEqual(new_model.feature_names, self.model.feature_names, "Feature names should be loaded")

        finally:
            # Clean up
            if os.path.exists(model_path):
                os.remove(model_path)


class TestLSTMModel(unittest.TestCase):
    """Test cases for the LSTM model."""

    def setUp(self):
        """Set up test data."""
        # Create synthetic sequence data
        self.n_samples = 100
        self.sequence_length = 10
        self.n_features = 5

        # Create random sequences
        self.X_train = np.random.rand(80, self.sequence_length, self.n_features)
        self.y_train = np.random.rand(80)
        self.X_val = np.random.rand(10, self.sequence_length, self.n_features)
        self.y_val = np.random.rand(10)
        self.X_test = np.random.rand(10, self.sequence_length, self.n_features)
        self.y_test = np.random.rand(10)

        # Model configuration
        self.config = {
            'lstm_units': [32, 16],
            'dense_units': [8],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'epochs': 2,  # Just a few epochs for testing
            'batch_size': 16,
            'patience': 5
        }

        self.model = EnergyPredictionLSTM(self.config)

    def test_build_model(self):
        """Test building the model architecture."""
        # Build the model
        model = self.model.build_model(input_shape=(self.sequence_length, self.n_features))

        # Check if model is created
        self.assertIsNotNone(model, "Model should be created")
        self.assertIsNotNone(self.model.model, "Model should be stored in the instance")

        # Check input shape
        self.assertEqual(model.input_shape, (None, self.sequence_length, self.n_features), "Input shape should match")

        # Check output shape
        self.assertEqual(model.output_shape, (None, 1), "Output shape should be (None, 1)")

    def test_train(self):
        """Test model training."""
        # Build the model
        self.model.build_model(input_shape=(self.sequence_length, self.n_features))

        # Train the model
        history = self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)

        # Check if history is returned
        self.assertIsNotNone(history, "Training history should be returned")
        self.assertIsNotNone(self.model.history, "Training history should be stored")

        # Check history keys
        self.assertIn('loss', history.history, "Loss should be recorded")
        self.assertIn('val_loss', history.history, "Validation loss should be recorded")

        # Check number of epochs
        self.assertLessEqual(len(history.history['loss']), self.config['epochs'], "Number of epochs should not exceed config")

    def test_predict(self):
        """Test model prediction."""
        # Build and train the model
        self.model.build_model(input_shape=(self.sequence_length, self.n_features))
        self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)

        # Make predictions
        predictions = self.model.predict(self.X_test)

        # Check predictions shape
        self.assertEqual(predictions.shape, (len(self.X_test), 1), "Predictions shape should match")

    def test_evaluate(self):
        """Test model evaluation."""
        # Build and train the model
        self.model.build_model(input_shape=(self.sequence_length, self.n_features))
        self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)

        # Evaluate model
        metrics = self.model.evaluate(self.X_test, self.y_test)

        # Check if metrics are calculated
        self.assertIn('loss', metrics, "Loss should be calculated")
        self.assertGreater(metrics['loss'], 0, "Loss should be positive")

    def test_save_load_model(self):
        """Test model saving and loading."""
        # Build and train the model
        self.model.build_model(input_shape=(self.sequence_length, self.n_features))
        self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)

        # Make predictions before saving
        predictions_before = self.model.predict(self.X_test)

        # Create temporary file for saving
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
            model_path = temp_file.name

        try:
            # Save the model
            self.model.save_model(model_path)

            # Create a new model instance
            new_model = EnergyPredictionLSTM(self.config)

            # Load the saved model
            new_model.load_model(model_path)

            # Make predictions with loaded model
            predictions_after = new_model.predict(self.X_test)

            # Check if predictions are similar (not exact due to TensorFlow non-determinism)
            np.testing.assert_allclose(
                predictions_before, predictions_after, rtol=1e-5, atol=1e-5,
                err_msg="Predictions should be similar before and after loading"
            )

        finally:
            # Clean up
            if os.path.exists(model_path):
                os.remove(model_path)


class TestAttentionLSTM(unittest.TestCase):
    """Test cases for the AttentionLSTM model."""

    def setUp(self):
        """Set up test data."""
        # Create synthetic sequence data
        self.n_samples = 100
        self.sequence_length = 10
        self.n_features = 5

        # Create random sequences
        self.X_train = np.random.rand(80, self.sequence_length, self.n_features)
        self.y_train = np.random.rand(80)
        self.X_val = np.random.rand(10, self.sequence_length, self.n_features)
        self.y_val = np.random.rand(10)
        self.X_test = np.random.rand(10, self.sequence_length, self.n_features)
        self.y_test = np.random.rand(10)

        # Model configuration
        self.config = {
            'lstm_units': [32, 16],
            'dense_units': [8],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'epochs': 2,  # Just a few epochs for testing
            'batch_size': 16,
            'patience': 5
        }

        self.model = AttentionLSTM(self.config)

    def test_build_model(self):
        """Test building the model architecture with attention."""
        # Build the model
        model = self.model.build_model(input_shape=(self.sequence_length, self.n_features))

        # Check if model is created
        self.assertIsNotNone(model, "Model should be created")
        self.assertIsNotNone(self.model.model, "Model should be stored in the instance")

        # Check input shape
        self.assertEqual(model.input_shape, (None, self.sequence_length, self.n_features), "Input shape should match")

        # Check output shape
        self.assertEqual(model.output_shape, (None, 1), "Output shape should be (None, 1)")

        # Check if model contains attention layer (check layer types)
        has_attention = any('attention' in layer.name.lower() for layer in model.layers)
        self.assertTrue(has_attention, "Model should contain attention mechanism")

    def test_train_and_predict(self):
        """Test model training and prediction with attention."""
        # Build the model
        self.model.build_model(input_shape=(self.sequence_length, self.n_features))

        # Train the model
        history = self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)

        # Check if history is returned
        self.assertIsNotNone(history, "Training history should be returned")

        # Make predictions
        predictions = self.model.predict(self.X_test)

        # Check predictions shape
        self.assertEqual(predictions.shape, (len(self.X_test), 1), "Predictions shape should match")


class TestEnsembleModel(unittest.TestCase):
    """Test cases for the Ensemble model."""

    def setUp(self):
        """Set up test data and models."""
        # Create synthetic data
        X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
        self.X_test = X[:20]
        self.y_test = y[:20]

        # Create mock models
        class MockModel1:
            def predict(self, X):
                return np.array([1.0] * len(X))

        class MockModel2:
            def predict(self, X):
                return np.array([2.0] * len(X))

        self.models = [MockModel1(), MockModel2()]

        # Initialize ensemble
        self.ensemble = EnergyPredictionEnsemble(self.models, weights=[0.3, 0.7])

    def test_predict(self):
        """Test ensemble prediction."""
        # Make predictions
        predictions = self.ensemble.predict(self.X_test)

        # Check predictions shape
        self.assertEqual(len(predictions), len(self.X_test), "Prediction length should match test data")

        # Check weighted average
        expected_pred = 0.3 * 1.0 + 0.7 * 2.0
        for pred in predictions:
            self.assertAlmostEqual(pred, expected_pred, delta=1e-6, msg="Prediction should match weighted average")

    def test_evaluate(self):
        """Test ensemble evaluation."""
        # Set the output to match the test target for predictable metrics
        class MockModelMatchTarget:
            def __init__(self, y):
                self.y = y

            def predict(self, X):
                return self.y

        # Create new ensemble with a perfect predictor
        perfect_models = [MockModelMatchTarget(self.y_test)]
        perfect_ensemble = EnergyPredictionEnsemble(perfect_models)

        # Evaluate model
        metrics = perfect_ensemble.evaluate(self.X_test, self.y_test)

        # Check if metrics are calculated
        self.assertIn('mse', metrics, "MSE should be calculated")
        self.assertIn('rmse', metrics, "RMSE should be calculated")
        self.assertIn('mae', metrics, "MAE should be calculated")
        self.assertIn('mape', metrics, "MAPE should be calculated")

        # Check if metrics are correct (should be very close to zero for perfect prediction)
        self.assertAlmostEqual(metrics['mse'], 0.0, delta=1e-10, msg="MSE should be zero for perfect prediction")
        self.assertAlmostEqual(metrics['rmse'], 0.0, delta=1e-10, msg="RMSE should be zero for perfect prediction")
        self.assertAlmostEqual(metrics['mae'], 0.0, delta=1e-10, msg="MAE should be zero for perfect prediction")

    def test_optimize_weights(self):
        """Test weight optimization."""
        # Create synthetic data with known optimal weights
        X = np.random.rand(50, 5)
        y = np.random.rand(50)

        # Create mock models with known prediction errors
        class MockModelA:
            def predict(self, X):
                return y + 0.1 * np.random.randn(len(X))  # Good predictor

        class MockModelB:
            def predict(self, X):
                return y + 0.5 * np.random.randn(len(X))  # Less good predictor

        # Create ensemble with mock models
        models = [MockModelA(), MockModelB()]
        ensemble = EnergyPredictionEnsemble(models, weights=[0.5, 0.5])

        # Optimize weights
        best_weights = ensemble.optimize_weights(X, y)

        # Check if weights are optimized (the better model should have higher weight)
        self.assertGreater(ensemble.weights[0], ensemble.weights[1],
                          "The better model should have higher weight after optimization")

        # Check if weights sum to 1
        self.assertAlmostEqual(sum(ensemble.weights), 1.0, delta=1e-6,
                           #  "Weights should sum to 1 after optimization")
        )



if __name__ == '__main__':
    unittest.main()

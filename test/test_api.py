"""
Tests for the API service.
"""

import os
import sys
import unittest
import json
import numpy as np
from fastapi.testclient import TestClient
from unittest import mock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the API
from src.api.main import app


class TestAPIEndpoints(unittest.TestCase):
    """Test cases for the API endpoints."""

    def setUp(self):
        """Set up test client and mocks."""
        self.client = TestClient(app)

        # Create patch for the preprocessors
        self.mock_preprocessors = mock.patch('src.api.main.preprocessors', {
            'scaler': mock.MagicMock(),
            'encoder': mock.MagicMock()
        }).start()

        # Create patch for the feature config
        self.mock_feature_config = mock.patch('src.api.main.feature_config', {
            'xgb_features': ['speed', 'temperature', 'weight'],
            'lstm_features': ['speed', 'temperature', 'weight'],
            'numeric_features': ['speed', 'temperature', 'weight'],
            'categorical_features': ['road_type'],
            'sequence_length': 10
        }).start()

        # Create patch for the models
        mock_xgb = mock.MagicMock()
        mock_xgb.predict.return_value = np.array([15.0])

        mock_lstm = mock.MagicMock()
        mock_lstm.predict.return_value = np.array([[14.0]])

        self.mock_models = mock.patch('src.api.main.models', {
            'xgboost': mock_xgb,
            'lstm': mock_lstm
        }).start()

        # Create patch for the config
        self.mock_config = mock.patch('src.api.main.config', {
            'prediction': {
                'ensemble_weights': {
                    'xgboost': 0.6,
                    'lstm': 0.4
                },
                'sequence_length': 10
            }
        }).start()

        # Sample valid request data
        self.valid_request = {
            "vehicle": {
                "vehicle_id": "model_y_2022",
                "weight": 2100.0,
                "drag_coefficient": 0.23,
                "frontal_area": 2.7,
                "battery_capacity": 75.0,
                "rolling_resistance": 0.01
            },
            "driving_conditions": [
                {
                    "timestamp": "2023-05-15T14:30:00Z",
                    "speed": 105.0,
                    "altitude": 150.0,
                    "temperature": 22.5,
                    "wind_speed": 15.0,
                    "precipitation": 0.0,
                    "road_type": "highway"
                },
                {
                    "timestamp": "2023-05-15T14:31:00Z",
                    "speed": 110.0,
                    "altitude": 155.0,
                    "temperature": 22.5,
                    "wind_speed": 15.0,
                    "precipitation": 0.0,
                    "road_type": "highway"
                }
            ]
        }

    def tearDown(self):
        """Clean up mocks."""
        mock.patch.stopall()

    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")

        # Check status code
        self.assertEqual(response.status_code, 200, "Health check should return 200")

        # Check response content
        data = response.json()
        self.assertEqual(data["status"], "healthy", "Health status should be 'healthy'")
        self.assertIn("models", data, "Response should include models info")
        self.assertIn("request_count", data, "Response should include request count")

    def test_vehicle_types_endpoint(self):
        """Test the vehicle types endpoint."""
        response = self.client.get("/api/v1/vehicle-types")

        # Check status code
        self.assertEqual(response.status_code, 200, "Endpoint should return 200")

        # Check response content
        data = response.json()
        self.assertIn("vehicles", data, "Response should include vehicles list")
        self.assertGreater(len(data["vehicles"]), 0, "Vehicles list should not be empty")

        # Check vehicle properties
        first_vehicle = data["vehicles"][0]
        required_props = ["id", "name", "weight", "drag_coefficient", "frontal_area", "battery_capacity"]
        for prop in required_props:
            self.assertIn(prop, first_vehicle, f"Vehicle should have {prop} property")

    def test_road_types_endpoint(self):
        """Test the road types endpoint."""
        response = self.client.get("/api/v1/road-types")

        # Check status code
        self.assertEqual(response.status_code, 200, "Endpoint should return 200")

        # Check response content
        data = response.json()
        self.assertIn("road_types", data, "Response should include road_types list")
        self.assertGreater(len(data["road_types"]), 0, "Road types list should not be empty")

        # Check road type properties
        first_road_type = data["road_types"][0]
        required_props = ["id", "name", "description"]
        for prop in required_props:
            self.assertIn(prop, first_road_type, f"Road type should have {prop} property")

    def test_predict_endpoint_with_valid_data(self):
        """Test the prediction endpoint with valid data."""
        # Mock preprocess_data function to return test data
        with mock.patch('src.api.main.preprocess_data') as mock_preprocess:
            # Set up the mock to return test data
            mock_preprocess.return_value = (
                mock.MagicMock(),  # XGBoost input
                mock.MagicMock(),  # LSTM input
                pd.DataFrame({
                    'timestamp': pd.to_datetime(['2023-05-15T14:30:00Z', '2023-05-15T14:31:00Z']),
                    'speed': [105.0, 110.0],
                    'energy_requirement': [0.2, 0.25]
                })
            )

            # Send the request
            response = self.client.post(
                "/api/v1/predict",
                json=self.valid_request
            )

            # Check status code
            self.assertEqual(response.status_code, 200, "Valid request should return 200")

            # Check response content
            data = response.json()
            self.assertIn("energy_consumption", data, "Response should include energy_consumption")
            self.assertIn("remaining_range", data, "Response should include remaining_range")
            self.assertIn("predictions", data, "Response should include predictions")
            self.assertIn("model_used", data, "Response should include model_used")

            # Check ensemble prediction (0.6 * 15.0 + 0.4 * 14.0 = 14.6)
            self.assertAlmostEqual(data["energy_consumption"], 14.6, places=1,
                                "Energy consumption should match expected ensemble prediction")

            # Check predictions array
            self.assertEqual(len(data["predictions"]), 2, "Should have 2 prediction points")
            self.assertIn("timestamp", data["predictions"][0], "Prediction should include timestamp")
            self.assertIn("speed", data["predictions"][0], "Prediction should include speed")
            self.assertIn("energy_rate", data["predictions"][0], "Prediction should include energy_rate")

    def test_predict_endpoint_with_specific_model(self):
        """Test the prediction endpoint with a specific model type."""
        # Mock preprocess_data function to return test data
        with mock.patch('src.api.main.preprocess_data') as mock_preprocess:
            mock_preprocess.return_value = (
                mock.MagicMock(),  # XGBoost input
                mock.MagicMock(),  # LSTM input
                pd.DataFrame({
                    'timestamp': pd.to_datetime(['2023-05-15T14:30:00Z', '2023-05-15T14:31:00Z']),
                    'speed': [105.0, 110.0],
                    'energy_requirement': [0.2, 0.25]
                })
            )

            # Request XGBoost model specifically
            response = self.client.post(
                "/api/v1/predict?model_type=xgboost",
                json=self.valid_request
            )

            # Check status code
            self.assertEqual(response.status_code, 200, "Valid request should return 200")

            # Check response content
            data = response.json()
            self.assertEqual(data["model_used"], "xgboost", "Model used should be xgboost")
            self.assertAlmostEqual(data["energy_consumption"], 15.0, places=1,
                                "Energy consumption should match XGBoost prediction")

    def test_predict_endpoint_with_invalid_data(self):
        """Test the prediction endpoint with invalid data."""
        # Invalid request with missing required fields
        invalid_request = {
            "vehicle": {
                "vehicle_id": "model_y_2022",
                # Missing required fields
            },
            "driving_conditions": [
                {
                    "timestamp": "2023-05-15T14:30:00Z",
                    # Missing required fields
                }
            ]
        }

        # Send the request
        response = self.client.post(
            "/api/v1/predict",
            json=invalid_request
        )

        # Check status code (should be 422 Unprocessable Entity)
        self.assertEqual(response.status_code, 422, "Invalid request should return 422")

    def test_predict_endpoint_with_invalid_model_type(self):
        """Test the prediction endpoint with an invalid model type."""
        # Send request with invalid model type
        response = self.client.post(
            "/api/v1/predict?model_type=invalid_model",
            json=self.valid_request
        )

        # Check status code (should be 400 Bad Request)
        self.assertEqual(response.status_code, 400, "Invalid model type should return 400")

        # Check error message
        data = response.json()
        self.assertIn("detail", data, "Response should include error detail")
        self.assertIn("Invalid model_type", data["detail"], "Error should mention invalid model type")

    def test_metrics_endpoint(self):
        """Test the metrics endpoint."""
        response = self.client.get("/metrics")

        # Check status code
        self.assertEqual(response.status_code, 200, "Metrics endpoint should return 200")

        # Check response content
        data = response.json()
        self.assertIn("total_requests", data, "Response should include total_requests")
        self.assertIn("error_count", data, "Response should include error_count")
        self.assertIn("success_rate", data, "Response should include success_rate")
        self.assertIn("models_loaded", data, "Response should include models_loaded")


# Mocking pandas for API tests
class MockDataFrame:
    """Mock pandas DataFrame for testing."""

    def __init__(self, data=None):
        self.data = data or {}
        self.shape = (len(data.get('timestamp', [])), len(data))
        self.columns = list(data.keys())
        self.index = list(range(len(data.get('timestamp', []))))

    def copy(self):
        return MockDataFrame(self.data)

    def sort_values(self, by):
        return self

    def groupby(self, by):
        return self

    def diff(self):
        return self

    def fillna(self, value):
        return self

    def mean(self):
        return 107.5  # Average of [105.0, 110.0]

    def tolist(self):
        return list(range(self.shape[0]))

    def iloc(self, idx):
        if isinstance(idx, slice):
            return self
        return self

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        return np.zeros(self.shape[0])

    def values(self):
        return np.zeros(self.shape)


# Mock pandas module
pd = mock.MagicMock()
pd.DataFrame = MockDataFrame
pd.to_datetime = lambda x: x
pd.api = mock.MagicMock()
pd.api.types = mock.MagicMock()
pd.api.types.is_datetime64_dtype = lambda x: False

# Apply mock
import sys
sys.modules['pandas'] = pd


if __name__ == '__main__':
    unittest.main()

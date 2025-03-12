"""
Prediction endpoint implementation for the Energy Prediction API.

This module contains the core prediction logic and handlers for the API endpoints.
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Any

from fastapi import HTTPException, BackgroundTasks
from .schemas import (
    PredictionRequest, PredictionResponse, PredictionResult,
    ConfidenceInterval, VehicleTypeResponse, RoadTypeResponse
)

# Configure logging
logger = logging.getLogger(__name__)


def preprocess_prediction_data(data: PredictionRequest) -> Tuple[Any, Any, pd.DataFrame]:
    """
    Preprocess prediction request data for model input.

    Args:
        data: Prediction request data

    Returns:
        Tuple of (XGBoost input, LSTM input, processed DataFrame)

    Raises:
        HTTPException: If preprocessing fails
    """
    try:
        # Convert to DataFrame
        vehicle_df = pd.DataFrame([data.vehicle.dict()])
        conditions_df = pd.DataFrame([c.dict() for c in data.driving_conditions])

        # Convert timestamps to datetime
        conditions_df['timestamp'] = pd.to_datetime(conditions_df['timestamp'])
        conditions_df = conditions_df.sort_values('timestamp')

        # Calculate missing features
        if 'acceleration' not in conditions_df or conditions_df['acceleration'].isnull().any():
            # Calculate acceleration from speed changes
            conditions_df['acceleration'] = conditions_df['speed'].diff() / \
                                          conditions_df['timestamp'].diff().dt.total_seconds()
            conditions_df['acceleration'] = conditions_df['acceleration'].fillna(0)

        # Calculate elevation changes
        conditions_df['elevation_change'] = conditions_df['altitude'].diff().fillna(0)

        # Add time-based features
        conditions_df['hour_of_day'] = conditions_df['timestamp'].dt.hour
        conditions_df['day_of_week'] = conditions_df['timestamp'].dt.dayofweek

        # Add vehicle data to each condition row
        for col in vehicle_df.columns:
            conditions_df[col] = vehicle_df[col].values[0]

        # Calculate derived features
        # Air density approximation based on temperature and standard pressure
        conditions_df['air_density'] = 1.225 * (288.15 / (conditions_df['temperature'] + 273.15))

        # Basic energy requirement calculation (simplified physics)
        conditions_df['energy_requirement'] = (
            # Rolling resistance
            conditions_df['rolling_resistance'] * conditions_df['weight'] * \
            conditions_df['speed'] +
            # Air resistance
            0.5 * conditions_df['air_density'] * conditions_df['drag_coefficient'] * \
            conditions_df['frontal_area'] * (conditions_df['speed'] / 3.6) ** 2 +
            # Acceleration energy
            conditions_df['weight'] * conditions_df['acceleration'] * \
            (conditions_df['speed'] / 3.6) +
            # Elevation energy
            conditions_df['weight'] * 9.81 * conditions_df['elevation_change']
        ) / 3600  # Convert to kWh

        # Prepare inputs for different models
        # Note: In a real implementation, you would use the preprocessors from the trained models
        # For simplicity, we'll return simple mock inputs

        # Mock XGBoost input (last row of features)
        xgb_input = conditions_df.iloc[-1:].copy()

        # Mock LSTM input (sequence data)
        sequence_length = min(10, len(conditions_df))
        lstm_input = conditions_df.iloc[-sequence_length:].copy()
        lstm_input = np.expand_dims(lstm_input.values, axis=0)

        return xgb_input, lstm_input, conditions_df

    except Exception as e:
        logger.error(f"Error preprocessing prediction data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error preprocessing prediction data: {str(e)}"
        )


def generate_predictions(
    xgb_input: Any,
    lstm_input: Any,
    processed_df: pd.DataFrame,
    model_type: str,
    models: Dict[str, Any],
    config: Dict[str, Any]
) -> PredictionResponse:
    """
    Generate energy consumption predictions using specified model(s).

    Args:
        xgb_input: Preprocessed input for XGBoost model
        lstm_input: Preprocessed input for LSTM model
        processed_df: Processed DataFrame with conditions
        model_type: Type of model to use (xgboost, lstm, or ensemble)
        models: Dictionary of loaded models
        config: API configuration

    Returns:
        Prediction response

    Raises:
        HTTPException: If prediction fails
    """
    try:
        # Check if requested models are available
        available_models = list(models.keys())
        if not available_models:
            raise HTTPException(
                status_code=503,
                detail="No prediction models are currently available"
            )

        # Validate model type
        if model_type not in ["xgboost", "lstm", "ensemble"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_type. Must be one of: xgboost, lstm, ensemble"
            )

        # For ensemble, check if both models are available
        if model_type == "ensemble" and (
            "xgboost" not in available_models or "lstm" not in available_models
        ):
            logger.warning(f"Ensemble requested but not all models available. Using {available_models[0]}")
            model_type = available_models[0]

        # If specific model requested but not available
        if model_type != "ensemble" and model_type not in available_models:
            raise HTTPException(
                status_code=400,
                detail=f"Requested model '{model_type}' is not available. Available models: {available_models}"
            )

        # Generate predictions based on model type
        if model_type == "xgboost" or model_type == "ensemble":
            xgb_model = models["xgboost"]
            xgb_pred = xgb_model.predict(xgb_input)[0]
        else:
            xgb_pred = 0

        if model_type == "lstm" or model_type == "ensemble":
            lstm_model = models["lstm"]
            lstm_pred = lstm_model.predict(lstm_input)[0][0]
        else:
            lstm_pred = 0

        # Calculate final prediction
        if model_type == "ensemble":
            # Get ensemble weights from config
            weights = config.get("prediction", {}).get(
                "ensemble_weights", {"xgboost": 0.6, "lstm": 0.4}
            )
            xgb_weight = weights.get("xgboost", 0.6)
            lstm_weight = weights.get("lstm", 0.4)

            # Weighted average
            energy_consumption = xgb_weight * xgb_pred + lstm_weight * lstm_pred
            model_used = f"ensemble (xgb: {xgb_weight:.2f}, lstm: {lstm_weight:.2f})"
        elif model_type == "xgboost":
            energy_consumption = xgb_pred
            model_used = "xgboost"
        else:  # lstm
            energy_consumption = lstm_pred
            model_used = "lstm"

        # Calculate confidence interval (simplified)
        confidence = None
        if model_type == "ensemble":
            # Use difference between models as a rough uncertainty estimate
            difference = abs(xgb_pred - lstm_pred)
            confidence = ConfidenceInterval(
                lower_bound=max(0, energy_consumption - difference / 2),
                upper_bound=energy_consumption + difference / 2,
                uncertainty=difference / energy_consumption if energy_consumption > 0 else 0
            )

        # Calculate remaining range
        battery_capacity = processed_df['battery_capacity'].iloc[0]
        avg_speed = processed_df['speed'].mean()

        # Simple range calculation: remaining capacity / consumption rate * average speed
        if energy_consumption > 0 and avg_speed > 0:
            # Energy per km
            energy_per_km = energy_consumption / (avg_speed * len(processed_df) / 3600)
            remaining_range = (battery_capacity - energy_consumption) / energy_per_km
        else:
            remaining_range = 0

        # Prepare detailed predictions for each time step
        timestamps = processed_df['timestamp'].tolist()
        speeds = processed_df['speed'].tolist()

        # Calculate energy allocation per time step (proportional to energy requirements if available)
        if 'energy_requirement' in processed_df.columns:
            # Normalize energy requirements
            energy_reqs = processed_df['energy_requirement'].values
            total_req = energy_reqs.sum()

            if total_req > 0:
                # Allocate total energy consumption proportionally
                energy_rates = energy_reqs * (energy_consumption / total_req)
            else:
                # Equal distribution if requirements are zero
                energy_rates = np.ones_like(energy_reqs) * (energy_consumption / len(energy_reqs))
        else:
            # Equal distribution if no energy requirements calculated
            energy_rates = np.ones(len(timestamps)) * (energy_consumption / len(timestamps))

        # Calculate cumulative energy consumption
        cumulative_energy = np.cumsum(energy_rates)

        # Create detailed predictions
        detailed_predictions = []
        for i in range(len(timestamps)):
            prediction = PredictionResult(
                timestamp=timestamps[i].isoformat(),
                speed=float(speeds[i]),
                energy_rate=float(energy_rates[i]),
                cumulative_energy=float(cumulative_energy[i])
            )
            detailed_predictions.append(prediction)

        # Create response
        response = PredictionResponse(
            energy_consumption=float(energy_consumption),
            remaining_range=float(remaining_range),
            predictions=detailed_predictions,
            confidence_interval=confidence,
            model_used=model_used
        )

        return response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating predictions: {str(e)}"
        )


def get_vehicle_types() -> VehicleTypeResponse:
    """
    Get available vehicle types with default parameters.

    Returns:
        Response with vehicle type information
    """
    # Define default vehicle types
    vehicle_types = [
        {
            "id": "model_3",
            "name": "Model 3",
            "weight": 1844.0,
            "drag_coefficient": 0.23,
            "frontal_area": 2.22,
            "battery_capacity": 75.0,
            "rolling_resistance": 0.01
        },
        {
            "id": "model_y",
            "name": "Model Y",
            "weight": 2100.0,
            "drag_coefficient": 0.23,
            "frontal_area": 2.66,
            "battery_capacity": 82.0,
            "rolling_resistance": 0.01
        },
        {
            "id": "model_s",
            "name": "Model S",
            "weight": 2215.0,
            "drag_coefficient": 0.208,
            "frontal_area": 2.34,
            "battery_capacity": 100.0,
            "rolling_resistance": 0.01
        },
        {
            "id": "model_x",
            "name": "Model X",
            "weight": 2487.0,
            "drag_coefficient": 0.24,
            "frontal_area": 2.72,
            "battery_capacity": 100.0,
            "rolling_resistance": 0.01
        }
    ]

    # Create response
    return VehicleTypeResponse(vehicles=vehicle_types)


def get_road_types() -> RoadTypeResponse:
    """
    Get available road types with descriptions.

    Returns:
        Response with road type information
    """
    # Define road types
    road_types = [
        {
            "id": "highway",
            "name": "Highway",
            "description": "High-speed road with limited access"
        },
        {
            "id": "urban",
            "name": "Urban",
            "description": "City roads with frequent stops and lower speeds"
        },
        {
            "id": "rural",
            "name": "Rural",
            "description": "Country roads with moderate speeds"
        },
        {
            "id": "mountain",
            "name": "Mountain",
            "description": "Roads with significant elevation changes"
        }
    ]

    # Create response
    return RoadTypeResponse(road_types=road_types)


def log_prediction(
    background_tasks: BackgroundTasks,
    request_data: PredictionRequest,
    response_data: Optional[PredictionResponse] = None,
    error: Optional[str] = None
) -> None:
    """
    Log prediction request and response for monitoring.

    Args:
        background_tasks: Background tasks runner
        request_data: Prediction request data
        response_data: Optional prediction response data
        error: Optional error message
    """
    def _log_prediction():
        # Get current timestamp
        timestamp = datetime.now().isoformat()

        # Create log entry
        log_entry = {
            "timestamp": timestamp,
            "request": request_data.dict(),
            "response": response_data.dict() if response_data else None,
            "error": error,
            "success": error is None
        }

        # In a real implementation, you would save this to a database or log file
        # For simplicity, we just log it
        if error:
            logger.error(f"Prediction error: {error}")
        else:
            logger.info(f"Successful prediction: {response_data.energy_consumption:.2f} kWh")

        # You could also send metrics to a monitoring system
        # For example, increment counters for successful/failed predictions

    # Add logging task to background tasks
    background_tasks.add_task(_log_prediction)

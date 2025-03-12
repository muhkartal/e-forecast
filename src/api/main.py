"""
FastAPI application for the Energy Prediction System.

This module provides a REST API for making real-time energy consumption
predictions using the trained machine learning models.
"""

import os
import json
import yaml
import logging
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("energy_prediction_api")


# Load configuration
def load_config():
    """Load application configuration from YAML file."""
    config_path = os.environ.get("CONFIG_PATH", "configs/api_config.yaml")
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        # Return default configuration
        return {
            "model_paths": {
                "xgboost": "models/trained/xgboost_model.joblib",
                "lstm": "models/trained/lstm_model.h5"
            },
            "preprocessing": {
                "scaler_path": "models/trained/feature_scaler.joblib",
                "encoder_path": "models/trained/categorical_encoder.joblib",
                "feature_config_path": "configs/feature_config.json"
            },
            "prediction": {
                "ensemble_weights": {
                    "xgboost": 0.6,
                    "lstm": 0.4
                },
                "sequence_length": 60
            }
        }


# API models
class VehicleData(BaseModel):
    """Data model for vehicle information."""
    vehicle_id: str = Field(..., description="Unique identifier for the vehicle")
    weight: float = Field(..., description="Vehicle weight in kg")
    drag_coefficient: float = Field(..., description="Aerodynamic drag coefficient")
    frontal_area: float = Field(..., description="Frontal area in m²")
    battery_capacity: float = Field(..., description="Battery capacity in kWh")
    rolling_resistance: Optional[float] = Field(0.01, description="Rolling resistance coefficient")
    wheel_radius: Optional[float] = Field(0.35, description="Wheel radius in meters")

    @validator('weight')
    def weight_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Weight must be positive')
        return v

    @validator('battery_capacity')
    def battery_capacity_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Battery capacity must be positive')
        return v


class DrivingCondition(BaseModel):
    """Data model for a single driving condition time point."""
    timestamp: str = Field(..., description="ISO format timestamp")
    speed: float = Field(..., description="Vehicle speed in km/h")
    acceleration: Optional[float] = Field(None, description="Acceleration in m/s²")
    altitude: float = Field(..., description="Altitude above sea level in meters")
    temperature: float = Field(..., description="Ambient temperature in Celsius")
    wind_speed: Optional[float] = Field(0.0, description="Wind speed in km/h")
    precipitation: Optional[float] = Field(0.0, description="Precipitation in mm")
    road_type: Optional[str] = Field("highway", description="Type of road (highway, urban, rural)")

    @validator('speed')
    def speed_must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError('Speed cannot be negative')
        return v


class PredictionRequest(BaseModel):
    """Data model for a prediction request."""
    vehicle: VehicleData
    driving_conditions: List[DrivingCondition] = Field(..., min_items=1)


class PredictionResult(BaseModel):
    """Data model for a single time point prediction result."""
    timestamp: str
    speed: float
    energy_rate: float
    cumulative_energy: float


class PredictionResponse(BaseModel):
    """Data model for the prediction response."""
    energy_consumption: float = Field(..., description="Total predicted energy consumption in kWh")
    remaining_range: float = Field(..., description="Estimated remaining range in km")
    predictions: List[PredictionResult] = Field(..., description="Detailed predictions for each time step")
    confidence_interval: Optional[Dict[str, float]] = Field(None, description="Confidence interval for the prediction")
    model_used: str = Field(..., description="Model used for the prediction")


# Create the FastAPI application
app = FastAPI(
    title="Energy Prediction API",
    description="API for predicting energy consumption in electric vehicles",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and preprocessors
models = {}
preprocessors = {}
feature_config = {}
config = {}
prediction_history = []
request_count = 0
error_count = 0


# Function to preprocess prediction data
def preprocess_data(data: PredictionRequest) -> tuple:
    """
    Preprocess the input data for model prediction.

    Args:
        data: Prediction request data

    Returns:
        Tuple containing preprocessed data for XGBoost and LSTM models
    """
    logger.debug("Preprocessing prediction data")

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
        conditions_df['rolling_resistance'] * conditions_df['weight'] * conditions_df['speed'] +
        # Air resistance
        0.5 * conditions_df['air_density'] * conditions_df['drag_coefficient'] *
        conditions_df['frontal_area'] * (conditions_df['speed'] / 3.6) ** 2 +
        # Acceleration energy
        conditions_df['weight'] * conditions_df['acceleration'] * (conditions_df['speed'] / 3.6) +
        # Elevation energy
        conditions_df['weight'] * 9.81 * conditions_df['elevation_change']
    ) / 3600  # Convert to kWh

    # Encode categorical features
    categorical_cols = [col for col in feature_config.get('categorical_features', [])
                       if col in conditions_df.columns]

    if categorical_cols and 'encoder' in preprocessors:
        # Use one-hot encoding for categorical features
        encoded_cats = pd.get_dummies(conditions_df[categorical_cols], prefix=categorical_cols)
        conditions_df = pd.concat([conditions_df.drop(categorical_cols, axis=1), encoded_cats], axis=1)

    # Scale numerical features
    numerical_cols = [col for col in feature_config.get('numeric_features', [])
                     if col in conditions_df.columns]

    if numerical_cols and 'scaler' in preprocessors:
        scaler = preprocessors['scaler']
        conditions_df[numerical_cols] = scaler.transform(conditions_df[numerical_cols])

    # Prepare inputs for different models
    # XGBoost input: aggregated features
    xgb_features = feature_config.get('xgb_features', conditions_df.columns.tolist())
    xgb_features = [f for f in xgb_features if f in conditions_df.columns]

    # Use the most recent data point for XGBoost
    xgb_input = conditions_df.iloc[-1:][xgb_features]

    # LSTM input: sequence of conditions
    lstm_features = feature_config.get('lstm_features', conditions_df.columns.tolist())
    lstm_features = [f for f in lstm_features if f in conditions_df.columns]

    sequence_length = config.get('prediction', {}).get('sequence_length', 60)

    if len(conditions_df) >= sequence_length:
        # Use the most recent sequence
        sequence = conditions_df[-sequence_length:][lstm_features].values
    else:
        # Pad with zeros if we don't have enough data
        padding_needed = sequence_length - len(conditions_df)
        padded_sequence = np.vstack([
            np.zeros((padding_needed, len(lstm_features))),
            conditions_df[lstm_features].values
        ])
        sequence = padded_sequence

    # Reshape for LSTM (samples, time steps, features)
    lstm_input = np.expand_dims(sequence, axis=0)

    return xgb_input, lstm_input, conditions_df


@app.on_event("startup")
async def startup_event():
    """Load models and preprocessors on startup."""
    global models, preprocessors, feature_config, config

    try:
        # Load application configuration
        config = load_config()
        logger.info("Loaded application configuration")

        # Load XGBoost model
        xgb_path = config["model_paths"]["xgboost"]
        logger.info(f"Loading XGBoost model from {xgb_path}")
        xgb_package = joblib.load(xgb_path)
        models["xgboost"] = xgb_package.get("model", xgb_package)

        # Load LSTM model
        lstm_path = config["model_paths"]["lstm"]
        logger.info(f"Loading LSTM model from {lstm_path}")
        models["lstm"] = tf.keras.models.load_model(lstm_path)

        # Load preprocessors
        scaler_path = config["preprocessing"]["scaler_path"]
        encoder_path = config["preprocessing"]["encoder_path"]
        feature_config_path = config["preprocessing"]["feature_config_path"]

        logger.info(f"Loading preprocessors from {scaler_path} and {encoder_path}")
        preprocessors["scaler"] = joblib.load(scaler_path)
        preprocessors["encoder"] = joblib.load(encoder_path)

        # Load feature configuration
        with open(feature_config_path, "r") as f:
            feature_config = json.load(f)

        logger.info("Startup completed successfully. API is ready.")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        # Continue without failing - we'll handle missing models in the endpoints
        logger.warning("API starting with limited functionality due to initialization errors")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models": {
            "xgboost": "xgboost" in models,
            "lstm": "lstm" in models
        },
        "request_count": request_count,
        "error_count": error_count,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/metrics")
async def metrics():
    """Get API usage metrics."""
    return {
        "total_requests": request_count,
        "error_count": error_count,
        "success_rate": (request_count - error_count) / request_count if request_count > 0 else 1.0,
        "models_loaded": list(models.keys()),
        "recent_history_size": len(prediction_history)
    }


def log_request(background_tasks: BackgroundTasks, request_data: Any, response_data: Any = None, error: str = None):
    """Log request and response data for monitoring."""
    global request_count, error_count

    def _log():
        global request_count, error_count
        timestamp = datetime.now().isoformat()

        if error:
            error_count += 1
            logger.error(f"Request error: {error}")

        request_count += 1

        # Add to history (limited size)
        prediction_history.append({
            "timestamp": timestamp,
            "request_data": request_data.dict() if hasattr(request_data, "dict") else str(request_data),
            "response_data": response_data,
            "error": error
        })

        # Keep history limited to last 100 requests
        if len(prediction_history) > 100:
            prediction_history.pop(0)

    background_tasks.add_task(_log)


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_energy_consumption(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    model_type: Optional[str] = Query("ensemble", description="Model to use (xgboost, lstm, or ensemble)")
):
    """
    Predict energy consumption based on vehicle and driving conditions.

    This endpoint receives vehicle specifications and driving conditions,
    then returns predicted energy consumption and range estimation.
    """
    try:
        # Check if requested models are available
        available_models = list(models.keys())
        if not available_models:
            raise HTTPException(
                status_code=503,
                detail="No prediction models are currently available"
            )

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

        # Preprocess input data
        xgb_input, lstm_input, processed_df = preprocess_data(request)

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
            weights = config.get("prediction", {}).get("ensemble_weights", {"xgboost": 0.6, "lstm": 0.4})
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
            confidence = {
                "lower_bound": max(0, energy_consumption - difference / 2),
                "upper_bound": energy_consumption + difference / 2,
                "uncertainty": difference / energy_consumption if energy_consumption > 0 else 0
            }

        # Calculate remaining range
        battery_capacity = request.vehicle.battery_capacity
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

        # Log successful request
        log_request(background_tasks, request, response.dict())

        return response

    except HTTPException as e:
        # Log HTTP exceptions without modification
        log_request(background_tasks, request, error=f"{e.status_code}: {e.detail}")
        raise

    except Exception as e:
        # Log unexpected errors
        error_msg = f"Prediction error: {str(e)}"
        log_request(background_tasks, request, error=error_msg)
        logger.exception("Unexpected error during prediction")

        raise HTTPException(
            status_code=500,
            detail=error_msg
        )


@app.get("/api/v1/vehicle-types")
async def get_vehicle_types():
    """Get available vehicle types with default parameters."""
    return {
        "vehicles": [
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
    }


@app.get("/api/v1/road-types")
async def get_road_types():
    """Get available road types with descriptions."""
    return {
        "road_types": [
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
    }


# Run application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

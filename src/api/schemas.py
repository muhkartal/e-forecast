"""
Pydantic models for the sEnergy Prediction API.

This module contains the request and response data models used by the API endpoints.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union, Any
from datetime import datetime


class VehicleData(BaseModel):
    """
    Model representing vehicle specification data.
    """
    vehicle_id: str = Field(...,
                         description="Unique identifier for the vehicle")
    weight: float = Field(..., gt=0,
                       description="Vehicle weight in kg")
    drag_coefficient: float = Field(..., gt=0, lt=1,
                                 description="Aerodynamic drag coefficient")
    frontal_area: float = Field(..., gt=0,
                             description="Frontal area in m²")
    battery_capacity: float = Field(..., gt=0,
                                 description="Battery capacity in kWh")
    rolling_resistance: Optional[float] = Field(0.01, gt=0, lt=1,
                                            description="Rolling resistance coefficient")
    wheel_radius: Optional[float] = Field(0.35, gt=0,
                                      description="Wheel radius in meters")

    @validator('weight')
    def weight_must_be_realistic(cls, v):
        """Validate that weight is within realistic bounds."""
        if v < 500 or v > 5000:
            raise ValueError('Weight must be between 500 and 5000 kg')
        return v

    @validator('battery_capacity')
    def battery_capacity_must_be_realistic(cls, v):
        """Validate that battery capacity is within realistic bounds."""
        if v < 10 or v > 200:
            raise ValueError('Battery capacity must be between 10 and 200 kWh')
        return v


class DrivingCondition(BaseModel):
    """
    Model representing a single driving condition data point.
    """
    timestamp: str = Field(...,
                        description="ISO format timestamp")
    speed: float = Field(..., ge=0,
                      description="Vehicle speed in km/h")
    acceleration: Optional[float] = Field(None,
                                      description="Acceleration in m/s²")
    altitude: float = Field(...,
                         description="Altitude above sea level in meters")
    temperature: float = Field(...,
                           description="Ambient temperature in Celsius")
    wind_speed: Optional[float] = Field(0.0, ge=0,
                                    description="Wind speed in km/h")
    precipitation: Optional[float] = Field(0.0, ge=0,
                                       description="Precipitation in mm")
    road_type: Optional[str] = Field("highway",
                                 description="Type of road (highway, urban, rural)")

    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Validate that timestamp is in ISO format."""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError('Timestamp must be in ISO format (e.g., 2023-05-15T14:30:00Z)')

    @validator('speed')
    def speed_must_be_realistic(cls, v):
        """Validate that speed is within realistic bounds."""
        if v > 300:
            raise ValueError('Speed cannot be greater than 300 km/h')
        return v

    @validator('temperature')
    def temperature_must_be_realistic(cls, v):
        """Validate that temperature is within realistic bounds."""
        if v < -50 or v > 60:
            raise ValueError('Temperature must be between -50 and 60 Celsius')
        return v

    @validator('road_type')
    def road_type_must_be_valid(cls, v):
        """Validate that road type is one of the allowed values."""
        valid_types = ['highway', 'urban', 'rural', 'mountain', 'suburban', 'offroad']
        if v and v.lower() not in valid_types:
            raise ValueError(f'Road type must be one of: {", ".join(valid_types)}')
        return v.lower() if v else v


class PredictionRequest(BaseModel):
    """
    Model representing a prediction request.
    """
    vehicle: VehicleData = Field(...,
                              description="Vehicle specification data")
    driving_conditions: List[DrivingCondition] = Field(..., min_items=1,
                                                   description="List of driving condition data points")

    @validator('driving_conditions')
    def validate_driving_conditions(cls, v):
        """Validate that there is at least one driving condition."""
        if not v:
            raise ValueError('At least one driving condition is required')
        return v


class PredictionResult(BaseModel):
    """
    Model representing a single prediction result.
    """
    timestamp: str = Field(...,
                        description="ISO format timestamp")
    speed: float = Field(...,
                      description="Vehicle speed in km/h")
    energy_rate: float = Field(...,
                           description="Energy consumption rate in kWh")
    cumulative_energy: float = Field(...,
                                description="Cumulative energy consumption in kWh")


class ConfidenceInterval(BaseModel):
    """
    Model representing a confidence interval for predictions.
    """
    lower_bound: float = Field(...,
                           description="Lower bound of the prediction")
    upper_bound: float = Field(...,
                           description="Upper bound of the prediction")
    uncertainty: float = Field(...,
                          description="Uncertainty as a percentage")


class PredictionResponse(BaseModel):
    """
    Model representing a prediction response.
    """
    energy_consumption: float = Field(...,
                                  description="Total predicted energy consumption in kWh")
    remaining_range: float = Field(...,
                               description="Estimated remaining range in km")
    predictions: List[PredictionResult] = Field(...,
                                           description="Detailed predictions for each time step")
    confidence_interval: Optional[ConfidenceInterval] = Field(None,
                                                        description="Confidence interval for the prediction")
    model_used: str = Field(...,
                        description="Model used for the prediction")


class VehicleTypeInfo(BaseModel):
    """
    Model representing information about a vehicle type.
    """
    id: str = Field(...,
                 description="Unique identifier for the vehicle type")
    name: str = Field(...,
                   description="Display name for the vehicle type")
    weight: float = Field(...,
                       description="Default weight in kg")
    drag_coefficient: float = Field(...,
                                 description="Default drag coefficient")
    frontal_area: float = Field(...,
                             description="Default frontal area in m²")
    battery_capacity: float = Field(...,
                                 description="Default battery capacity in kWh")
    rolling_resistance: float = Field(...,
                                  description="Default rolling resistance coefficient")


class VehicleTypeResponse(BaseModel):
    """
    Model representing a response with available vehicle types.
    """
    vehicles: List[VehicleTypeInfo] = Field(...,
                                       description="List of available vehicle types")


class RoadTypeInfo(BaseModel):
    """
    Model representing information about a road type.
    """
    id: str = Field(...,
                 description="Unique identifier for the road type")
    name: str = Field(...,
                   description="Display name for the road type")
    description: str = Field(...,
                         description="Description of the road type")


class RoadTypeResponse(BaseModel):
    """
    Model representing a response with available road types.
    """
    road_types: List[RoadTypeInfo] = Field(...,
                                     description="List of available road types")


class HealthResponse(BaseModel):
    """
    Model representing a health check response.
    """
    status: str = Field(...,
                     description="Health status of the API")
    models: Dict[str, bool] = Field(...,
                                description="Status of loaded models")
    request_count: int = Field(...,
                           description="Total number of processed requests")
    error_count: int = Field(...,
                         description="Total number of errors")
    timestamp: str = Field(...,
                        description="Current timestamp")


class MetricsResponse(BaseModel):
    """
    Model representing an API metrics response.
    """
    total_requests: int = Field(...,
                            description="Total number of processed requests")
    error_count: int = Field(...,
                         description="Total number of errors")
    success_rate: float = Field(...,
                            description="Request success rate")
    models_loaded: List[str] = Field(...,
                                description="List of loaded models")
    recent_history_size: int = Field(...,
                               description="Size of recent request history")

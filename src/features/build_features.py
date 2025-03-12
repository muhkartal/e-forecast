"""
Feature generation for the Energy Prediction System.

This module provides functions for generating advanced features from raw data
to improve model performance in predicting energy consumption.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Class for building features from processed data.

    This class provides methods to construct advanced features for energy
    consumption prediction based on vehicle, environmental, and driving data.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature builder.

        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        logger.info("Initialized FeatureBuilder")

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate advanced features from processed data.

        Args:
            df: Processed DataFrame

        Returns:
            DataFrame with additional features
        """
        logger.info(f"Building features for dataframe with shape {df.shape}")

        # Create a copy to avoid modifying the original
        result_df = df.copy()

        # Apply feature transformations
        result_df = self.add_physics_features(result_df)
        result_df = self.add_temporal_features(result_df)
        result_df = self.add_environmental_features(result_df)
        result_df = self.add_statistical_features(result_df)
        result_df = self.add_interaction_features(result_df)

        logger.info(f"Generated features, new shape: {result_df.shape}")
        return result_df

    def add_physics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add physics-based features related to energy consumption.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with additional physics features
        """
        logger.debug("Adding physics-based features")
        result_df = df.copy()

        # Check if necessary columns exist
        required_cols = ['speed', 'weight', 'acceleration', 'drag_coefficient',
                       'frontal_area', 'altitude']
        missing_cols = [col for col in required_cols if col not in result_df.columns]

        if missing_cols:
            logger.warning(f"Missing columns for physics features: {missing_cols}")
            # Return original if missing required columns
            return result_df

        try:
            # Add air density if not present (based on temperature or standard value)
            if 'air_density' not in result_df.columns:
                if 'temperature' in result_df.columns:
                    # Calculate air density based on temperature (simplified model)
                    # Formula: density = 1.225 * (288.15 / (T + 273.15))
                    result_df['air_density'] = 1.225 * (288.15 / (result_df['temperature'] + 273.15))
                else:
                    # Use standard air density at sea level
                    result_df['air_density'] = 1.225

            # Add rolling resistance coefficient if not present
            if 'rolling_resistance' not in result_df.columns:
                result_df['rolling_resistance'] = 0.01  # Default value

            # Convert speed from km/h to m/s for physics calculations
            speed_ms = result_df['speed'] / 3.6

            # Calculate forces and power components
            # 1. Rolling resistance force: F_r = C_r * m * g
            result_df['force_rolling'] = result_df['rolling_resistance'] * result_df['weight'] * 9.81

            # 2. Aerodynamic drag force: F_d = 0.5 * ρ * C_d * A * v²
            result_df['force_drag'] = 0.5 * result_df['air_density'] * \
                                    result_df['drag_coefficient'] * \
                                    result_df['frontal_area'] * speed_ms**2

            # 3. Acceleration force: F_a = m * a
            result_df['force_acceleration'] = result_df['weight'] * result_df['acceleration']

            # 4. Gradient force: F_g = m * g * sin(α)
            # Calculate gradient (slope) from elevation changes if available
            if 'elevation_change' in result_df.columns and 'distance' in result_df.columns:
                # Calculate slope as rise over run
                result_df['slope'] = result_df['elevation_change'] / result_df['distance']
                # Approximate sin(α) ≈ tan(α) for small angles
                result_df['force_gradient'] = result_df['weight'] * 9.81 * result_df['slope']
            elif 'elevation_change' in result_df.columns:
                # Use a default distance if not available
                default_distance = 100  # meters
                result_df['slope'] = result_df['elevation_change'] / default_distance
                result_df['force_gradient'] = result_df['weight'] * 9.81 * result_df['slope']
            else:
                result_df['slope'] = 0
                result_df['force_gradient'] = 0

            # 5. Total force: F_total = F_r + F_d + F_a + F_g
            result_df['force_total'] = result_df['force_rolling'] + \
                                     result_df['force_drag'] + \
                                     result_df['force_acceleration'] + \
                                     result_df['force_gradient']

            # 6. Power: P = F * v
            result_df['power'] = result_df['force_total'] * speed_ms

            # 7. Energy rate in kW: P (W) / 1000
            result_df['energy_rate_kw'] = result_df['power'] / 1000

            # 8. Specific energy consumption (kWh/km): energy_rate_kw / (speed_km/h)
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                result_df['specific_consumption'] = np.where(
                    result_df['speed'] > 0,
                    result_df['energy_rate_kw'] / result_df['speed'],
                    0
                )

            # 9. Regenerative braking potential (negative acceleration indicates braking)
            result_df['regen_potential'] = np.where(
                result_df['acceleration'] < 0,
                -result_df['force_acceleration'] * speed_ms / 1000 * 0.7,  # 70% efficiency
                0
            )

            # 10. Net energy consumption rate (considering regenerative braking)
            result_df['net_energy_rate_kw'] = result_df['energy_rate_kw'] - result_df['regen_potential']

        except Exception as e:
            logger.error(f"Error calculating physics features: {str(e)}")

        return result_df

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with additional temporal features
        """
        logger.debug("Adding temporal features")
        result_df = df.copy()

        # Check if timestamp column exists
        if 'timestamp' not in result_df.columns:
            logger.warning("Missing timestamp column for temporal features")
            return result_df

        try:
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(result_df['timestamp']):
                result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])

            # Extract basic time components
            result_df['hour'] = result_df['timestamp'].dt.hour
            result_df['day'] = result_df['timestamp'].dt.day
            result_df['day_of_week'] = result_df['timestamp'].dt.dayofweek
            result_df['month'] = result_df['timestamp'].dt.month
            result_df['quarter'] = result_df['timestamp'].dt.quarter
            result_df['year'] = result_df['timestamp'].dt.year

            # Create cyclical features for time (to handle periodicity)
            # Hour of day as sine and cosine components
            hours_in_day = 24
            result_df['hour_sin'] = np.sin(2 * np.pi * result_df['hour'] / hours_in_day)
            result_df['hour_cos'] = np.cos(2 * np.pi * result_df['hour'] / hours_in_day)

            # Day of week as sine and cosine components
            days_in_week = 7
            result_df['day_of_week_sin'] = np.sin(2 * np.pi * result_df['day_of_week'] / days_in_week)
            result_df['day_of_week_cos'] = np.cos(2 * np.pi * result_df['day_of_week'] / days_in_week)

            # Month of year as sine and cosine components
            months_in_year = 12
            result_df['month_sin'] = np.sin(2 * np.pi * result_df['month'] / months_in_year)
            result_df['month_cos'] = np.cos(2 * np.pi * result_df['month'] / months_in_year)

            # Create categorical time features
            # Time of day categories
            result_df['time_of_day'] = pd.cut(
                result_df['hour'],
                bins=[0, 6, 12, 18, 24],
                labels=['night', 'morning', 'afternoon', 'evening'],
                include_lowest=True
            )

            # Weekend indicator
            result_df['is_weekend'] = (result_df['day_of_week'] >= 5).astype(int)

            # Rush hour indicator (simplified)
            rush_hour_morning = (result_df['hour'] >= 7) & (result_df['hour'] <= 9)
            rush_hour_evening = (result_df['hour'] >= 16) & (result_df['hour'] <= 19)
            result_df['is_rush_hour'] = (rush_hour_morning | rush_hour_evening).astype(int)

            # Season determination (Northern Hemisphere)
            # Winter: Dec, Jan, Feb (12, 1, 2)
            # Spring: Mar, Apr, May (3, 4, 5)
            # Summer: Jun, Jul, Aug (6, 7, 8)
            # Fall: Sep, Oct, Nov (9, 10, 11)
            conditions = [
                (result_df['month'].isin([12, 1, 2])),
                (result_df['month'].isin([3, 4, 5])),
                (result_df['month'].isin([6, 7, 8])),
                (result_df['month'].isin([9, 10, 11]))
            ]
            season_labels = ['winter', 'spring', 'summer', 'fall']
            result_df['season'] = np.select(conditions, season_labels, default='unknown')

            # Calculate time differences if multiple timestamps
            if len(result_df) > 1:
                result_df['time_diff'] = result_df['timestamp'].diff().dt.total_seconds()
                # Fill first row with appropriate value
                result_df['time_diff'] = result_df['time_diff'].fillna(
                    result_df['time_diff'].median() if not result_df['time_diff'].empty else 0
                )

                # Calculate time-based rates (change per second)
                if 'speed' in result_df.columns:
                    result_df['speed_change_rate'] = result_df['speed'].diff() / result_df['time_diff']
                    result_df['speed_change_rate'] = result_df['speed_change_rate'].fillna(0)

                if 'altitude' in result_df.columns:
                    result_df['altitude_change_rate'] = result_df['altitude'].diff() / result_df['time_diff']
                    result_df['altitude_change_rate'] = result_df['altitude_change_rate'].fillna(0)

        except Exception as e:
            logger.error(f"Error calculating temporal features: {str(e)}")

        return result_df

    def add_environmental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add environmental and weather-based features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with additional environmental features
        """
        logger.debug("Adding environmental features")
        result_df = df.copy()

        try:
            # Temperature features (if available)
            if 'temperature' in result_df.columns:
                # Temperature comfort zone deviation (optimal range: 18-24°C)
                result_df['temp_comfort_deviation'] = np.minimum(
                    np.maximum(18 - result_df['temperature'], 0),  # below comfort zone
                    np.maximum(result_df['temperature'] - 24, 0)   # above comfort zone
                )

                # Temperature impact on battery (simplified model)
                # Battery efficiency is typically reduced in extreme temperatures
                # Optimal range: 15-30°C
                result_df['battery_temp_efficiency'] = 1.0 - 0.01 * np.minimum(
                    np.maximum(15 - result_df['temperature'], 0),    # cold impact
                    np.maximum(result_df['temperature'] - 30, 0) * 1.5  # hot impact (1.5x worse)
                )

                # HVAC energy requirement estimation
                # Heating when cold, cooling when hot
                result_df['hvac_load'] = np.where(
                    result_df['temperature'] < 15,
                    (15 - result_df['temperature']) * 0.1,  # heating load
                    np.where(
                        result_df['temperature'] > 25,
                        (result_df['temperature'] - 25) * 0.07,  # cooling load
                        0  # no significant HVAC load
                    )
                )

            # Wind features (if available)
            if 'wind_speed' in result_df.columns and 'wind_direction' in result_df.columns:
                # Calculate headwind/tailwind component if driving direction available
                if 'driving_direction' in result_df.columns:
                    # Angle difference between wind and driving direction
                    result_df['wind_angle_diff'] = (result_df['wind_direction'] - result_df['driving_direction']) % 360

                    # Headwind component (positive = headwind, negative = tailwind)
                    result_df['headwind'] = result_df['wind_speed'] * np.cos(np.radians(result_df['wind_angle_diff']))

                    # Crosswind component (absolute value)
                    result_df['crosswind'] = np.abs(result_df['wind_speed'] * np.sin(np.radians(result_df['wind_angle_diff'])))
                else:
                    # Use wind speed as a feature without direction information
                    result_df['wind_factor'] = result_df['wind_speed'] * 0.05  # simplified impact factor
            elif 'wind_speed' in result_df.columns:
                # Use wind speed as a feature without direction
                result_df['wind_factor'] = result_df['wind_speed'] * 0.05  # simplified impact factor

            # Precipitation and road condition features (if available)
            if 'precipitation' in result_df.columns:
                # Create road condition based on precipitation
                result_df['road_condition'] = pd.cut(
                    result_df['precipitation'],
                    bins=[-0.01, 0.01, 2.5, 7.6, 50, 1000],
                    labels=['dry', 'light_rain', 'moderate_rain', 'heavy_rain', 'extreme'],
                    include_lowest=True
                )

                # Precipitation impact factor
                result_df['precipitation_factor'] = np.minimum(result_df['precipitation'] * 0.02, 0.15)

        except Exception as e:
            logger.error(f"Error calculating environmental features: {str(e)}")

        return result_df

    def add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add statistical and rolling window features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with additional statistical features
        """
        logger.debug("Adding statistical features")
        result_df = df.copy()

        try:
            # Create rolling window features if enough data points
            min_periods = min(5, len(result_df) // 2)

            if len(result_df) >= 10:
                # Speed statistics
                if 'speed' in result_df.columns:
                    # Rolling mean and std with adaptive window sizes
                    result_df['speed_rolling_mean_5'] = result_df['speed'].rolling(
                        window=5, min_periods=min_periods).mean()
                    result_df['speed_rolling_std_5'] = result_df['speed'].rolling(
                        window=5, min_periods=min_periods).std()

                    if len(result_df) >= 20:
                        result_df['speed_rolling_mean_10'] = result_df['speed'].rolling(
                            window=10, min_periods=min_periods).mean()
                        result_df['speed_rolling_std_10'] = result_df['speed'].rolling(
                            window=10, min_periods=min_periods).std()

                    # Speed variability (coefficient of variation)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        result_df['speed_variability'] = np.where(
                            result_df['speed_rolling_mean_5'] > 0,
                            result_df['speed_rolling_std_5'] / result_df['speed_rolling_mean_5'],
                            0
                        )

                    # Speed change events (acceleration/deceleration)
                    if 'acceleration' in result_df.columns:
                        # Count acceleration and deceleration events
                        result_df['accel_events'] = (result_df['acceleration'] > 0.5).astype(int)
                        result_df['decel_events'] = (result_df['acceleration'] < -0.5).astype(int)

                        # Cumulative counts of events
                        result_df['accel_events_cum'] = result_df['accel_events'].cumsum()
                        result_df['decel_events_cum'] = result_df['decel_events'].cumsum()

                        # Rolling sum of events (frequency in last 10 periods)
                        if len(result_df) >= 10:
                            result_df['accel_events_freq'] = result_df['accel_events'].rolling(
                                window=10, min_periods=min_periods).sum()
                            result_df['decel_events_freq'] = result_df['decel_events'].rolling(
                                window=10, min_periods=min_periods).sum()

                # Compute exponentially weighted features for smoother transitions
                if 'speed' in result_df.columns:
                    result_df['speed_ewm_mean'] = result_df['speed'].ewm(span=5).mean()

                if 'acceleration' in result_df.columns:
                    result_df['accel_ewm_mean'] = result_df['acceleration'].ewm(span=5).mean()

                # Fill NaN values created by rolling windows
                for col in result_df.columns:
                    if pd.api.types.is_numeric_dtype(result_df[col]) and result_df[col].isna().any():
                        result_df[col] = result_df[col].fillna(method='bfill').fillna(method='ffill').fillna(0)

        except Exception as e:
            logger.error(f"Error calculating statistical features: {str(e)}")

        return result_df

    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add interaction and derived features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with additional interaction features
        """
        logger.debug("Adding interaction features")
        result_df = df.copy()

        try:
            # Create interaction features between key variables

            # Vehicle efficiency factor (weight and aerodynamics)
            if all(col in result_df.columns for col in ['weight', 'drag_coefficient', 'frontal_area']):
                result_df['vehicle_efficiency_factor'] = result_df['weight'] * \
                                                      result_df['drag_coefficient'] * \
                                                      result_df['frontal_area'] / 1000

            # Speed-temperature interaction (effect of temperature on efficiency at different speeds)
            if all(col in result_df.columns for col in ['speed', 'temperature']):
                # High speed with extreme temperature amplifies energy consumption
                speed_factor = np.minimum(result_df['speed'] / 100, 1.5)  # cap at 1.5x
                temp_deviation = np.abs(result_df['temperature'] - 20)  # deviation from optimal 20°C
                result_df['speed_temp_interaction'] = speed_factor * temp_deviation

            # Driving style features
            if all(col in result_df.columns for col in ['speed', 'acceleration', 'speed_variability']):
                # Aggressive driving score
                accel_factor = np.maximum(result_df['acceleration'] * 2, 0)  # only consider positive acceleration
                speed_var_factor = result_df['speed_variability'] * 5
                result_df['aggressive_driving_score'] = accel_factor + speed_var_factor

                # Efficiency score (inverse of aggressive driving)
                result_df['efficiency_score'] = 1 / (1 + result_df['aggressive_driving_score'])

            # Traffic condition inference
            if all(col in result_df.columns for col in ['speed', 'speed_variability', 'accel_events_freq', 'decel_events_freq']):
                # Traffic density score
                slow_speed_factor = np.maximum(1 - result_df['speed'] / 60, 0)  # more traffic if slow
                stop_start_factor = (result_df['accel_events_freq'] + result_df['decel_events_freq']) / 10
                result_df['traffic_density_score'] = (slow_speed_factor + stop_start_factor + result_df['speed_variability']) / 3

            # Total energy impact factors
            energy_factors = []

            # Add available factors
            if 'vehicle_efficiency_factor' in result_df.columns:
                energy_factors.append(result_df['vehicle_efficiency_factor'] * 0.2)

            if 'aggressive_driving_score' in result_df.columns:
                energy_factors.append(result_df['aggressive_driving_score'] * 0.3)

            if 'traffic_density_score' in result_df.columns:
                energy_factors.append(result_df['traffic_density_score'] * 0.15)

            if 'hvac_load' in result_df.columns:
                energy_factors.append(result_df['hvac_load'])

            if 'precipitation_factor' in result_df.columns:
                energy_factors.append(result_df['precipitation_factor'])

            if 'wind_factor' in result_df.columns:
                energy_factors.append(result_df['wind_factor'])

            # Combine factors if any are available
            if energy_factors:
                result_df['total_energy_impact'] = sum(energy_factors)

        except Exception as e:
            logger.error(f"Error calculating interaction features: {str(e)}")

        return result_df

    def select_features(self, df: pd.DataFrame, feature_list: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Select specific features from the DataFrame.

        Args:
            df: Input DataFrame with all features
            feature_list: Optional list of features to select

        Returns:
            DataFrame with selected features
        """
        if feature_list is None:
            # Default list of recommended features
            feature_list = self.config.get('recommended_features', [])

            if not feature_list:
                # Return all features if no list is specified
                return df

        # Select only features that exist in the DataFrame
        valid_features = [f for f in feature_list if f in df.columns]

        # Log any missing features
        missing_features = [f for f in feature_list if f not in df.columns]
        if missing_features:
            logger.warning(f"Some requested features are not available: {missing_features}")

        # Return DataFrame with selected features
        return df[valid_features]


# Create feature sets for specific model types
def create_feature_sets(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]:
    """
    Create specific feature sets for different model types.

    Args:
        df: DataFrame with all features
        config: Optional configuration dictionary

    Returns:
        Dictionary of feature sets for each model type
    """
    config = config or {}
    feature_builder = FeatureBuilder(config)

    # Generate all features
    full_features_df = feature_builder.build_features(df)

    # Define feature sets for each model type
    feature_sets = {}

    # XGBoost likes a wide range of features
    xgb_features = config.get('xgboost_features', [])
    if not xgb_features:
        # Default XGBoost features if not specified
        numeric_cols = full_features_df.select_dtypes(include=np.number).columns.tolist()
        xgb_features = numeric_cols

    feature_sets['xgboost'] = feature_builder.select_features(full_features_df, xgb_features)

    # LSTM typically uses fewer, but sequential features
    lstm_features = config.get('lstm_features', [])
    if not lstm_features:
        # Default LSTM features if not specified
        primary_features = [
            'speed', 'acceleration', 'altitude', 'energy_rate_kw',
            'force_total', 'battery_temp_efficiency', 'hour_sin', 'hour_cos'
        ]
        lstm_features = [f for f in primary_features if f in full_features_df.columns]

    feature_sets['lstm'] = feature_builder.select_features(full_features_df, lstm_features)

    # Neural network features
    nn_features = config.get('nn_features', [])
    if not nn_features:
        # Default NN features if not specified (similar to XGBoost but maybe fewer)
        nn_features = xgb_features[:min(len(xgb_features), 20)]  # Limit to 20 important features

    feature_sets['neural_network'] = feature_builder.select_features(full_features_df, nn_features)

    # Keep the full feature set
    feature_sets['all'] = full_features_df

    return feature_sets


# Direct functions for easier usage
def build_features(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Build features from raw data.

    Args:
        df: Input DataFrame
        config: Optional configuration parameters

    Returns:
        DataFrame with engineered features
    """
    feature_builder = FeatureBuilder(config)
    return feature_builder.build_features(df)


def select_important_features(df: pd.DataFrame, target_col: str,
                           n_features: int = 20, method: str = 'mutual_info') -> List[str]:
    """
    Select the most important features using statistical methods.

    Args:
        df: Input DataFrame with features
        target_col: Target column name
        n_features: Number of features to select
        method: Feature selection method ('mutual_info', 'f_regression', 'correlation')

    Returns:
        List of selected feature names
    """
    try:
        # Check if sklearn is available
        import sklearn
        from sklearn.feature_selection import mutual_info_regression, f_regression

        # Select only numeric columns
        numeric_df = df.select_dtypes(include=np.number)

        # Drop the target column from features
        if target_col in numeric_df.columns:
            X = numeric_df.drop(columns=[target_col])
            y = numeric_df[target_col]
        else:
            X = numeric_df
            y = df[target_col]

        feature_names = X.columns.tolist()

        # Apply feature selection method
        if method == 'mutual_info':
            # Mutual information regression (works for non-linear relationships)
            mi_scores = mutual_info_regression(X, y)
            importance = dict(zip(feature_names, mi_scores))
        elif method == 'f_regression':
            # F-test for regression (linear relationships)
            f_scores, _ = f_regression(X, y)
            importance = dict(zip(feature_names, f_scores))
        elif method == 'correlation':
            # Simple correlation coefficient
            correlations = X.corrwith(y).abs()
            importance = dict(zip(feature_names, correlations))
        else:
            raise ValueError(f"Unknown feature selection method: {method}")

        # Sort features by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in sorted_features[:n_features]]

        return selected_features

    except ImportError:
        logger.warning("sklearn not available, using basic correlation for feature selection")

        # Fallback to basic correlation
        if target_col in df.columns:
            numeric_df = df.select_dtypes(include=np.number)
            correlations = numeric_df.corrwith(df[target_col]).abs()
            correlations = correlations.dropna().sort_values(ascending=False)

            # Remove target column
            if target_col in correlations.index:
                correlations = correlations.drop(target_col)

            return correlations.index.tolist()[:n_features]
        else:
            # If target not available, return numeric columns
            return df.select_dtypes(include=np.number).columns.tolist()[:n_features]
    except Exception as e:
        logger.error(f"Error in feature selection: {str(e)}")
        return df.columns.tolist()[:n_features]  # Return first n_features as fallback

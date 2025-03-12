"""
XGBoost model implementation for energy prediction.

This module contains the implementation of the XGBoost-based energy prediction model,
including training, hyperparameter optimization, evaluation, and inference.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import joblib
import yaml
import logging
import matplotlib.pyplot as plt
import shap
from typing import Dict, Tuple, List, Optional, Union, Any

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Configure logging
logger = logging.getLogger(__name__)


class XGBoostEnergyPredictor:
    """
    XGBoost-based model for predicting EV energy consumption.

    This class provides methods to train, optimize, evaluate, and use XGBoost
    regression models for energy consumption prediction.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the XGBoost predictor with configuration parameters.

        Args:
            config: Dictionary containing model configuration parameters
        """
        self.config = config
        self.model = None
        self.feature_names = None
        self.best_params = None
        logger.info("Initialized XGBoostEnergyPredictor with config: %s", config)

    def train_basic(self, X_train: Union[np.ndarray, pd.DataFrame],
                    y_train: Union[np.ndarray, pd.Series]) -> xgb.XGBRegressor:
        """
        Train a basic XGBoost model with default parameters.

        Args:
            X_train: Training features
            y_train: Training target values

        Returns:
            Trained XGBoost model
        """
        logger.info("Training basic XGBoost model")

        # Store feature names if available
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()

        # Initialize model with parameters from config or defaults
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=self.config.get('n_estimators', 100),
            max_depth=self.config.get('max_depth', 6),
            learning_rate=self.config.get('learning_rate', 0.1),
            subsample=self.config.get('subsample', 0.8),
            colsample_bytree=self.config.get('colsample_bytree', 0.8),
            random_state=self.config.get('random_seed', 42),
            n_jobs=self.config.get('n_jobs', -1)
        )

        # Train the model
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            verbose=self.config.get('verbose', False)
        )

        self.model = model
        logger.info("Basic XGBoost model training completed")
        return model

    def hyperparameter_optimization(self,
                                   X_train: Union[np.ndarray, pd.DataFrame],
                                   y_train: Union[np.ndarray, pd.Series],
                                   X_val: Union[np.ndarray, pd.DataFrame],
                                   y_val: Union[np.ndarray, pd.Series],
                                   max_evals: int = 50) -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
        """
        Optimize hyperparameters using Bayesian optimization.

        Args:
            X_train: Training features
            y_train: Training target values
            X_val: Validation features
            y_val: Validation target values
            max_evals: Maximum number of evaluations for optimization

        Returns:
            Tuple containing the optimized model and the best parameters
        """
        logger.info("Starting hyperparameter optimization with %d max evaluations", max_evals)

        # Store feature names if available
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()

        def objective(params):
            """
            Objective function for hyperopt optimization.

            Args:
                params: Parameters to evaluate

            Returns:
                Dictionary with loss value and status
            """
            # Create model with current parameters
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=int(params['n_estimators']),
                max_depth=int(params['max_depth']),
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                min_child_weight=int(params['min_child_weight']),
                gamma=params['gamma'],
                reg_alpha=params.get('reg_alpha', 0),
                reg_lambda=params.get('reg_lambda', 1),
                random_state=self.config.get('random_seed', 42),
                n_jobs=self.config.get('n_jobs', -1)
            )

            # Train the model
            evaluation_results = {}
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False,
                eval_metric='rmse',
                callbacks=[xgb.callback.EvaluationMonitor(period=100)]
            )

            # Predict and compute metrics
            y_pred = model.predict(X_val)
            mape = mean_absolute_percentage_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))

            # Log current evaluation
            logger.debug("Trial with params: %s, MAPE: %.4f, RMSE: %.4f",
                        params, mape, rmse)

            return {
                'loss': mape,  # We want to minimize MAPE
                'rmse': rmse,
                'status': STATUS_OK,
                'model': model
            }

        # Define the search space
        space = {
            'n_estimators': hp.quniform('n_estimators', 50, 500, 10),
            'max_depth': hp.quniform('max_depth', 3, 12, 1),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
            'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
            'gamma': hp.uniform('gamma', 0, 1),
            'reg_alpha': hp.loguniform('reg_alpha', np.log(0.01), np.log(10)),
            'reg_lambda': hp.loguniform('reg_lambda', np.log(0.01), np.log(10))
        }

        # Run the optimization
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials
        )

        # Get the best parameters
        best_params = {
            'n_estimators': int(best['n_estimators']),
            'max_depth': int(best['max_depth']),
            'learning_rate': best['learning_rate'],
            'subsample': best['subsample'],
            'colsample_bytree': best['colsample_bytree'],
            'min_child_weight': int(best['min_child_weight']),
            'gamma': best['gamma'],
            'reg_alpha': best['reg_alpha'],
            'reg_lambda': best['reg_lambda'],
            'objective': 'reg:squarederror',
            'random_state': self.config.get('random_seed', 42),
            'n_jobs': self.config.get('n_jobs', -1)
        }

        logger.info("Best hyperparameters found: %s", best_params)

        # Train the final model with the best parameters
        model = xgb.XGBRegressor(**best_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=self.config.get('verbose', False)
        )

        self.model = model
        self.best_params = best_params

        return model, best_params

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Generate predictions using the trained model.

        Args:
            X: Feature data for prediction

        Returns:
            Array of predictions

        Raises:
            ValueError: If model has not been trained
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_basic() or hyperparameter_optimization() first.")

        logger.debug("Generating predictions for %d samples", len(X))
        return self.model.predict(X)

    def evaluate(self, X_test: Union[np.ndarray, pd.DataFrame],
                y_test: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Evaluate model performance on test data.

        Args:
            X_test: Test features
            y_test: Test target values

        Returns:
            Dictionary containing evaluation metrics

        Raises:
            ValueError: If model has not been trained
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_basic() or hyperparameter_optimization() first.")

        logger.info("Evaluating model on %d test samples", len(X_test))
        y_pred = self.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }

        logger.info("Evaluation metrics: MSE=%.4f, RMSE=%.4f, MAE=%.4f, MAPE=%.2f%%",
                   mse, rmse, mae, mape)

        return metrics

    def feature_importance(self, plot: bool = False,
                          save_path: Optional[str] = None) -> Dict[str, float]:
        """
        Get and optionally visualize feature importance.

        Args:
            plot: Whether to create a plot
            save_path: Path to save the plot, if None and plot=True, plot is displayed

        Returns:
            Dictionary mapping feature names to importance values

        Raises:
            ValueError: If model has not been trained
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_basic() or hyperparameter_optimization() first.")

        # Get feature importance
        importance = self.model.feature_importances_

        # Map to feature names if available
        if self.feature_names:
            importance_dict = dict(zip(self.feature_names, importance))
            sorted_importance = {k: v for k, v in sorted(importance_dict.items(),
                                                        key=lambda item: item[1],
                                                        reverse=True)}
        else:
            importance_dict = {f"feature_{i}": imp for i, imp in enumerate(importance)}
            sorted_importance = {k: v for k, v in sorted(importance_dict.items(),
                                                        key=lambda item: item[1],
                                                        reverse=True)}

        # Create plot if requested
        if plot:
            plt.figure(figsize=(10, 6))
            features = list(sorted_importance.keys())
            values = list(sorted_importance.values())

            # Plot horizontal bar chart
            plt.barh(features, values)
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title('XGBoost Feature Importance')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info("Feature importance plot saved to %s", save_path)
            else:
                plt.show()

        return sorted_importance

    def shap_analysis(self, X_sample: Union[np.ndarray, pd.DataFrame],
                     plot: bool = True,
                     save_path: Optional[str] = None,
                     max_display: int = 10) -> None:
        """
        Perform SHAP value analysis for model explainability.

        Args:
            X_sample: Sample data for SHAP analysis
            plot: Whether to create plots
            save_path: Directory to save plots
            max_display: Maximum number of features to display

        Raises:
            ValueError: If model has not been trained
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_basic() or hyperparameter_optimization() first.")

        try:
            # Create explainer
            explainer = shap.Explainer(self.model)

            # Calculate SHAP values
            shap_values = explainer(X_sample)

            if plot:
                # Summary plot
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, max_display=max_display)
                if save_path:
                    plt.savefig(f"{save_path}/shap_summary.png", dpi=300, bbox_inches='tight')
                plt.tight_layout()
                plt.show()

                # Dependence plots for top features
                if self.feature_names:
                    top_features = list(self.feature_importance().keys())[:min(5, len(self.feature_names))]

                    for feature in top_features:
                        plt.figure(figsize=(8, 6))
                        shap.dependence_plot(feature, shap_values.values, X_sample,
                                           feature_names=self.feature_names)
                        if save_path:
                            plt.savefig(f"{save_path}/shap_dependence_{feature}.png",
                                      dpi=300, bbox_inches='tight')
                        plt.tight_layout()
                        plt.show()

            logger.info("SHAP analysis completed")

        except Exception as e:
            logger.error("Error during SHAP analysis: %s", str(e))
            raise

    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.

        Args:
            path: Path to save the model

        Raises:
            ValueError: If model has not been trained
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_basic() or hyperparameter_optimization() first.")

        # Create a model package with metadata
        model_package = {
            'model': self.model,
            'feature_names': self.feature_names,
            'best_params': self.best_params,
            'config': self.config
        }

        joblib.dump(model_package, path)
        logger.info("Model saved to %s", path)

    def load_model(self, path: str) -> xgb.XGBRegressor:
        """
        Load a saved model from disk.

        Args:
            path: Path to the saved model

        Returns:
            Loaded XGBoost model
        """
        # Load the model package
        model_package = joblib.load(path)

        # Extract components
        self.model = model_package['model']
        self.feature_names = model_package['feature_names']
        self.best_params = model_package['best_params']
        self.config = model_package['config']

        logger.info("Model loaded from %s", path)
        return self.model


# Example usage function
def example_usage():
    """Example usage of the XGBoostEnergyPredictor."""
    # Load configuration
    with open("configs/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)["xgboost"]

    # Create synthetic data for demonstration
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train model
    predictor = XGBoostEnergyPredictor(config)
    predictor.train_basic(X_train, y_train)

    # Evaluate model
    metrics = predictor.evaluate(X_test, y_test)
    print(f"Model performance: {metrics}")

    # Get feature importance
    importance = predictor.feature_importance(plot=True)
    print(f"Feature importance: {importance}")

    # Save and load model
    predictor.save_model("models/xgboost_model.joblib")
    predictor.load_model("models/xgboost_model.joblib")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Run example
    example_usage()

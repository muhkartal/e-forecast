"""
LSTM model implementation for energy prediction.

This module contains the implementation of a deep learning LSTM-based model
for energy consumption prediction in electric vehicles.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

# Configure logging
logger = logging.getLogger(__name__)


class EnergyPredictionLSTM:
    """
    LSTM-based model for predicting EV energy consumption.

    This class provides methods to build, train, evaluate, and use LSTM
    models for energy consumption prediction based on sequential data.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LSTM predictor with configuration parameters.

        Args:
            config: Dictionary containing model configuration parameters
        """
        self.config = config
        self.model = None
        self.history = None
        logger.info("Initialized EnergyPredictionLSTM with config: %s", config)

    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Create a deep LSTM model architecture.

        Args:
            input_shape: Shape of input data (sequence_length, features)

        Returns:
            Compiled Keras model
        """
        logger.info("Building LSTM model with input shape: %s", input_shape)

        # Get model parameters from config
        lstm_units = self.config.get('lstm_units', [128, 64])
        dense_units = self.config.get('dense_units', [32])
        dropout_rate = self.config.get('dropout_rate', 0.2)
        learning_rate = self.config.get('learning_rate', 0.001)
        bidirectional = self.config.get('bidirectional', False)

        # Create sequential model
        model = Sequential()

        # First LSTM layer with return sequences for stacking
        if bidirectional:
            model.add(Bidirectional(
                LSTM(lstm_units[0],
                     activation='tanh',
                     return_sequences=len(lstm_units) > 1),
                input_shape=input_shape
            ))
        else:
            model.add(LSTM(
                lstm_units[0],
                activation='tanh',
                return_sequences=len(lstm_units) > 1,
                input_shape=input_shape
            ))

        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

        # Additional LSTM layers
        for i, units in enumerate(lstm_units[1:]):
            is_last_lstm = i == len(lstm_units) - 2

            if bidirectional:
                model.add(Bidirectional(
                    LSTM(units, activation='tanh', return_sequences=not is_last_lstm)
                ))
            else:
                model.add(LSTM(
                    units, activation='tanh', return_sequences=not is_last_lstm
                ))

            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))

        # Dense layers
        for units in dense_units:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(dropout_rate / 2))

        # Output layer
        model.add(Dense(1, activation='linear'))

        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
        )

        # Save and return the model
        self.model = model
        logger.info("Model compilation completed")
        model.summary(print_fn=logger.info)

        return model

    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray,
             y_val: np.ndarray,
             model_dir: Optional[str] = None) -> tf.keras.callbacks.History:
        """
        Train the LSTM model with callbacks for early stopping and checkpointing.

        Args:
            X_train: Training features (shape: [samples, sequence_length, features])
            y_train: Training target values
            X_val: Validation features
            y_val: Validation target values
            model_dir: Directory to save model checkpoints

        Returns:
            Training history object

        Raises:
            ValueError: If model has not been built
        """
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")

        logger.info("Starting model training with %d training samples and %d validation samples",
                   len(X_train), len(X_val))

        # Get training parameters from config
        epochs = self.config.get('epochs', 100)
        batch_size = self.config.get('batch_size', 32)
        patience = self.config.get('patience', 10)

        # Define callbacks
        callbacks = []

        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)

        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(reduce_lr)

        # Model checkpoint
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
            checkpoint_path = os.path.join(model_dir, "lstm_model_{epoch:02d}_{val_loss:.4f}.h5")
            checkpoint = ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
            callbacks.append(checkpoint)

        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=2
        )

        self.history = history
        logger.info("Model training completed")

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the trained model.

        Args:
            X: Feature data for prediction (shape: [samples, sequence_length, features])

        Returns:
            Array of predictions

        Raises:
            ValueError: If model has not been trained
        """
        if self.model is None:
            raise ValueError("Model has not been built or trained yet.")

        logger.debug("Generating predictions for %d samples", len(X))
        return self.model.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
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
            raise ValueError("Model has not been built or trained yet.")

        logger.info("Evaluating model on %d test samples", len(X_test))

        # Run evaluation
        results = self.model.evaluate(X_test, y_test, verbose=0)

        # Get metric names from model
        metric_names = [loss_name for loss_name in self.model.metrics_names]

        # Create metrics dictionary
        metrics = dict(zip(metric_names, results))

        # Add derived metrics
        if 'mae' in metrics:
            # Calculate MAPE
            y_pred = self.predict(X_test)
            mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100
            metrics['mape'] = mape

        logger.info("Evaluation metrics: %s", metrics)
        return metrics

    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot the training history.

        Args:
            save_path: Path to save the plot, if None plot is displayed

        Raises:
            ValueError: If model has not been trained
        """
        if self.history is None:
            raise ValueError("Model has not been trained yet. No history to plot.")

        plt.figure(figsize=(12, 5))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot metrics (MAE)
        plt.subplot(1, 2, 2)
        if 'mae' in self.history.history:
            plt.plot(self.history.history['mae'], label='Training MAE')
            plt.plot(self.history.history['val_mae'], label='Validation MAE')
            plt.xlabel('Epoch')
            plt.ylabel('Mean Absolute Error')
            plt.title('Training and Validation MAE')
            plt.legend()
            plt.grid(True, alpha=0.3)
        elif 'rmse' in self.history.history:
            plt.plot(self.history.history['rmse'], label='Training RMSE')
            plt.plot(self.history.history['val_rmse'], label='Validation RMSE')
            plt.xlabel('Epoch')
            plt.ylabel('Root Mean Squared Error')
            plt.title('Training and Validation RMSE')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Training history plot saved to %s", save_path)
        else:
            plt.show()

    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.

        Args:
            path: Path to save the model

        Raises:
            ValueError: If model has not been trained
        """
        if self.model is None:
            raise ValueError("Model has not been built or trained yet.")

        # Save the Keras model
        self.model.save(path)

        # Save configuration separately for easier loading
        config_path = os.path.splitext(path)[0] + "_config.npy"
        np.save(config_path, self.config)

        logger.info("Model saved to %s", path)
        logger.info("Configuration saved to %s", config_path)

    def load_model(self, path: str) -> tf.keras.Model:
        """
        Load a saved model from disk.

        Args:
            path: Path to the saved model

        Returns:
            Loaded Keras model
        """
        # Load the Keras model
        self.model = load_model(path)

        # Try to load configuration
        config_path = os.path.splitext(path)[0] + "_config.npy"
        try:
            self.config = np.load(config_path, allow_pickle=True).item()
            logger.info("Configuration loaded from %s", config_path)
        except:
            logger.warning("Could not load model configuration from %s", config_path)

        logger.info("Model loaded from %s", path)
        return self.model

    def get_feature_importance(self,
                             X_sample: np.ndarray,
                             feature_names: List[str],
                             n_permutations: int = 10,
                             plot: bool = True,
                             save_path: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate feature importance using permutation importance method.

        Args:
            X_sample: Sample data to use for importance calculation
            feature_names: Names of features (must match the number of features)
            n_permutations: Number of permutations for each feature
            plot: Whether to create a plot
            save_path: Path to save the plot

        Returns:
            Dictionary of feature importance values

        Raises:
            ValueError: If model has not been trained
        """
        if self.model is None:
            raise ValueError("Model has not been built or trained yet.")

        if len(feature_names) != X_sample.shape[2]:
            raise ValueError(f"Number of feature names ({len(feature_names)}) does not match "
                           f"number of features in data ({X_sample.shape[2]})")

        # Calculate baseline error
        y_pred_baseline = self.model.predict(X_sample)
        y_true = y_pred_baseline  # We use the model's own predictions as ground truth
        baseline_error = np.mean((y_true - y_pred_baseline) ** 2)

        # Calculate importance for each feature
        importance_values = {}

        for i, feature_name in enumerate(feature_names):
            errors = []

            # Perform multiple permutations
            for _ in range(n_permutations):
                # Create a copy of the data
                X_permuted = X_sample.copy()

                # Shuffle the values of the current feature across samples
                permuted_values = X_permuted[:, :, i].flatten()
                np.random.shuffle(permuted_values)
                X_permuted[:, :, i] = permuted_values.reshape(X_permuted[:, :, i].shape)

                # Make predictions with permuted feature
                y_pred_permuted = self.model.predict(X_permuted)

                # Calculate error increase
                permuted_error = np.mean((y_true - y_pred_permuted) ** 2)
                error_increase = permuted_error - baseline_error
                errors.append(error_increase)

            # Average importance across permutations
            importance_values[feature_name] = np.mean(errors)

        # Normalize importance values
        max_importance = max(importance_values.values())
        if max_importance > 0:
            for feature in importance_values:
                importance_values[feature] /= max_importance

        # Sort by importance
        sorted_importance = {k: v for k, v in sorted(importance_values.items(),
                                                    key=lambda item: item[1],
                                                    reverse=True)}

        # Create plot if requested
        if plot:
            plt.figure(figsize=(10, 6))
            features = list(sorted_importance.keys())
            values = list(sorted_importance.values())

            plt.barh(features, values)
            plt.xlabel('Normalized Importance')
            plt.ylabel('Feature')
            plt.title('LSTM Feature Importance (Permutation Method)')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info("Feature importance plot saved to %s", save_path)
            else:
                plt.show()

        return sorted_importance


class AttentionLSTM(EnergyPredictionLSTM):
    """
    Extension of the LSTM model with attention mechanism.

    This class provides attention capability to better handle long sequences
    and focus on the most relevant parts of the input.
    """

    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Create a deep LSTM model architecture with attention mechanism.

        Args:
            input_shape: Shape of input data (sequence_length, features)

        Returns:
            Compiled Keras model
        """
        logger.info("Building Attention LSTM model with input shape: %s", input_shape)

        # Get model parameters from config
        lstm_units = self.config.get('lstm_units', [128, 64])
        dense_units = self.config.get('dense_units', [32])
        dropout_rate = self.config.get('dropout_rate', 0.2)
        learning_rate = self.config.get('learning_rate', 0.001)

        # Model input
        inputs = tf.keras.Input(shape=input_shape)

        # First LSTM layer with return sequences
        if len(lstm_units) > 1:
            x = LSTM(lstm_units[0], activation='tanh', return_sequences=True)(inputs)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)

            # Middle LSTM layers
            for units in lstm_units[1:-1]:
                x = LSTM(units, activation='tanh', return_sequences=True)(x)
                x = BatchNormalization()(x)
                x = Dropout(dropout_rate)(x)

            # Last LSTM layer with return sequences for attention
            x = LSTM(lstm_units[-1], activation='tanh', return_sequences=True)(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)
        else:
            # Single LSTM layer
            x = LSTM(lstm_units[0], activation='tanh', return_sequences=True)(inputs)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)

        # Attention mechanism
        attention = tf.keras.layers.Dense(1, activation='tanh')(x)
        attention = tf.keras.layers.Flatten()(attention)
        attention_weights = tf.keras.layers.Activation('softmax')(attention)

        # Apply attention
        context = tf.keras.layers.Dot(axes=1)([x, tf.keras.layers.Reshape((input_shape[0], 1))(attention_weights)])
        context = tf.keras.layers.Flatten()(context)

        # Dense layers
        for units in dense_units:
            context = Dense(units, activation='relu')(context)
            context = Dropout(dropout_rate / 2)(context)

        # Output layer
        outputs = Dense(1, activation='linear')(context)

        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
        )

        # Save and return the model
        self.model = model
        logger.info("Model compilation completed")
        model.summary(print_fn=logger.info)

        return model

    def visualize_attention(self,
                           X_sample: np.ndarray,
                           sample_idx: int = 0,
                           feature_names: Optional[List[str]] = None,
                           save_path: Optional[str] = None) -> None:
        """
        Visualize attention weights for a sample.

        Args:
            X_sample: Sample data
            sample_idx: Index of the sample to visualize
            feature_names: Names of features
            save_path: Path to save the plot

        Raises:
            ValueError: If model is not an attention model
        """
        if self.model is None:
            raise ValueError("Model has not been built or trained yet.")

        # Extract attention layer
        attention_layer = None
        for layer in self.model.layers:
            if 'activation' in layer.name and isinstance(layer, tf.keras.layers.Activation):
                attention_layer = layer
                break

        if attention_layer is None:
            raise ValueError("Could not find attention layer in the model.")

        # Create a model to get attention weights
        attention_model = Model(inputs=self.model.input,
                                outputs=attention_layer.output)

        # Get attention weights for the sample
        sample = X_sample[sample_idx:sample_idx+1]
        attention_weights = attention_model.predict(sample)[0]

        # Plot attention weights
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(attention_weights)), attention_weights)
        plt.xlabel('Time Step')
        plt.ylabel('Attention Weight')
        plt.title('Attention Weights Distribution')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Attention visualization saved to %s", save_path)
        else:
            plt.show()


# Example usage function
def example_usage():
    """Example usage of the EnergyPredictionLSTM."""
    # Create synthetic data
    sequence_length = 20
    n_features = 10
    n_samples = 1000

    # Create random sequences
    X = np.random.rand(n_samples, sequence_length, n_features)

    # Create target values (simple function of the inputs)
    y = np.sum(X[:, -5:, :2].mean(axis=2), axis=1) * 10

    # Split data
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    # Model configuration
    config = {
        'lstm_units': [64, 32],
        'dense_units': [16],
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'epochs': 20,
        'batch_size': 32,
        'patience': 5
    }

    # Initialize and build model
    model = EnergyPredictionLSTM(config)
    model.build_model(input_shape=(sequence_length, n_features))

    # Train model
    model.train(X_train, y_train, X_val, y_val, model_dir="models/checkpoints")

    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    print(f"Model performance: {metrics}")

    # Plot training history
    model.plot_training_history()

    # Feature importance
    feature_names = [f"Feature_{i}" for i in range(n_features)]
    importance = model.get_feature_importance(
        X_sample=X_test[:100],
        feature_names=feature_names,
        plot=True
    )
    print(f"Feature importance: {importance}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Run example
    example_usage()

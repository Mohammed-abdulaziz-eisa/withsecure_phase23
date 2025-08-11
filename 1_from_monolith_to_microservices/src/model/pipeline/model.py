"""
Model training module for the binary classification pipeline.

This module handles model training, validation, and artifact management
for the Logistic Regression binary classifier.
"""

import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import psutil
from config import settings
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from .preparation import FeatureProcessor, prepare_data


@dataclass
class ModelMetrics:
    """Data class for storing model performance metrics."""

    model_name: str
    cv_roc_auc_mean: float
    cv_roc_auc_std: float
    training_time: float
    n_features_used: int
    n_training_samples: int
    timestamp: str | None = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class BinaryClassificationPipeline:
    """
    Production ML pipeline for binary classification using Logistic Regression.

    Implements MLOps best practices including:
    - Model versioning and serialization
    - Comprehensive logging and monitoring
    - Performance validation
    - Error handling and recovery
    - Artifact management
    - Memory efficiency and health monitoring
    """

    def __init__(self):
        """Initialize the ML pipeline."""
        self.feature_processor = None
        self.model = None
        self.metrics = {}
        self.is_trained = False
        self._last_training_time = None

        # Set random seeds for reproducibility
        np.random.seed(settings.random_state)

        # Create directories
        self._create_directories()

        logger.info("BinaryClassificationPipeline initialized")

    def _create_directories(self) -> None:
        """Create necessary directories for artifacts."""
        Path(settings.model_path).mkdir(parents=True, exist_ok=True)

    def _log_memory_usage(self) -> None:
        """Log current memory usage for monitoring."""
        memory = psutil.virtual_memory()
        logger.info(f"Memory usage: {memory.percent}%")

    def train_and_validate(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train Logistic Regression model with cross-validation and performance monitoring.

        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info("Starting model training and validation...")
        self._log_memory_usage()

        try:
            if X_train.size == 0 or y_train.size == 0:
                raise ValueError("Empty training data or labels")

            # Initialize feature processor
            self.feature_processor = FeatureProcessor()

            # Fit preprocessing pipeline (includes SMOTE resampling)
            X_train_processed, y_train_processed = self.feature_processor.fit(
                X_train, y_train
            )

            # Initialize model with hyperparameters
            self.model = LogisticRegression(
                C=settings.lr_C,
                max_iter=settings.lr_max_iter,
                random_state=settings.random_state,
                class_weight="balanced",
            )

            # Train with cross-validation
            start_time = time.time()

            cv = StratifiedKFold(
                n_splits=settings.n_cv_folds,
                shuffle=True,
                random_state=settings.random_state,
            )
            cv_scores = cross_val_score(
                self.model,
                X_train_processed,
                y_train_processed,
                cv=cv,
                scoring="roc_auc",
            )

            # Train final model on full processed dataset
            self.model.fit(X_train_processed, y_train_processed)

            training_time = time.time() - start_time
            self._last_training_time = training_time

            # Store metrics
            self.metrics = ModelMetrics(
                model_name=settings.model_name,
                cv_roc_auc_mean=cv_scores.mean(),
                cv_roc_auc_std=cv_scores.std(),
                training_time=training_time,
                n_features_used=X_train.shape[1],
                n_training_samples=X_train.shape[0],
            )

            self.is_trained = True

            logger.info("Logistic Regression trained successfully:")
            logger.info(
                f"  ROC-AUC = {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"
            )
            logger.info(f"  Training time: {training_time:.2f}s")

        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise

    def predict(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the trained model.

        Args:
            X_test: Test features

        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        try:
            # Apply preprocessing
            X_test_processed = self.feature_processor.transform(X_test)

            # Make predictions
            predictions = self.model.predict(X_test_processed)
            probabilities = self.model.predict_proba(X_test_processed)

            # Validate predictions
            self._validate_predictions(predictions, probabilities)

            # Log prediction statistics
            self._log_prediction_stats(predictions, probabilities)

            return predictions, probabilities

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def _validate_predictions(
        self, predictions: np.ndarray, probabilities: np.ndarray
    ) -> None:
        """
        Validate prediction outputs.

        Args:
            predictions: Model predictions
            probabilities: Prediction probabilities
        """
        try:
            # Check prediction shape
            if predictions.size == 0:
                raise ValueError("Empty predictions")

            # Check probability shape
            if probabilities.size == 0:
                raise ValueError("Empty probabilities")

            # Check for valid class labels
            unique_predictions = np.unique(predictions)
            valid_labels = [-1, 1]
            if not all(pred in valid_labels for pred in unique_predictions):
                raise ValueError(f"Invalid prediction labels: {unique_predictions}")

            # Check probability sums
            prob_sums = np.sum(probabilities, axis=1)
            if not np.allclose(prob_sums, 1.0, atol=1e-6):
                raise ValueError("Probability sums must equal 1.0")

            logger.info("Prediction validation passed")

        except Exception as e:
            logger.error(f"Prediction validation failed: {str(e)}")
            raise

    def _log_prediction_stats(
        self, predictions: np.ndarray, probabilities: np.ndarray
    ) -> None:
        """
        Log prediction statistics.

        Args:
            predictions: Model predictions
            probabilities: Prediction probabilities
        """
        try:
            n_predictions = len(predictions)
            class_counts = dict(
                zip(*np.unique(predictions, return_counts=True), strict=False)
            )
            avg_prob = np.mean(probabilities, axis=0)

            logger.info("Prediction statistics:")
            logger.info(f"  Total predictions: {n_predictions}")
            logger.info(f"  Class distribution: {class_counts}")
            logger.info(f"  Average probabilities: {avg_prob}")

        except Exception as e:
            logger.error(f"Failed to log prediction stats: {str(e)}")

    def save_artifacts(self) -> None:
        """Save model artifacts as ONNX format."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving artifacts")

        logger.info("Saving ONNX model...")
        self._log_memory_usage()

        try:
            # Create timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Convert sklearn model to ONNX
            # Get the actual number of features from training data
            n_features = self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else 10
            initial_type = [('float_input', FloatTensorType([None, n_features]))]
            
            logger.info(f"Converting model to ONNX with {n_features} features...")
            onnx_model = convert_sklearn(
                self.model, 
                initial_types=initial_type,
                target_opset=11
            )

            # Save ONNX model
            onnx_model_path = Path(settings.model_path) / f"model_{timestamp}.onnx"
            onnx.save(onnx_model, onnx_model_path)

            logger.info(f"ONNX model saved to {onnx_model_path}")
            logger.info("Model artifacts saved successfully")

        except Exception as e:
            logger.error(f"Failed to save ONNX model: {str(e)}")
            raise

    def health_check(self) -> dict[str, Any]:
        """
        Perform health check on the pipeline.

        Returns:
            Dictionary containing health status
        """
        return {
            "is_trained": self.is_trained,
            "model_loaded": self.model is not None,
            "feature_processor_ready": self.feature_processor is not None,
            "last_training_time": self._last_training_time,
            "metrics_available": bool(self.metrics),
        }


def build_model() -> None:
    """Build and train the ML model."""
    logger.info("Starting model building process...")

    try:
        # Prepare data
        X_train, y_train, X_test = prepare_data()

        # Initialize and train pipeline
        pipeline = BinaryClassificationPipeline()
        pipeline.train_and_validate(X_train, y_train)

        # Save artifacts
        pipeline.save_artifacts()

        logger.info("Model building completed successfully")

    except Exception as e:
        logger.error(f"Model building failed: {str(e)}")
        raise

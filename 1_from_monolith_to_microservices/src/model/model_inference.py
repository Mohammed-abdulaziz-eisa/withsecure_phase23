"""
This module provides functionality for making predictions.

It contains the ModelInferenceService class, which offers methods
to load a model from a file, and to make predictions using the loaded model.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import onnxruntime as ort
import pandas as pd
from loguru import logger

from config import settings


class ModelInferenceService:
    """
    A service class for making predictions.

    This class provides functionalities to load a ML model from
    a specified path, and make predictions using the loaded model.

    Attributes:
        onnx_session: ONNX runtime session for model inference.
        model_path: Directory to extract the model from.
        model_name: Name of the saved model to use.

    Methods:
        __init__: Constructor that initializes the ModelInferenceService.
        load_model: Loads the ONNX model from file.
        predict: Makes a prediction using the loaded ONNX model.
        predict_batch: Makes batch predictions using the loaded ONNX model.
        get_model_info: Gets information about the loaded ONNX model.
        health_check: Performs a health check on the model service.
        """

    def __init__(self) -> None:
        """Initialize the ModelInferenceService."""
        self.onnx_session = None
        logger.info("ModelInferenceService initialized")

    def load_model(self, model_name: str | None = None) -> None:
        """
        Load the ONNX model from the specified path.

        Args:
            model_name: Name of the model to load. If None, uses the default model name.

        Raises:
            FileNotFoundError: If the ONNX model file does not exist.
        """
        model_path = Path(settings.model_path)
        logger.info(f'Loading ONNX model from {model_path}')

        if not model_path.exists():
            raise FileNotFoundError(f'Model directory {model_path} does not exist!')

        try:
            # Look for ONNX model files
            onnx_files = list(model_path.glob("model_*.onnx"))

            if not onnx_files:
                raise FileNotFoundError(f'No ONNX model files found in {model_path}')

            # Load the most recent ONNX model
            latest_onnx = max(onnx_files, key=lambda x: x.stat().st_mtime)
            
            # Load ONNX model
            self.onnx_session = ort.InferenceSession(
                str(latest_onnx), 
                providers=["CPUExecutionProvider"]
            )
            
            logger.info(f"Loaded ONNX model from {latest_onnx}")

        except Exception as e:
            logger.error(f"ONNX model loading failed: {str(e)}")
            raise

    def predict(self, input_parameters: list[float]) -> list[int]:
        """
        Make a prediction using the loaded ONNX model.

        Args:
            input_parameters: List of input features for prediction.

        Returns:
            List containing the predicted class.
        """
        if self.onnx_session is None:
            raise ValueError("ONNX model not loaded. Call load_model() first.")

        logger.info('Making ONNX prediction!')

        try:
            # Convert to numpy array and reshape
            X = np.array(input_parameters, dtype=np.float32).reshape(1, -1)

            # Make prediction using ONNX model
            input_name = self.onnx_session.get_inputs()[0].name
            prediction = self.onnx_session.run(None, {input_name: X})[0]
            prediction = prediction.flatten()

            logger.info(f"Prediction completed: {prediction[0]}")
            return prediction.tolist()

        except Exception as e:
            logger.error(f"ONNX prediction failed: {str(e)}")
            raise

    def predict_batch(self, input_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Make batch predictions using the loaded ONNX model.

        Args:
            input_data: Numpy array of input features for batch prediction.

        Returns:
            Tuple of (predictions, probabilities) as numpy arrays.
        """
        if self.onnx_session is None:
            raise ValueError("ONNX model not loaded. Call load_model() first.")

        logger.info('Making ONNX batch predictions!')

        try:
            # Ensure float32 for ONNX compatibility
            input_data = input_data.astype(np.float32)

            # Make predictions using ONNX model
            input_name = self.onnx_session.get_inputs()[0].name
            onnx_outputs = self.onnx_session.run(None, {input_name: input_data})
            
            predictions = onnx_outputs[0].flatten()
            # ONNX sklearn models typically output predictions and probabilities
            if len(onnx_outputs) > 1:
                probabilities = onnx_outputs[1]
            else:
                # If no probabilities, create dummy ones
                probabilities = np.zeros((len(predictions), 2))
                probabilities[predictions == 1, 1] = 1.0
                probabilities[predictions == 0, 0] = 1.0

            logger.info(f"Batch prediction completed: {len(predictions)} predictions")
            return predictions, probabilities

        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the loaded ONNX model.

        Returns:
            Dictionary containing ONNX model information.
        """
        if self.onnx_session is None:
            return {"status": "No ONNX model loaded"}

        # ONNX model info
        inputs = self.onnx_session.get_inputs()
        outputs = self.onnx_session.get_outputs()
        
        model_info = {
            "model_type": "ONNX",
            "format": "onnx",
            "input_name": inputs[0].name if inputs else "unknown",
            "input_shape": inputs[0].shape if inputs else "unknown",
            "input_type": str(inputs[0].type) if inputs else "unknown",
            "output_name": outputs[0].name if outputs else "unknown",
            "output_shape": outputs[0].shape if outputs else "unknown",
            "providers": self.onnx_session.get_providers(),
        }
        
        # Add feature count from input shape
        if inputs and len(inputs[0].shape) >= 2:
            model_info["n_features"] = inputs[0].shape[1]

        return model_info

    def health_check(self) -> dict[str, Any]:
        """
        Perform a health check on the model service.

        Returns:
            Dictionary containing health check results.
        """
        return {
            "onnx_model_loaded": self.onnx_session is not None,
            "model_info": self.get_model_info(),
        }

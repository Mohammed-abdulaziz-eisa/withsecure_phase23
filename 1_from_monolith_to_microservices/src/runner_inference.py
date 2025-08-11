"""
Main application script for running the ML model service.

This script initializes the ModelInferenceService, loads the ML model,
makes predictions based on predefined input parameters, and logs the output.
It demonstrates the typical workflow of using the ModelInferenceService in
a practical application context.
"""

import sys
from pathlib import Path

import numpy as np
from loguru import logger
from model.model_inference import ModelInferenceService
from model.pipeline.collection import load_test_data

# Add the src directory to Python path for proper imports
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


@logger.catch
def main():
    """
    Run the model inference application.

    Loads the model, makes predictions on test data, and demonstrates
    both single and batch prediction capabilities.
    """
    logger.info("Starting model inference service...")

    # Initialize inference service
    inference_service = ModelInferenceService()
    
    # Load the model
    inference_service.load_model()

    # Load test data for batch prediction
    test_data = load_test_data()
    X_test = test_data.values.astype(np.float32)
    logger.info(f"Loaded test data: {X_test.shape}")

    # Make batch predictions
    predictions, probabilities = inference_service.predict_batch(X_test)

    logger.info("Batch prediction results:")
    logger.info(f"  Total predictions: {len(predictions)}")
    logger.info(
        f"  Class distribution: {dict(zip(*np.unique(predictions, return_counts=True), strict=False))}"
    )

    # Example single prediction
    sample_features = X_test[0].tolist()
    single_prediction = inference_service.predict(sample_features)
    logger.info(f"Single prediction: {single_prediction}")

    # Display service health status
    health_status = inference_service.health_check()
    logger.info(f"Service health check: {health_status['onnx_model_loaded']}")
    
    logger.info("Inference service completed successfully")


if __name__ == '__main__':
    main()

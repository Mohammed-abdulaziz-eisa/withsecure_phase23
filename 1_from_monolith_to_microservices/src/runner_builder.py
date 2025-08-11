"""
Main application script for running the ML model builder service.

This script initializes the ModelBuilderService, trains the ML model
and logs the output. It demonstrates the typical workflow of using
the ModelBuilderService in a practical application context.
"""

from loguru import logger

from model.model_builder import ModelBuilderService


@logger.catch
def main():
    """
    Run the model building application.

    Trains a model and saves it to the configured model directory.
    """
    logger.info('Starting model building service...')
    
    # Initialize and run the model builder
    builder_service = ModelBuilderService()
    builder_service.train_model()
    
    logger.info('Model building completed successfully')


if __name__ == '__main__':
    main()

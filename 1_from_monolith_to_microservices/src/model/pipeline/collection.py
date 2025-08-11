"""
Data collection module for the binary classification pipeline.

This module handles loading and validation of training and test datasets.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from config import settings
from loguru import logger


def _load_csv_data(path: str, data_type: str) -> pd.DataFrame:
    """
    Internal function to load CSV data with consistent error handling.

    Args:
        path: Path to the CSV file.
        data_type: Type of data being loaded (for logging).

    Returns:
        DataFrame containing the loaded data.
    """
    logger.info(f"Loading {data_type} from {path}")
    
    try:
        data = pd.read_csv(path, header=None)
        logger.info(f"{data_type.capitalize()} loaded successfully: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Failed to load {data_type} from {path}: {str(e)}")
        raise


def load_data(path: str = None) -> pd.DataFrame:
    """
    Load training data from CSV file.

    Args:
        path: Path to the CSV file. If None, uses default from settings.

    Returns:
        DataFrame containing the loaded data.
    """
    path = path or settings.train_data_path
    return _load_csv_data(path, "training data")


def load_labels(path: str = None) -> np.ndarray:
    """
    Load training labels from CSV file.

    Args:
        path: Path to the labels CSV file. If None, uses default from settings.

    Returns:
        Numpy array containing the labels.
    """
    path = path or settings.train_labels_path
    data = _load_csv_data(path, "training labels")
    return data.values.ravel()


def load_test_data(path: str = None) -> pd.DataFrame:
    """
    Load test data from CSV file.

    Args:
        path: Path to the test data CSV file. If None, uses default from settings.

    Returns:
        DataFrame containing the test data.
    """
    path = path or settings.test_data_path
    return _load_csv_data(path, "test data")


def validate_data_files() -> bool:
    """
    Validate that all required data files exist.

    Returns:
        True if all files exist, False otherwise.
    """
    required_files = [
        settings.train_data_path,
        settings.train_labels_path,
        settings.test_data_path,
    ]

    for file_path in required_files:
        if not Path(file_path).exists():
            logger.error(f"Required file not found: {file_path}")
            return False

    logger.info("All required data files found")
    return True

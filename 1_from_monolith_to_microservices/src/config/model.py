"""
This module sets up the ML model configuration.

It utilizes Pydantic's BaseSettings for configuration management,
allowing settings to be read from environment variables and a .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSettings(BaseSettings):
    """
    ML model configuration settings for the application.

    Attributes:
        model_config (SettingsConfigDict): Model config, loaded from .env file.
        model_path (str): Filesystem path to the model directory.
        model_name (str): Name of the ML model.
        train_data_path (str): Path to training data CSV file.
        train_labels_path (str): Path to training labels CSV file.
        test_data_path (str): Path to test data CSV file.
        random_state (int): Random state for reproducibility.
        n_cv_folds (int): Number of cross-validation folds.
        lr_C (float): Logistic regression regularization parameter.
        lr_max_iter (int): Maximum iterations for logistic regression.
    """

    model_config = SettingsConfigDict(
        env_file="config/.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Core model settings
    model_path: str = "model/models"
    model_name: str = "ml_pipeline"

    # Data paths
    train_data_path: str = "../data/train_data.csv"
    train_labels_path: str = "../data/train_labels.csv"
    test_data_path: str = "../data/test_data.csv"

    # Hyperparameters
    random_state: int = 42
    n_cv_folds: int = 5
    lr_C: float = 0.1
    lr_max_iter: int = 1000


model_settings = ModelSettings()

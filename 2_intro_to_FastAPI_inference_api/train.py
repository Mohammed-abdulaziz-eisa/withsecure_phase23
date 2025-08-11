import click

# import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


@click.command()
@click.option("--train-data", required=True, help="Path to the training data CSV file")
@click.option(
    "--train-labels", required=True, help="Path to the training labels CSV file"
)
@click.option(
    "--onnx-output", default="model/model.onnx", help="Path to save the ONNX model"
)
def main(train_data: str, train_labels: str, onnx_output: str) -> None:
    """
    Trains a classifier on the provided training data and labels,
    and exports the trained model to ONNX format.

    Steps:
      1) Load feature matrix and labels from CSV (no headers)
      2) Train a RobustScaler + LogisticRegression pipeline
      3) Export the fitted pipeline to ONNX and save to onnx_output
    """

    # 1) Load data
    X_df = pd.read_csv(train_data, header=None)
    y_arr = pd.read_csv(train_labels, header=None).values.ravel()

    X = X_df.values.astype(np.float32)

    # 2) Train pipeline
    pipeline = Pipeline(
        steps=[
            ("scaler", RobustScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(X, y_arr)

    # 3) Export to ONNX
    n_features = X.shape[1]
    initial_types = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(pipeline, initial_types=initial_types)

    # Ensure output directory exists
    out_path = Path(onnx_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Model saved to {out_path}")


if __name__ == "__main__":
    main()

import io
from typing import Any, List

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
import pandas as pd
import uvicorn
import fastapi as fastapi_pkg

MODEL_PATH = "model/model.onnx"

app = FastAPI()

# Lazy session init
session: ort.InferenceSession | None = None


# app can start even if model is missing and return clear error
def get_session() -> ort.InferenceSession:
    global session
    if session is None:
        try:
            session = ort.InferenceSession(
                MODEL_PATH, providers=["CPUExecutionProvider"]
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to load ONNX model: {e}"
            )
    return session


class InputProcessor:
    """
    [ADD] Add preprocessing code if needed
    """

    def __init__(self, session):
        self.session = session

    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        # preprocessing is assumed to be embedded in the ONNX pipeline
        # If needed we can implement transformations to match training
        # Ensure numeric types only; raise 400 on conversion failure
        try:
            numeric_df = df.apply(pd.to_numeric, errors="raise")
        except Exception as exc:  # ValueError or ParserError
            raise HTTPException(
                status_code=400, detail=f"Non-numeric value in CSV: {exc}"
            )
        return numeric_df.values.astype(np.float32)


@app.get("/health")
def health() -> dict[str, Any]:
    """
    Liveness/readiness probe. Returns 200 only if the model can be loaded.
    """
    try:
        sess = get_session()
        # Minimal sanity: access inputs to ensure session is usable
        _ = sess.get_inputs()
        return {"status": "ok"}
    except HTTPException as http_exc:
        # Surface as 503 to fail container health checks
        raise HTTPException(status_code=503, detail=str(http_exc.detail))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Health check failed: {exc}")


@app.get("/info")
def info() -> dict[str, Any]:
    """
    Model and runtime metadata.
    """
    try:
        sess = get_session()
        inputs = sess.get_inputs()
        input_name: str = inputs[0].name if inputs else "unknown"
        input_shape: List[Any] = list(inputs[0].shape) if inputs else []

        return {
            "model_path": MODEL_PATH,
            "input_name": input_name,
            "input_shape": input_shape,
            "providers": sess.get_providers(),
            "versions": {
                "fastapi": fastapi_pkg.__version__,
                "numpy": np.__version__,
                "onnxruntime": ort.__version__,
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Info retrieval failed: {exc}")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives a CSV file containing feature data, runs inference using the ONNX model,
    and returns the prediction results.

    The CSV file should not contain a header row. Each row should represent a sample,
    and columns should match the expected feature order.

    Args:
        file (UploadFile): CSV file uploaded via multipart/form-data.

    Returns:
        dict: A dictionary with the key 'prediction' containing the model's output.

    Example response:
    {
        "prediction": [1]
    }
    [ADD] Completed: receive CSV, run ONNX inference, return prediction
    """
    try:
        content = await file.read()
        if content is None or len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Load CSV with no header
        try:
            df = pd.read_csv(io.StringIO(content.decode("utf-8")), header=None)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid CSV content: {exc}")

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV contains no rows")

        sess = get_session()

        # Validate expected number of features if shape is known from ONNX
        onnx_inputs = sess.get_inputs()
        expected_features = None
        if onnx_inputs and len(onnx_inputs[0].shape) >= 2:
            maybe_feat = onnx_inputs[0].shape[1]
            if isinstance(maybe_feat, int):
                expected_features = maybe_feat
        if expected_features is not None and df.shape[1] != expected_features:
            raise HTTPException(
                status_code=400,
                detail=f"CSV has {df.shape[1]} columns; expected {expected_features}",
            )

        processor = InputProcessor(sess)
        input_array = processor.preprocess(df)

        # Build input name mapping
        input_name = onnx_inputs[0].name
        outputs = sess.run(None, {input_name: input_array})

        # Return first output as prediction list
        pred = outputs[0]
        pred_list = pred.tolist() if isinstance(pred, np.ndarray) else [pred]
        return {"prediction": pred_list}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

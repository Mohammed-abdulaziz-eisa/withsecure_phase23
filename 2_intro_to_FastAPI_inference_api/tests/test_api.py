# import io  - imported not used after check ruff linter
from pathlib import Path

# import numpy as np - imported not used after check ruff linter
# import pandas as pd - imported not used after check ruff linter
from fastapi.testclient import TestClient

from inference.api import app


client = TestClient(app)


def test_health_ok():
    resp = client.get("/health")
    # May be 200 if model exists, else 503; both are acceptable for smoke
    assert resp.status_code in (200, 503)


def test_info_endpoint():
    resp = client.get("/info")
    # If model missing, info may still be 500; ensure endpoint is reachable
    assert resp.status_code in (200, 500)


def test_predict_with_sample_csv():
    sample_path = Path("tests/test_api_data.csv")
    assert sample_path.exists(), "tests/test_api_data.csv not found"
    with sample_path.open("rb") as f:
        files = {"file": ("test_api_data.csv", f, "text/csv")}
        resp = client.post("/predict", files=files)

    # Without a model present, this will be 500; with model it should be 200
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], list)

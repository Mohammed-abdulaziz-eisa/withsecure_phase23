# withsecure_phase 2 & 3 
The `withsecure_phase23` project showcases a systematic approach to MLOps architecture evolution:

1. **Monolithic Architecture** → **Microservices Architecture**
2. **Model Serving** → **FastAPI + ONNX Inference API**
3. **Development Pipeline** → **Production-Ready Containerized Service**

## Architecture Evolution

```
┌─────────────────────────────────────────────────────────────┐
│           Phase 1: Original Monolithic                      │
│                 withsecure/src/runner.py                    │
│              (Training + Inference Combined)                │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                Microservices Transformation                 │
│           1_from_monolith_to_microservices/               │
│  ┌─────────────────┐  ┌─────────────────┐                │
│  │ ModelBuilder    │  │ ModelInference  │                │
│  │ Service         │  │ Service         │                │
│  │ (Training)      │  │ (Inference)     │                │
│  └─────────────────┘  └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                FastAPI Inference API                       │
│           2_intro_to_FastAPI_inference_api/               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              FastAPI Service                        │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │   │
│  │  │   /health   │ │    /info    │ │  /predict   │   │   │
│  │  │   Health    │ │ Model Info  │ │ Predictions │   │   │
│  │  │   Check     │ │             │ │             │   │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│                    ONNX Runtime                           │
│                 (model/model.onnx)                        │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
withsecure_phase23/
├── 1_from_monolith_to_microservices/          # Microservices transformation
│   ├── src/
│   │   ├── config/                             # Configuration management
│   │   │   ├── __init__.py                     # Package initialization
│   │   │   ├── logger.py                       # Logging configuration
│   │   │   └── model.py                        # Model configuration
│   │   ├── logs/                               # Application logs
│   │   ├── model/                              # Model services and pipeline
│   │   │   ├── models/                         # Trained model artifacts
│   │   │   ├── pipeline/                       # Data processing pipeline
│   │   │   │   ├── __init__.py                 # Pipeline package
│   │   │   │   ├── collection.py               # Data collection
│   │   │   │   ├── preparation.py              # Data preparation
│   │   │   │   └── model.py                    # ML pipeline
│   │   │   ├── __init__.py                     # Model package
│   │   │   ├── model_builder.py                # Training service
│   │   │   └── model_inference.py              # Inference service
│   │   ├── __init__.py                         # Source package
│   │   ├── runner_builder.py                   # Training service entry point
│   │   └── runner_inference.py                 # Inference service entry point
│   ├── data/                                   # Training and test datasets
│   │   ├── train_data.csv                      # Training features
│   │   ├── train_labels.csv                    # Training labels
│   │   ├── test_data.csv                       # Test features
│   │   └── test_labels.csv                     # Test labels
│   ├── pyproject.toml                          # Poetry dependencies and config
│   ├── poetry.lock                             # Locked dependencies
│   ├── setup.cfg                               # Code quality configuration
│   ├── Makefile                                # Build and run commands
│   └── README.md                               # Microservices documentation
│
├── 2_intro_to_FastAPI_inference_api/          # FastAPI inference service
│   ├── inference/
│   │   ├── __init__.py                         # Package initialization
│   │   └── api.py                              # FastAPI application
│   ├── model/                                  # ONNX model storage
│   ├── tests/
│   │   ├── test_api.py                         # API tests
│   │   └── test_api_data.csv                   # Test data
│   ├── data/                                   # Training datasets
│   │   ├── train_data.csv                      # Training features
│   │   ├── train_labels.csv                    # Training labels
│   │   ├── test_data.csv                       # Test features
│   │   └── test_labels.csv                     # Test labels
│   ├── candidate_package/                      # Candidate evaluation package
│   │   ├── inference/                          # Inference service template
│   │   └── tests/                              # Test templates
│   ├── train.py                                # Training and ONNX conversion
│   ├── pyproject.toml                          # uv dependencies
│   ├── uv.lock                                 # Locked dependencies
│   ├── Dockerfile                              # Container configuration
│   ├── Makefile                                # Build and deployment commands
│   └── README.md                               # FastAPI documentation
│
└── README.md                                   # This file
```

## Usage

### Step 1: Microservices Transformation

To get started, clone the `withsecure_phase23` repository:

```bash
git clone https://github.com/Mohammed-abdulaziz-eisa/withsecure_phase23.git
```

Navigate to the microservices module:
```bash
cd withsecure_phase23/1_from_monolith_to_microservices
```

Install dependencies:
```bash
poetry install
```

Run the services:
```bash
# Training/Model Building
make run_builder

# Inference/Prediction
make run_inference
```

### Step 2: FastAPI Inference API

Navigate to the FastAPI module:
```bash
cd withsecure_phase23/2_intro_to_FastAPI_inference_api
```

Set up the environment:
```bash
make build_env
```

Train the model and convert to ONNX:
```bash
make train
```

Build and run the Docker container:
```bash
make build
make run
```

Test the API:
```bash
make test
```

## Usage Examples

### Microservices Usage

**Training Service:**
```python
from model.model_builder import ModelBuilderService

builder = ModelBuilderService()
builder.train_model()
```

**Inference Service:**
```python
from model.model_inference import ModelInferenceService

inference = ModelInferenceService()
inference.load_model()
prediction = inference.predict([0.1, 0.2, 0.3, ...])
```

### FastAPI Usage

**Health Check:**
```bash
curl http://localhost:8001/health
```

**Model Information:**
```bash
curl http://localhost:8001/info
```

## Development Workflow

### 1. **Code Quality**
```bash
# Microservices
cd 1_from_monolith_to_microservices
make clean

# FastAPI
cd 2_intro_to_FastAPI_inference_api
make lint
```

### 2. **Testing**
```bash
# Microservices
cd 1_from_monolith_to_microservices
poetry run pytest

# FastAPI
cd 2_intro_to_FastAPI_inference_api
make test
```

### 3. **Building and Deployment**
```bash
# FastAPI Docker
cd 2_intro_to_FastAPI_inference_api
make build
make run
```



### Data Flow
1. **Training Data**: `data/train_data.csv` and `data/train_labels.csv`
2. **Model Artifacts**: Trained models saved to `src/model/models/`
3. **ONNX Conversion**: `train.py` converts models to ONNX format



### API Testing with Postman

#### 1. Health Check
- **Method:** GET  
- **URL:** `http://localhost:8001/health`
- **Expected Response:**  
  ```json
  {
    "status": "ok"
  }
  ```

#### 2. Model Information
- **Method:** GET  
- **URL:** `http://localhost:8001/info`
- **Expected Response:**  
  Returns model metadata, input schema, and versioning details.

#### 3. Prediction Endpoint
- **Method:** POST  
- **URL:** `http://localhost:8001/predict`
- **Body:**  
  - Form-data:  
    - Key: `file`  
    - Value: Attach a CSV file containing the input features.
- **Expected Response:**  
  ```json
  {
    "prediction": [1]  # example 
  }
  ```

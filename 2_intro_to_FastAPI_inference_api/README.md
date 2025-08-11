# FastAPI Model Inference API

I should continue phase2 from `1_from_monolith_to_microservices` and continue with calling only `ModelService` 
from model inference. NOT CREATING `train.py` again or all this things but after seeing that assignment have a 
indeed structure i prefer to make it as the company need after that we can discuss how to fully converted from 
`1_from_monolith_to_microservices` with little calling as i mentioned before in my previous project: [Rental 
Price Prediction App to Micro Transformation API Development](https://github.com/Mohammed-abdulaziz-eisa/
Rental-Price-Prediction-App-to-Micro-Transformation-API-Development). you can explore it and see the methodlogy.
## Architecture Overview

This project represents **Phase 3** of the MLOps architecture evolution
```
Phase 2: Microservices Architecture
├── ModelBuilderService (Training)
└── ModelInferenceService (Inference)

Phase 3: FastAPI Inference API
├── Extends ModelInferenceService capabilities
├── Provides RESTful API endpoints
├── Containerized deployment with Docker
└── Production-ready monitoring and health checks
```


## Project Structure

```
2_intro_to_FastAPI_inference_api/
├── inference/                    # FastAPI application
│   ├── __init__.py             # Package initialization
│   └── api.py                  # Main API endpoints
├── model/                       # ONNX model storage
├── data/                        # Training and test datasets
├── tests/                       # Test suite and test data
├── Dockerfile                   # Container configuration
├── Makefile                     # Build and deployment automation
├── pyproject.toml               # Project dependencies and configuration
├── train.py                     # Model training and ONNX conversion
└── README.md                    # This documentation
```

##  Start

### 1. Environment Setup

```bash
# first detecting if uv is available
make ensure_uv

# Clean cache and temporary files if founded 
make clean

# Install dependencies and create virtual environment
make build_env
```

### 2. Model Training

```bash
# Train the model and export to ONNX format
make train
```

**Note**: This step creates the `model/model.onnx` file required for the API to function.

### 3. Build and Run

```bash
# Build Docker container
make build

# Run the inference service
make run
```
-  I change the port becasue i use 8000 in ohter side project 
The API will be available at `http://localhost:8001`

**Port Configuration**: The API runs on port 8001 to avoid conflicts with other services.

### 4. Testing

```bash
# Run the test suite
make test

# Stop the service
make stop
```

## API Endpoints - tested using Postman 

### Health Check
- **GET** `/health` - Service health status
- **Response**: `{"status": "ok"}` (200) or error details (503)

### Model Information
- **GET** `/info` - Model metadata and runtime information
- **Response**: Model path, input specifications, and version details

### Prediction
- **POST** `/predict` - Make predictions from CSV data
- **Input**: CSV file via multipart/form-data
- **Response**: `{"prediction": [results]}`

## Model Details

The service uses a **Logistic Regression with Robust Scaling** pipeline:

- **Preprocessing**: RobustScaler for feature normalization
- **Classifier**: LogisticRegression with balanced class weights
- **Format**: ONNX for efficient inference
- **Input**: Numeric features 
- **Output**: Binary classification predictions

## Docker Deployment

### Build Image
```bash
docker build -t inference_api .
```

### Run Container
```bash
docker run -d -p 8001:8000 --name inference_container inference_api
```

### Health Checks
The container includes automatic health checks that probe the `/health` endpoint every 10 seconds.

## Testing

### Test Suite
- **Unit Tests**: pytest-based test suite
- **Integration Tests**: FastAPI TestClient for API testing
- **Test Data**: Sample CSV files for validation

### Running Tests
```bash
# Run all tests
make test
```


### Workflow Features
- Automatic dependency installation
- Code formatting and linting
- testing
- Docker image building

### Common Issues

1. **Model Not Found**: Ensure `make train` is run before `make build`
2. **Port Conflicts**: Change port mapping in `make run` if 8001 is occupied
3. **Import Errors**: Run tests with `PYTHONPATH=.` for proper module resolution

### Health Check Failures
- Check if model file exists in `model/` directory
- Verify container logs: `docker logs inference_container`
- Ensure training completed successfully


### Monitoring
- Health check endpoints for container orchestration
- Log aggregation and analysis
- Metrics collection via `/info` endpoint


# withsecure From Monolith to Microservices architecture Transformation & Model Inference API

#### Second Phase i called it `From Monolith to Microservices architecture`

in machine learning projects, you always have two main phases: first, you train your model, and then you use this trained model to make predictions (inference). it's important to keep these two steps separate — training and inference should not be mixed together in the same code or service.

so, to make things clean and production-ready, i split the project into two microservices: one for building (training) the model, and another for running inference (making predictions). this way, each part does its own job, and you can update or scale them independently.

the idea is to move from a single big (monolithic) app to a microservices setup, where you have a "model builder" microservice and a "model inference" microservice. this separation makes the system more robust and easier to manage.


## Architecture Overview

The monolithic `ModelService` has been decomposed into two focused services:

### 1. ModelBuilderService (`src/model/model_builder.py`)
- **Responsibility**: Model training and building
- **Functionality**: 
  - Trains ( Logistic Regression with L2 regularization)
  - Saves model artifacts to the specified directory 
  - Handles model versioning and serialization with onnx 

### 2. ModelInferenceService (`src/model/model_inference.py`)
- **Responsibility**: Model loading and inference
- **Functionality**:
  - Loads trained models from disk
  - Performs single and batch predictions
  - Provides health checks and model information

## Key Benefits of This Transformation

1. **Separation of Concerns**: Training and inference are now independent operations
2. **Scalability**: Each service can be scaled independently based on demand
3. **Maintainability**: Clear boundaries make the codebase easier to understand and modify
4. **Deployment Flexibility**: Services can be deployed separately or together
5. **Resource Optimization**: Training can run on GPU instances while inference runs on CPU instances

## Usage

### Development
```bash
make install    # Install dependencies
make clean      # Clean up cache files
``` 

### Training/Model Building
```bash
make run_builder
# or
cd src && poetry run python3 runner_builder.py
```

### Inference/Prediction
```bash
make run_inference
# or
cd src && poetry run python3 runner_inference.py
```


## Project Structure

```
src/
├── config/          # Configuration management
├── logs/            # Application logs
├── model/           # Model services and pipeline
│   ├── models/      # Trained model artifacts
│   └── pipeline/    # Data processing pipeline
├── runner_builder.py    # Model building service entry point
└── runner_inference.py  # Model inference service entry point
```


## Next Steps

This microservices architecture serves as the foundation for the FastAPI inference API in the next module, enabling seamless integration between the training pipeline and the production inference service.

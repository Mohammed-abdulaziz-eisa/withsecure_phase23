"""
Model package for the withsecure microservices transformation.

This package contains the ModelBuilderService and ModelInferenceService
classes that separate training and inference responsibilities.
"""

from .model_builder import ModelBuilderService
from .model_inference import ModelInferenceService

__all__ = ["ModelBuilderService", "ModelInferenceService"]

"""
Inference module facade maintaining backward compatibility.
Delegates to domain models, application services, and Evaluator-Optimizer controller.
"""
from typing import Dict, Any, List, Optional
from src.domain.models import RetrievedContextChunk, InferenceResponse, EvaluationResult
from src.application.evaluator_optimizer import (
    EvaluatorOptimizerController,
    extract_evaluator_json,
    NOT_FOUND_MESSAGE,
)
from src.application.inference_service import (
    InferenceService,
    InferencePipeline,
    LegacyInferencePipelineAdapter,
)

__all__ = [
    "InferencePipeline",
    "InferenceService",
    "LegacyInferencePipelineAdapter",
    "EvaluatorOptimizerController",
    "extract_evaluator_json",
    "NOT_FOUND_MESSAGE",
]

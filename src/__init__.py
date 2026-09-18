"""
Ares-Nexus Local RAG Chatbot package.
Clean Architecture, SOLID Principles, and Evaluator-Optimizer Pattern.
"""
from src.domain.models import (
    DocumentChunk,
    RetrievedContextChunk,
    EvaluationResult,
    InferenceResponse,
    IngestionResult,
)
from src.domain.interfaces import (
    VectorStoreRepository,
    LLMClient,
    EmbeddingGenerator,
    ChunkerStrategy,
)
from src.infrastructure.chunking import ParagraphChunkerStrategy
from src.infrastructure.vector_store import (
    ChromaVectorStoreRepository,
    VectorStoreFactory,
)
from src.infrastructure.llm import (
    OllamaLLMClient,
    OllamaEmbeddingGenerator,
    OllamaService,
    LLMClientFactory,
    EmbeddingGeneratorFactory,
)
from src.application.evaluator_optimizer import (
    EvaluatorOptimizerController,
    extract_evaluator_json,
    NOT_FOUND_MESSAGE,
)
from src.application.ingestion_service import IngestionService, IngestionPipeline
from src.application.inference_service import InferenceService, InferencePipeline

__all__ = [
    # Domain Models
    "DocumentChunk",
    "RetrievedContextChunk",
    "EvaluationResult",
    "InferenceResponse",
    "IngestionResult",
    # Domain Interfaces
    "VectorStoreRepository",
    "LLMClient",
    "EmbeddingGenerator",
    "ChunkerStrategy",
    # Infrastructure
    "ParagraphChunkerStrategy",
    "ChromaVectorStoreRepository",
    "VectorStoreFactory",
    "OllamaLLMClient",
    "OllamaEmbeddingGenerator",
    "OllamaService",
    "LLMClientFactory",
    "EmbeddingGeneratorFactory",
    # Application Services
    "EvaluatorOptimizerController",
    "extract_evaluator_json",
    "NOT_FOUND_MESSAGE",
    "IngestionService",
    "IngestionPipeline",
    "InferenceService",
    "InferencePipeline",
]

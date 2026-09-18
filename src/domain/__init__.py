"""
Domain layer for Ares-Nexus Local RAG Chatbot.
"""
from .models import (
    DocumentChunk,
    RetrievedContextChunk,
    EvaluationResult,
    InferenceResponse,
    IngestionResult,
)
from .interfaces import (
    VectorStoreRepository,
    LLMClient,
    EmbeddingGenerator,
    ChunkerStrategy,
)

__all__ = [
    "DocumentChunk",
    "RetrievedContextChunk",
    "EvaluationResult",
    "InferenceResponse",
    "IngestionResult",
    "VectorStoreRepository",
    "LLMClient",
    "EmbeddingGenerator",
    "ChunkerStrategy",
]

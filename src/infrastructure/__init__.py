"""
Infrastructure layer for Ares-Nexus Local RAG Chatbot.
"""
from .chunking import ParagraphChunkerStrategy
from .vector_store import ChromaVectorStoreRepository, VectorStoreFactory
from .llm import (
    OllamaLLMClient,
    OllamaEmbeddingGenerator,
    OllamaService,
    LLMClientFactory,
    EmbeddingGeneratorFactory,
)

__all__ = [
    "ParagraphChunkerStrategy",
    "ChromaVectorStoreRepository",
    "VectorStoreFactory",
    "OllamaLLMClient",
    "OllamaEmbeddingGenerator",
    "OllamaService",
    "LLMClientFactory",
    "EmbeddingGeneratorFactory",
]

"""
Ollama client module facade maintaining backward compatibility.
Delegates to infrastructure LLM clients and embedding generators.
"""
from src.infrastructure.llm.ollama_client import (
    OllamaLLMClient,
    OllamaEmbeddingGenerator,
    OllamaService,
)

__all__ = [
    "OllamaLLMClient",
    "OllamaEmbeddingGenerator",
    "OllamaService",
]

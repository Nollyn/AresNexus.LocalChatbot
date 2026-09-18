"""
Abstract interfaces adhering to the Interface Segregation and Dependency Inversion Principles.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
from .models import DocumentChunk, RetrievedContextChunk


class VectorStoreRepository(ABC):
    """
    Abstract interface for vector database operations.
    Decouples storage engines (ChromaDB, OpenSearch, pgvector, etc.) from business logic.
    """

    @abstractmethod
    def upsert(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        """
        Upsert document chunks and their corresponding embedding vectors into the collection.

        :param chunks: List of domain DocumentChunk objects.
        :param embeddings: List of float vectors corresponding to each chunk.
        :raises ValueError: If chunks and embeddings lengths do not match.
        """
        pass

    @abstractmethod
    def query_similarity(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        distance_threshold: float = 0.85,
    ) -> List[RetrievedContextChunk]:
        """
        Query the vector store for semantic nearest neighbors matching the query embedding.

        :param query_embedding: Query vector representation.
        :param top_k: Maximum number of closest chunks to retrieve.
        :param distance_threshold: Maximum allowable distance threshold (cosine distance).
        :return: List of retrieved context chunks filtered by distance.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Return the total count of documents/chunks currently indexed in the vector store.
        """
        pass

    @abstractmethod
    def reset_collection(self) -> None:
        """
        Clear or recreate the active collection to allow clean re-indexing.
        """
        pass


class LLMClient(ABC):
    """
    Abstract interface for Large Language Model text generation.
    Enables swappable backends (Ollama, Bedrock, OpenAI, vLLM, etc.).
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate text completion from the language model.

        :param prompt: The main user/orchestrator prompt text.
        :param system_instruction: Optional system persona prompt.
        :param temperature: Sampling temperature (0.0 for deterministic outputs).
        :param format: Optional structured output format constraint (e.g., 'json').
        :param options: Optional backend-specific runtime options.
        :return: Generated string response.
        :raises RuntimeError: If generation fails after retries/fallbacks.
        """
        pass

    @abstractmethod
    def is_healthy(self) -> bool:
        """
        Check if the underlying LLM provider/server is reachable and operational.
        """
        pass


class EmbeddingGenerator(ABC):
    """
    Abstract interface for embedding generation.
    Decouples vector representation providers (Ollama, SentenceTransformers, Bedrock, etc.).
    """

    @abstractmethod
    def generate_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """
        Generate a normalized dense vector embedding for the input text.

        :param text: Input string text to embed.
        :param model: Optional explicit model identifier override.
        :return: Vector embedding as a list of floats.
        :raises ValueError: If input text is empty or invalid.
        :raises RuntimeError: If embedding service fails.
        """
        pass


class ChunkerStrategy(ABC):
    """
    Strategy pattern interface for document chunking and metadata extraction.
    """

    @abstractmethod
    def chunk_document(self, file_path: Path) -> List[DocumentChunk]:
        """
        Parse and partition a file on disk into structured document chunks.

        :param file_path: Path to the target document.
        :return: List of structured DocumentChunk domain objects.
        :raises FileNotFoundError: If file does not exist.
        """
        pass

    @abstractmethod
    def chunk_text(self, text: str, filename: str = "document.txt", timestamp: Optional[str] = None) -> List[DocumentChunk]:
        """
        Partition raw text into structured document chunks.

        :param text: Raw text content to chunk.
        :param filename: Source filename identifier for metadata tracking.
        :param timestamp: Document timestamp or modification time.
        :return: List of structured DocumentChunk domain objects.
        """
        pass

"""
Domain entities and value objects for Ares-Nexus RAG.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass(frozen=True)
class DocumentChunk:
    """
    Represents a discrete semantic chunk extracted from a source document.
    """
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedContextChunk:
    """
    Represents a retrieved context chunk with similarity score and distance metrics.
    """
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    distance: float
    similarity: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization and backward compatibility."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
            "distance": self.distance,
            "similarity": self.similarity,
        }


@dataclass(frozen=True)
class EvaluationResult:
    """
    Represents the structured verification verdict produced by an Evaluator LLM.
    """
    score: float
    feedback: str
    verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "feedback": self.feedback,
            "verified": self.verified,
        }


@dataclass(frozen=True)
class InferenceResponse:
    """
    Encapsulates the complete response payload from the RAG inference pipeline.
    """
    query: str
    answer: str
    retrieved_chunks: List[RetrievedContextChunk]
    iterations: int
    verified: bool
    score: float
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for client output and backward compatibility."""
        return {
            "query": self.query,
            "answer": self.answer,
            "retrieved_chunks": [c.to_dict() for c in self.retrieved_chunks],
            "iterations": self.iterations,
            "verified": self.verified,
            "score": self.score,
            "history": self.history,
        }


@dataclass(frozen=True)
class IngestionResult:
    """
    Represents the summary report of an ingestion execution.
    """
    status: str
    file_path: str
    chunks_ingested: int
    total_collection_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status,
            "file": self.file_path,
            "chunks_ingested": self.chunks_ingested,
            "total_collection_count": self.total_collection_count,
        }

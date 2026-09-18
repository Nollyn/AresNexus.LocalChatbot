"""
Ingestion module facade maintaining backward compatibility.
Delegates to domain strategies, infrastructure repositories, and application services.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.domain.models import DocumentChunk
from src.infrastructure.chunking.paragraph_chunker import ParagraphChunkerStrategy
from src.infrastructure.vector_store.factory import VectorStoreFactory
from src.infrastructure.llm.factory import EmbeddingGeneratorFactory
from src.application.ingestion_service import IngestionService, IngestionPipeline
from src.config import CHROMA_DIR, COLLECTION_NAME, DEFAULT_DOCUMENT_PATH, EMBEDDING_MODEL


class DocumentChunker:
    """Legacy helper facade delegating to ParagraphChunkerStrategy."""

    @staticmethod
    def chunk_document(file_path: Path) -> List[Dict[str, Any]]:
        strategy = ParagraphChunkerStrategy()
        chunks = strategy.chunk_document(file_path)
        return [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "metadata": c.metadata,
            }
            for c in chunks
        ]


class LegacyIngestionPipelineFacade(IngestionService):
    """
    Backward-compatible adapter for legacy IngestionPipeline construction.
    """

    def __init__(
        self,
        chroma_dir: Optional[Path] = None,
        collection_name: str = COLLECTION_NAME,
        ollama_service: Optional[Any] = None,
    ):
        target_dir = chroma_dir or CHROMA_DIR
        vector_repo = VectorStoreFactory.create_vector_store(
            provider="chroma",
            persist_directory=target_dir,
            collection_name=collection_name,
        )
        if ollama_service is not None:
            embedding_gen = ollama_service
        else:
            embedding_gen = EmbeddingGeneratorFactory.create_embedding_generator(
                provider="ollama",
                model_name=EMBEDDING_MODEL,
            )

        super().__init__(
            vector_store=vector_repo,
            embedding_generator=embedding_gen,
            chunker_strategy=ParagraphChunkerStrategy(),
        )
        self.chroma_client = getattr(vector_repo, "_client", None)
        self.collection = getattr(vector_repo, "_collection", None)
        self.collection_name = collection_name
        self.chroma_dir = target_dir
        self.ollama = embedding_gen


__all__ = [
    "DocumentChunker",
    "IngestionPipeline",
    "IngestionService",
    "LegacyIngestionPipelineFacade",
]

"""
Application Ingestion Service.
Orchestrates document parsing, vector embedding generation, and vector store repository persistence.
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union

from src.domain.interfaces import VectorStoreRepository, EmbeddingGenerator, ChunkerStrategy
from src.domain.models import IngestionResult
from src.infrastructure.chunking.paragraph_chunker import ParagraphChunkerStrategy
from src.infrastructure.vector_store.factory import VectorStoreFactory
from src.infrastructure.llm.factory import EmbeddingGeneratorFactory
from src.config import DEFAULT_DOCUMENT_PATH, EMBEDDING_MODEL, CHROMA_DIR, COLLECTION_NAME

logger = logging.getLogger(__name__)


class IngestionService:
    """
    Orchestrates the ingestion lifecycle adhering strictly to the Single Responsibility Principle.
    Depends only on abstract interfaces (VectorStoreRepository, EmbeddingGenerator, ChunkerStrategy).
    """

    def __init__(
        self,
        vector_store: VectorStoreRepository,
        embedding_generator: EmbeddingGenerator,
        chunker_strategy: Optional[ChunkerStrategy] = None,
    ):
        """
        :param vector_store: Target vector store repository for upserting embeddings.
        :param embedding_generator: Provider for calculating vector representations.
        :param chunker_strategy: Document parsing and segmentation strategy.
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.chunker_strategy = chunker_strategy or ParagraphChunkerStrategy()

    def ingest_file(
        self,
        file_path: Optional[Union[str, Path]] = None,
        reset_collection: bool = False,
    ) -> Dict[str, Any]:
        """
        Parse and index a document into the vector repository.

        :param file_path: Path to document file (defaults to DEFAULT_DOCUMENT_PATH).
        :param reset_collection: If True, resets existing collection before indexing.
        :return: Ingestion summary dictionary.
        :raises ValueError: If no chunks can be extracted from the target file.
        :raises FileNotFoundError: If target file does not exist.
        """
        target_path = Path(file_path) if file_path else DEFAULT_DOCUMENT_PATH
        logger.info("Starting ingestion for file: %s (reset=%s)", target_path, reset_collection)

        if reset_collection:
            logger.info("Resetting vector collection before ingestion...")
            self.vector_store.reset_collection()

        chunks = self.chunker_strategy.chunk_document(target_path)
        if not chunks:
            raise ValueError(f"No valid chunks extracted from '{target_path}'")

        embeddings = []
        for chunk in chunks:
            section_id = chunk.metadata.get("section_id", "unknown_section")
            logger.info("Computing embedding for chunk [%s]...", section_id)
            vector = self.embedding_generator.generate_embedding(chunk.text)
            embeddings.append(vector)

        # Upsert into vector repository
        self.vector_store.upsert(chunks=chunks, embeddings=embeddings)

        total_count = self.vector_store.count()
        logger.info("Ingestion complete. Total items in vector store: %d", total_count)

        result = IngestionResult(
            status="success",
            file_path=str(target_path),
            chunks_ingested=len(chunks),
            total_collection_count=total_count,
        )
        return result.to_dict()


class LegacyIngestionPipelineAdapter(IngestionService):
    """
    Adapter preserving 100% backward compatibility for legacy IngestionPipeline construction.
    """

    def __init__(
        self,
        chroma_dir: Optional[Path] = None,
        collection_name: str = COLLECTION_NAME,
        ollama_service: Optional[Any] = None,
        embedding_model: str = EMBEDDING_MODEL,
        vector_store: Optional[VectorStoreRepository] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        chunker_strategy: Optional[ChunkerStrategy] = None,
    ):
        if vector_store is not None and embedding_generator is not None:
            super().__init__(
                vector_store=vector_store,
                embedding_generator=embedding_generator,
                chunker_strategy=chunker_strategy or ParagraphChunkerStrategy(),
            )
            self.chroma_client = getattr(vector_store, "_client", None)
            self.collection = getattr(vector_store, "_collection", None)
            self.collection_name = collection_name
            self.ollama = embedding_generator
            return

        target_dir = chroma_dir or CHROMA_DIR
        repo = VectorStoreFactory.create_vector_store(
            provider="chroma",
            persist_directory=target_dir,
            collection_name=collection_name,
        )

        if ollama_service is not None:
            embed_gen = ollama_service
        else:
            embed_gen = EmbeddingGeneratorFactory.create_embedding_generator(
                provider="ollama",
                model_name=embedding_model,
            )

        super().__init__(
            vector_store=repo,
            embedding_generator=embed_gen,
            chunker_strategy=chunker_strategy or ParagraphChunkerStrategy(),
        )
        self.chroma_client = getattr(repo, "_client", None)
        self.collection = getattr(repo, "_collection", None)
        self.collection_name = collection_name
        self.chroma_dir = target_dir
        self.ollama = embed_gen


IngestionPipeline = LegacyIngestionPipelineAdapter

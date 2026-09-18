"""
ChromaDB Vector Store Repository implementation.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from src.domain.interfaces import VectorStoreRepository
from src.domain.models import DocumentChunk, RetrievedContextChunk

logger = logging.getLogger(__name__)


class ChromaVectorStoreRepository(VectorStoreRepository):
    """
    Concrete VectorStoreRepository implementation backed by ChromaDB.
    Encapsulates all direct ChromaDB client calls, distance calculations, and collection management.
    """

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "ares_nexus_knowledge",
        distance_metric: str = "cosine",
    ):
        """
        :param persist_directory: Path on disk for ChromaDB persistent storage.
        :param collection_name: Name of the vector collection.
        :param distance_metric: Space metric for HNSW index (default: cosine).
        """
        self.persist_directory = persist_directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.distance_metric = distance_metric

        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        self._collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """Retrieve or initialize the ChromaDB collection with configured metric space."""
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric}
        )

    def upsert(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        """
        Upsert document chunks and vectors into ChromaDB.

        :param chunks: List of DocumentChunk domain entities.
        :param embeddings: List of dense float vectors.
        :raises ValueError: If chunks and embeddings lengths do not match.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks count ({len(chunks)}) does not match embeddings count ({len(embeddings)})"
            )

        if not chunks:
            logger.warning("Empty chunk list provided to upsert; skipping.")
            return

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info("Upserted %d items into Chroma collection '%s'", len(chunks), self.collection_name)

    def query_similarity(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        distance_threshold: float = 0.85,
    ) -> List[RetrievedContextChunk]:
        """
        Query nearest neighbors in ChromaDB and filter results by distance threshold.

        :param query_embedding: Query embedding vector.
        :param top_k: Top K results to retrieve.
        :param distance_threshold: Maximum allowable cosine distance.
        :return: List of RetrievedContextChunk entities.
        """
        if not query_embedding:
            return []

        collection_count = self.count()
        if collection_count == 0:
            logger.warning("Query attempted on empty collection '%s'", self.collection_name)
            return []

        effective_top_k = min(top_k, collection_count)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved_chunks: List[RetrievedContextChunk] = []

        if not results or not results.get("documents") or not results["documents"][0]:
            return retrieved_chunks

        doc_list = results["documents"][0]
        meta_list = results["metadatas"][0] if results.get("metadatas") else [{}] * len(doc_list)
        dist_list = results["distances"][0] if results.get("distances") else [0.0] * len(doc_list)
        id_list = results["ids"][0] if results.get("ids") else [f"id_{i}" for i in range(len(doc_list))]

        for chunk_id, doc_text, meta, dist in zip(id_list, doc_list, meta_list, dist_list):
            if dist > distance_threshold:
                logger.debug(
                    "Filtered out chunk '%s' due to distance %.4f > threshold %.4f",
                    chunk_id, dist, distance_threshold
                )
                continue

            # Similarity score: 1 - cosine_distance clamped between 0 and 1
            similarity = max(0.0, min(1.0, 1.0 - dist))
            retrieved_chunks.append(
                RetrievedContextChunk(
                    chunk_id=chunk_id,
                    text=doc_text,
                    metadata=meta,
                    distance=dist,
                    similarity=round(similarity, 4),
                )
            )

        logger.info(
            "Retrieved %d relevant chunks from collection '%s' (threshold: %.2f)",
            len(retrieved_chunks), self.collection_name, distance_threshold
        )
        return retrieved_chunks

    def count(self) -> int:
        """Return total document count in the active collection."""
        return self._collection.count()

    def reset_collection(self) -> None:
        """Delete and recreate the active collection."""
        logger.info("Resetting collection '%s'...", self.collection_name)
        try:
            self._client.delete_collection(name=self.collection_name)
        except Exception as e:
            logger.debug("Collection deletion note: %s", e)
        self._collection = self._get_or_create_collection()

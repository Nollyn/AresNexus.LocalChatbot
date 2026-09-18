"""
Factory pattern implementation for VectorStoreRepository instances.
"""
from pathlib import Path
from typing import Optional, Dict, Any

from src.domain.interfaces import VectorStoreRepository
from src.config import CHROMA_DIR, COLLECTION_NAME
from .chroma_repository import ChromaVectorStoreRepository


class VectorStoreFactory:
    """
    Factory responsible for instantiating concrete VectorStoreRepository implementations.
    """

    @staticmethod
    def create_vector_store(
        provider: str = "chroma",
        persist_directory: Optional[Path] = None,
        collection_name: Optional[str] = None,
        **kwargs: Any,
    ) -> VectorStoreRepository:
        """
        Instantiate a vector store repository based on provider identifier.

        :param provider: Provider name (e.g., 'chroma').
        :param persist_directory: Path for local storage persistence.
        :param collection_name: Collection identifier.
        :param kwargs: Additional provider-specific parameters.
        :return: Configured VectorStoreRepository instance.
        :raises ValueError: If provider is unsupported.
        """
        provider_key = provider.lower().strip()
        if provider_key == "chroma":
            target_dir = persist_directory or CHROMA_DIR
            target_collection = collection_name or COLLECTION_NAME
            return ChromaVectorStoreRepository(
                persist_directory=target_dir,
                collection_name=target_collection,
                **kwargs,
            )
        raise ValueError(f"Unsupported vector store provider: '{provider}'")

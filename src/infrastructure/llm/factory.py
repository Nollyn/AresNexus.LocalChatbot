"""
Factory implementations for LLMClient and EmbeddingGenerator.
"""
from typing import Optional, Any
from src.domain.interfaces import LLMClient, EmbeddingGenerator
from src.config import OLLAMA_HOST, LLM_MODEL, EMBEDDING_MODEL
from .ollama_client import OllamaLLMClient, OllamaEmbeddingGenerator


class LLMClientFactory:
    """
    Factory for producing LLMClient instances based on configuration.
    """

    @staticmethod
    def create_llm_client(
        provider: str = "ollama",
        host: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMClient:
        """
        Instantiate an LLMClient for the chosen provider.

        :param provider: Provider name (default: 'ollama').
        :param host: Server host address.
        :param model_name: Name of the LLM model.
        :param kwargs: Additional provider-specific configurations.
        :return: Configured LLMClient instance.
        :raises ValueError: If provider is not supported.
        """
        provider_key = provider.lower().strip()
        if provider_key == "ollama":
            return OllamaLLMClient(
                host=host or OLLAMA_HOST,
                model_name=model_name or LLM_MODEL,
                **kwargs,
            )
        raise ValueError(f"Unsupported LLM provider: '{provider}'")


class EmbeddingGeneratorFactory:
    """
    Factory for producing EmbeddingGenerator instances.
    """

    @staticmethod
    def create_embedding_generator(
        provider: str = "ollama",
        host: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbeddingGenerator:
        """
        Instantiate an EmbeddingGenerator for the chosen provider.

        :param provider: Provider name (default: 'ollama').
        :param host: Server host address.
        :param model_name: Name of the embedding model.
        :param kwargs: Additional provider-specific configurations.
        :return: Configured EmbeddingGenerator instance.
        :raises ValueError: If provider is not supported.
        """
        provider_key = provider.lower().strip()
        if provider_key == "ollama":
            return OllamaEmbeddingGenerator(
                host=host or OLLAMA_HOST,
                model_name=model_name or EMBEDDING_MODEL,
                **kwargs,
            )
        raise ValueError(f"Unsupported embedding provider: '{provider}'")

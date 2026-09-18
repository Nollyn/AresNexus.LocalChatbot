"""
Ollama infrastructure implementations for LLM generation and vector embedding.
"""
import logging
from typing import List, Optional, Dict, Any
import requests
import ollama

from src.domain.interfaces import LLMClient, EmbeddingGenerator
from src.config import OLLAMA_HOST, LLM_MODEL, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class OllamaLLMClient(LLMClient):
    """
    Concrete LLMClient implementation interacting with a local Ollama instance.
    Features automated SDK execution with resilient HTTP REST endpoint fallback.
    """

    def __init__(
        self,
        host: str = OLLAMA_HOST,
        model_name: str = LLM_MODEL,
        request_timeout: int = 120,
    ):
        """
        :param host: Base URL for the Ollama server.
        :param model_name: Target LLM model name (e.g. 'llama3.2').
        :param request_timeout: Timeout in seconds for LLM generation.
        """
        self.host = host.rstrip("/")
        self.model_name = model_name
        self.request_timeout = request_timeout
        self._client = ollama.Client(host=self.host)

    def is_healthy(self) -> bool:
        """
        Perform a health check against Ollama /api/tags endpoint.
        """
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                available_models = [m.get("name", "") for m in data.get("models", [])]
                logger.info("Ollama is online. Models: %s", available_models)
                return True
            return False
        except Exception as err:
            logger.warning("Ollama health check failed: %s", err)
            return False

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate completion text from Ollama.

        :param prompt: User/instruction prompt.
        :param system_instruction: Optional system instruction / persona.
        :param temperature: Sampling temperature.
        :param format: Output format (e.g., 'json').
        :param options: Additional runtime parameters.
        :return: Trimmed generated text.
        :raises RuntimeError: If generation fails via both SDK and REST.
        """
        gen_options = {"temperature": temperature}
        if options:
            gen_options.update(options)

        # 1. Try Python SDK
        try:
            generate_kwargs: Dict[str, Any] = {
                "model": self.model_name,
                "prompt": prompt,
                "system": system_instruction or "",
                "options": gen_options,
            }
            if format:
                generate_kwargs["format"] = format

            response = self._client.generate(**generate_kwargs)
            if "response" in response:
                return response["response"].strip()
        except Exception as sdk_err:
            logger.warning("Ollama SDK generate call failed (%s), trying HTTP fallback...", sdk_err)

        # 2. Try Direct HTTP REST Endpoint
        try:
            url = f"{self.host}/api/generate"
            payload: Dict[str, Any] = {
                "model": self.model_name,
                "prompt": prompt,
                "system": system_instruction or "",
                "stream": False,
                "options": gen_options,
            }
            if format:
                payload["format"] = format

            resp = requests.post(url, json=payload, timeout=self.request_timeout)
            resp.raise_for_status()
            data = resp.json()
            if "response" in data:
                return data["response"].strip()
            raise RuntimeError(f"Unexpected response structure from Ollama REST generate: {data}")
        except Exception as http_err:
            logger.error("Ollama HTTP REST generation failed: %s", http_err)
            raise RuntimeError(
                f"LLM generation failed for model '{self.model_name}': {http_err}"
            ) from http_err


class OllamaEmbeddingGenerator(EmbeddingGenerator):
    """
    Concrete EmbeddingGenerator implementation using Ollama embeddings API.
    """

    def __init__(
        self,
        host: str = OLLAMA_HOST,
        model_name: str = EMBEDDING_MODEL,
        request_timeout: int = 30,
    ):
        """
        :param host: Base URL for Ollama.
        :param model_name: Embedding model name (e.g. 'nomic-embed-text').
        :param request_timeout: Timeout in seconds for embedding generation.
        """
        self.host = host.rstrip("/")
        self.model_name = model_name
        self.request_timeout = request_timeout
        self._client = ollama.Client(host=self.host)

    def generate_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """
        Generate embedding vector for input text.

        :param text: Input string.
        :param model: Optional model override.
        :return: List of floats representing the embedding vector.
        :raises ValueError: If input text is empty.
        :raises RuntimeError: If embedding fails.
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty or whitespace-only text.")

        target_model = model or self.model_name

        # 1. Try Python SDK
        try:
            response = self._client.embeddings(model=target_model, prompt=text)
            if "embedding" in response and response["embedding"]:
                return response["embedding"]
        except Exception as sdk_err:
            logger.warning("Ollama SDK embeddings call failed (%s), trying HTTP fallback...", sdk_err)

        # 2. Try Direct HTTP REST Endpoint
        try:
            url = f"{self.host}/api/embeddings"
            payload = {"model": target_model, "prompt": text}
            resp = requests.post(url, json=payload, timeout=self.request_timeout)
            resp.raise_for_status()
            data = resp.json()
            if "embedding" in data:
                return data["embedding"]
            raise RuntimeError(f"Unexpected response structure from Ollama REST embeddings: {data}")
        except Exception as http_err:
            logger.error("Ollama HTTP REST embedding failed: %s", http_err)
            raise RuntimeError(
                f"Embedding generation failed for model '{target_model}': {http_err}"
            ) from http_err


class OllamaService(LLMClient, EmbeddingGenerator):
    """
    Composite adapter combining LLM generation and embedding capabilities
    for backward compatibility and convenience.
    """

    def __init__(
        self,
        host: str = OLLAMA_HOST,
        embedding_model: str = EMBEDDING_MODEL,
        llm_model: str = LLM_MODEL,
    ):
        self.host = host.rstrip("/")
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self._llm_client = OllamaLLMClient(host=self.host, model_name=self.llm_model)
        self._embedding_generator = OllamaEmbeddingGenerator(host=self.host, model_name=self.embedding_model)

    def is_healthy(self) -> bool:
        return self._llm_client.is_healthy()

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        options: Optional[Dict[str, Any]] = None,
        format: Optional[str] = None,
    ) -> str:
        """Supports both new signature and legacy parameters."""
        effective_sys = system_instruction if system_instruction is not None else system
        if model and model != self.llm_model:
            temp_client = OllamaLLMClient(host=self.host, model_name=model)
            return temp_client.generate(
                prompt=prompt,
                system_instruction=effective_sys,
                temperature=temperature,
                format=format,
                options=options,
            )
        return self._llm_client.generate(
            prompt=prompt,
            system_instruction=effective_sys,
            temperature=temperature,
            format=format,
            options=options,
        )

    def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """Legacy method name alias for generate_embedding."""
        return self.generate_embedding(text=text, model=model)

    def generate_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        return self._embedding_generator.generate_embedding(text=text, model=model)

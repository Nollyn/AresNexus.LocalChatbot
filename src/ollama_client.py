"""
Ollama client wrapper for embeddings and LLM inference with robust error handling.
"""
import logging
from typing import List, Optional, Dict, Any
import requests
import ollama
from .config import OLLAMA_HOST, EMBEDDING_MODEL, LLM_MODEL

logger = logging.getLogger(__name__)


class OllamaService:
    """Manages interactions with local Ollama instance."""

    def __init__(self, host: str = OLLAMA_HOST, embedding_model: str = EMBEDDING_MODEL, llm_model: str = LLM_MODEL):
        self.host = host.rstrip("/")
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.client = ollama.Client(host=self.host)

    def is_healthy(self) -> bool:
        """Check if Ollama server is responsive and models are available."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                logger.info("Ollama is reachable. Available models: %s", models)
                return True
            return False
        except Exception as e:
            logger.warning("Ollama health check failed: %s", e)
            return False

    def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """
        Generate embedding vector for input text using Ollama.
        """
        target_model = model or self.embedding_model
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")

        try:
            # Try python SDK first
            response = self.client.embeddings(model=target_model, prompt=text)
            if "embedding" in response and response["embedding"]:
                return response["embedding"]
        except Exception as sdk_err:
            logger.warning("Ollama SDK embeddings call failed (%s), trying HTTP endpoint...", sdk_err)

        # Fallback to direct HTTP REST endpoint
        try:
            url = f"{self.host}/api/embeddings"
            payload = {"model": target_model, "prompt": text}
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if "embedding" in data:
                return data["embedding"]
            raise RuntimeError(f"Unexpected response format from Ollama embeddings: {data}")
        except Exception as http_err:
            logger.error("Failed to generate embedding via HTTP endpoint: %s", http_err)
            raise RuntimeError(f"Embedding generation failed for model '{target_model}': {http_err}") from http_err

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate text response from local LLM.
        """
        target_model = model or self.llm_model
        gen_options = {"temperature": temperature}
        if options:
            gen_options.update(options)

        try:
            response = self.client.generate(
                model=target_model,
                prompt=prompt,
                system=system or "",
                options=gen_options,
            )
            if "response" in response:
                return response["response"].strip()
        except Exception as sdk_err:
            logger.warning("Ollama SDK generate call failed (%s), trying HTTP endpoint...", sdk_err)

        # HTTP fallback
        try:
            url = f"{self.host}/api/generate"
            payload = {
                "model": target_model,
                "prompt": prompt,
                "system": system or "",
                "stream": False,
                "options": gen_options,
            }
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            if "response" in data:
                return data["response"].strip()
            raise RuntimeError(f"Unexpected response format from Ollama generate: {data}")
        except Exception as http_err:
            logger.error("Failed to generate completion via HTTP endpoint: %s", http_err)
            raise RuntimeError(f"LLM generation failed for model '{target_model}': {http_err}") from http_err

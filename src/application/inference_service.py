"""
Application Inference Service.
Coordinates semantic retrieval, prompt formatting, and closed-loop Evaluator-Optimizer execution.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.domain.interfaces import VectorStoreRepository, LLMClient, EmbeddingGenerator
from src.domain.models import RetrievedContextChunk, InferenceResponse
from src.application.evaluator_optimizer import EvaluatorOptimizerController, NOT_FOUND_MESSAGE
from src.application.workflow import LangGraphRAGWorkflow, create_evaluator_optimizer_workflow
from src.infrastructure.vector_store.factory import VectorStoreFactory
from src.infrastructure.llm.factory import LLMClientFactory, EmbeddingGeneratorFactory
from src.config import (
    DEFAULT_TOP_K,
    DISTANCE_THRESHOLD,
    MAX_RETRIES,
    TRUST_THRESHOLD,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
)

logger = logging.getLogger(__name__)


class InferenceService:
    """
    High-level orchestration service for RAG inferences.
    Depends strictly on abstract interfaces (VectorStoreRepository, LLMClient, EmbeddingGenerator).
    """

    def __init__(
        self,
        vector_store: VectorStoreRepository,
        llm_client: LLMClient,
        embedding_generator: EmbeddingGenerator,
        evaluator_optimizer: Optional[EvaluatorOptimizerController] = None,
        top_k: int = DEFAULT_TOP_K,
        distance_threshold: float = DISTANCE_THRESHOLD,
    ):
        """
        :param vector_store: Vector repository interface.
        :param llm_client: Large language model interface.
        :param embedding_generator: Embedding generator interface.
        :param evaluator_optimizer: Optional custom Evaluator-Optimizer controller.
        :param top_k: Default maximum chunks to retrieve.
        :param distance_threshold: Maximum distance cutoff for retrieval.
        """
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.embedding_generator = embedding_generator
        self.top_k = top_k
        self.distance_threshold = distance_threshold
        self.evaluator_optimizer = evaluator_optimizer or EvaluatorOptimizerController(
            llm_client=self.llm_client,
            max_retries=MAX_RETRIES,
            trust_threshold=TRUST_THRESHOLD,
        )
        self.workflow = create_evaluator_optimizer_workflow(
            vector_store=self.vector_store,
            llm_client=self.llm_client,
            embedding_generator=self.embedding_generator,
            top_k=self.top_k,
            distance_threshold=self.distance_threshold,
            max_retries=self.evaluator_optimizer.max_retries,
            trust_threshold=self.evaluator_optimizer.trust_threshold,
        )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedContextChunk]:
        """
        Generate query embedding and perform nearest neighbor search against vector store.

        :param query: User question.
        :param top_k: Optional top_k override.
        :return: List of retrieved and filtered RetrievedContextChunk entities.
        """
        if not query or not query.strip():
            return []

        effective_k = top_k or self.top_k
        query_vector = self.embedding_generator.generate_embedding(query.strip())
        retrieved_chunks = self.vector_store.query_similarity(
            query_embedding=query_vector,
            top_k=effective_k,
            distance_threshold=self.distance_threshold,
        )
        return retrieved_chunks

    def query(self, query_text: str) -> Dict[str, Any]:
        """
        Execute full RAG inference with LangGraph StateGraph Evaluator-Optimizer verification.

        :param query_text: User question string.
        :return: Complete response dictionary.
        """
        if not query_text or not query_text.strip():
            return {
                "query": query_text,
                "answer": "Please provide a valid, non-empty query.",
                "retrieved_chunks": [],
                "iterations": 0,
                "verified": False,
                "score": 0.0,
                "history": [],
            }

        final_state = self.workflow.invoke(query_text)
        retrieved_raw = final_state.get("retrieved_context", [])
        converted_chunks: List[RetrievedContextChunk] = []
        for c in retrieved_raw:
            if isinstance(c, RetrievedContextChunk):
                converted_chunks.append(c)
            elif isinstance(c, dict):
                converted_chunks.append(
                    RetrievedContextChunk(
                        chunk_id=c.get("chunk_id", ""),
                        text=c.get("text", ""),
                        metadata=c.get("metadata", {}),
                        distance=c.get("distance", 0.0),
                        similarity=c.get("similarity", 1.0),
                    )
                )

        history = final_state.get("history", [])
        iterations = len(history) or final_state.get("retry_count", 1)

        response = InferenceResponse(
            query=final_state.get("query", query_text),
            answer=final_state.get("final_answer", final_state.get("current_draft", "")),
            retrieved_chunks=converted_chunks,
            iterations=iterations,
            verified=final_state.get("verified", False),
            score=final_state.get("evaluation_score", 0.0),
            history=history,
        )
        return response.to_dict()

    def construct_prompt(self, query: str, retrieved_chunks: Any) -> str:
        """Helper for backwards compatibility with tests and callers."""
        converted_chunks: List[RetrievedContextChunk] = []
        for c in retrieved_chunks:
            if isinstance(c, RetrievedContextChunk):
                converted_chunks.append(c)
            elif isinstance(c, dict):
                converted_chunks.append(
                    RetrievedContextChunk(
                        chunk_id=c.get("chunk_id", ""),
                        text=c.get("text", ""),
                        metadata=c.get("metadata", {}),
                        distance=c.get("distance", 0.0),
                        similarity=c.get("similarity", 1.0),
                    )
                )
        return self.evaluator_optimizer.construct_optimizer_prompt(query, converted_chunks)


class LegacyInferencePipelineAdapter(InferenceService):
    """
    Adapter preserving 100% backward compatibility for legacy InferencePipeline instantiations.
    """

    def __init__(
        self,
        chroma_dir: Optional[Path] = None,
        collection_name: str = COLLECTION_NAME,
        ollama_service: Optional[Any] = None,
        embedding_model: str = EMBEDDING_MODEL,
        llm_model: str = LLM_MODEL,
        top_k: int = DEFAULT_TOP_K,
        distance_threshold: float = DISTANCE_THRESHOLD,
        max_retries: int = MAX_RETRIES,
        trust_threshold: float = TRUST_THRESHOLD,
    ):
        target_dir = chroma_dir or CHROMA_DIR
        vector_repo = VectorStoreFactory.create_vector_store(
            provider="chroma",
            persist_directory=target_dir,
            collection_name=collection_name,
        )

        if ollama_service is not None:
            llm_client = ollama_service
            embedding_gen = ollama_service
        else:
            llm_client = LLMClientFactory.create_llm_client(
                provider="ollama",
                model_name=llm_model,
            )
            embedding_gen = EmbeddingGeneratorFactory.create_embedding_generator(
                provider="ollama",
                model_name=embedding_model,
            )

        evaluator_opt = EvaluatorOptimizerController(
            llm_client=llm_client,
            max_retries=max_retries,
            trust_threshold=trust_threshold,
        )

        super().__init__(
            vector_store=vector_repo,
            llm_client=llm_client,
            embedding_generator=embedding_gen,
            evaluator_optimizer=evaluator_opt,
            top_k=top_k,
            distance_threshold=distance_threshold,
        )
        self.ollama = ollama_service or llm_client
        self.chroma_client = getattr(vector_repo, "_client", None)
        self.collection = getattr(vector_repo, "_collection", None)
        self.max_retries = max_retries
        self.trust_threshold = trust_threshold
        self.llm_model = llm_model
        self.embedding_model = embedding_model


InferencePipeline = LegacyInferencePipelineAdapter

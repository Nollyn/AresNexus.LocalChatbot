"""
LangGraph StateGraph Workflow Orchestrator for Ares-Nexus Local RAG Chatbot.
Implements the multi-agent Evaluator-Optimizer loop as a native, stateful graph topology.
"""
import logging
from typing import Dict, Any, List, Optional, Callable
from langgraph.graph import StateGraph, START, END

from src.domain.interfaces import VectorStoreRepository, LLMClient, EmbeddingGenerator
from src.domain.models import RetrievedContextChunk
from src.domain.state import AgentState
from src.application.evaluator_optimizer import (
    EvaluatorOptimizerController,
    extract_evaluator_json,
    NOT_FOUND_MESSAGE,
)
from src.config import (
    DEFAULT_TOP_K,
    DISTANCE_THRESHOLD,
    MAX_RETRIES,
    TRUST_THRESHOLD,
    SAFETY_FALLBACK_MESSAGE,
    OPTIMIZER_SYSTEM_PROMPT,
    EVALUATOR_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class LangGraphRAGWorkflow:
    """
    Encapsulates the compiled LangGraph StateGraph for multi-agent RAG verification.
    """

    def __init__(
        self,
        vector_store: VectorStoreRepository,
        llm_client: LLMClient,
        embedding_generator: EmbeddingGenerator,
        top_k: int = DEFAULT_TOP_K,
        distance_threshold: float = DISTANCE_THRESHOLD,
        max_retries: int = MAX_RETRIES,
        trust_threshold: float = TRUST_THRESHOLD,
        optimizer_system_prompt: str = OPTIMIZER_SYSTEM_PROMPT,
        evaluator_system_prompt: str = EVALUATOR_SYSTEM_PROMPT,
    ):
        """
        :param vector_store: Concrete or mock VectorStoreRepository implementation.
        :param llm_client: Concrete or mock LLMClient implementation.
        :param embedding_generator: Concrete or mock EmbeddingGenerator implementation.
        :param top_k: Maximum number of semantic chunks to retrieve.
        :param distance_threshold: Maximum allowable distance cutoff.
        :param max_retries: Maximum revision iterations allowed.
        :param trust_threshold: Minimum score required for verified acceptance.
        :param optimizer_system_prompt: System prompt template for Optimizer agent.
        :param evaluator_system_prompt: System prompt template for Evaluator agent.
        """
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.embedding_generator = embedding_generator
        self.top_k = top_k
        self.distance_threshold = distance_threshold
        self.max_retries = max_retries
        self.trust_threshold = trust_threshold
        self.optimizer_system_prompt = optimizer_system_prompt
        self.evaluator_system_prompt = evaluator_system_prompt

        self._controller = EvaluatorOptimizerController(
            llm_client=self.llm_client,
            max_retries=self.max_retries,
            trust_threshold=self.trust_threshold,
            optimizer_prompt_template=self.optimizer_system_prompt,
            evaluator_prompt_template=self.evaluator_system_prompt,
        )

        self.workflow_graph = self._build_graph()
        self.app = self.workflow_graph.compile()

    def node_retrieve(self, state: AgentState) -> Dict[str, Any]:
        """
        Graph Node: Invokes VectorStoreRepository to fetch context chunks based on state query.
        """
        query = (state.get("query") or "").strip()
        if not query:
            return {"retrieved_context": []}

        try:
            query_vector = self.embedding_generator.generate_embedding(query)
            retrieved_chunks = self.vector_store.query_similarity(
                query_embedding=query_vector,
                top_k=self.top_k,
                distance_threshold=self.distance_threshold,
            )
        except Exception as err:
            logger.error("Error during LangGraph node_retrieve: %s", err)
            retrieved_chunks = []

        return {"retrieved_context": retrieved_chunks}

    def node_optimize(self, state: AgentState) -> Dict[str, Any]:
        """
        Graph Node: Invokes LLMClient (Optimizer) to draft or structurally rewrite the response.
        """
        query = state.get("query", "")
        retrieved_raw = state.get("retrieved_context", [])
        retrieved_chunks: List[RetrievedContextChunk] = []

        for item in retrieved_raw:
            if isinstance(item, RetrievedContextChunk):
                retrieved_chunks.append(item)
            elif isinstance(item, dict):
                retrieved_chunks.append(
                    RetrievedContextChunk(
                        chunk_id=item.get("chunk_id", ""),
                        text=item.get("text", ""),
                        metadata=item.get("metadata", {}),
                        distance=item.get("distance", 0.0),
                        similarity=item.get("similarity", 1.0),
                    )
                )

        previous_draft = state.get("current_draft")
        feedback = state.get("evaluation_feedback")
        current_retries = state.get("retry_count", 0)
        iteration_number = current_retries + 1

        print(f"\n[1. BORRADOR / DRAFT] Optimizer generating Draft v{iteration_number} (Iteration {iteration_number}/{self.max_retries})...")
        if feedback:
            print(f"[*] Applying previous feedback to refine response...")

        draft = self._controller.generate_draft(
            query=query,
            retrieved_chunks=retrieved_chunks,
            previous_draft=previous_draft,
            feedback=feedback,
        )
        print(f"[+] [Draft v{iteration_number} Generated]:\n{draft}\n")

        return {
            "current_draft": draft,
            "retry_count": iteration_number,
        }

    def node_evaluate(self, state: AgentState) -> Dict[str, Any]:
        """
        Graph Node: Invokes LLM auditor to execute claim-by-claim evaluation, writing JSON score and feedback to state.
        """
        query = state.get("query", "")
        retrieved_raw = state.get("retrieved_context", [])
        retrieved_chunks: List[RetrievedContextChunk] = []

        for item in retrieved_raw:
            if isinstance(item, RetrievedContextChunk):
                retrieved_chunks.append(item)
            elif isinstance(item, dict):
                retrieved_chunks.append(
                    RetrievedContextChunk(
                        chunk_id=item.get("chunk_id", ""),
                        text=item.get("text", ""),
                        metadata=item.get("metadata", {}),
                        distance=item.get("distance", 0.0),
                        similarity=item.get("similarity", 1.0),
                    )
                )

        draft = state.get("current_draft", "")
        iteration_number = state.get("retry_count", 1)

        print(f"[2. CRITICA / EVALUATOR] Auditing Draft v{iteration_number} against retrieved context...")
        eval_result = self._controller.evaluate_draft(
            query=query,
            retrieved_chunks=retrieved_chunks,
            draft=draft,
        )

        score = eval_result.score
        feedback = eval_result.feedback
        verified = score >= self.trust_threshold

        history = list(state.get("history") or [])
        history.append({
            "iteration": iteration_number,
            "draft": draft,
            "score": score,
            "feedback": feedback,
        })

        if verified:
            print(f"[3. ACEPTACION / ACCEPTED] Score: {score:.2f} >= Threshold ({self.trust_threshold:.2f}) -> ACCEPTED")
            print(f"    Evaluator Feedback: {feedback}")
            final_answer = draft
        else:
            print(f"[!] [RECHAZO / REJECTED] Score: {score:.2f} < Threshold ({self.trust_threshold:.2f}) -> REJECTED")
            print(f"    Evaluator Critique: {feedback}")
            if iteration_number < self.max_retries:
                print(f"[*] [REINTENTO / RETRY] Queueing retry with corrective feedback for Draft v{iteration_number + 1}...")
                final_answer = draft
            else:
                print(f"[!] [LIMITE ALCANZADO / LIMIT REACHED] Max retries ({self.max_retries}) reached without meeting trust threshold.")
                final_answer = SAFETY_FALLBACK_MESSAGE

        return {
            "evaluation_score": score,
            "evaluation_feedback": feedback,
            "verified": verified,
            "final_answer": final_answer,
            "history": history,
        }

    def should_continue(self, state: AgentState) -> str:
        """
        Conditional edge routing function:
        - If score < 0.90 and retry_count < 3 -> return 'node_optimize'
        - If score >= 0.90 or max retries exceeded -> return END
        """
        score = state.get("evaluation_score", 0.0)
        retries = state.get("retry_count", 0)

        if score < self.trust_threshold and retries < self.max_retries:
            return "node_optimize"
        return END

    def _build_graph(self) -> StateGraph:
        """
        Construct and stitch the StateGraph lifecycle:
        START -> node_retrieve -> node_optimize -> node_evaluate -> should_continue -> (node_optimize | END)
        """
        workflow = StateGraph(AgentState)

        # Register formal nodes
        workflow.add_node("node_retrieve", self.node_retrieve)
        workflow.add_node("node_optimize", self.node_optimize)
        workflow.add_node("node_evaluate", self.node_evaluate)

        # Connect linear edges
        workflow.add_edge(START, "node_retrieve")
        workflow.add_edge("node_retrieve", "node_optimize")
        workflow.add_edge("node_optimize", "node_evaluate")

        # Connect conditional routing edge
        workflow.add_conditional_edges(
            "node_evaluate",
            self.should_continue,
            {
                "node_optimize": "node_optimize",
                END: END,
            },
        )

        return workflow

    def invoke(self, inputs: Any) -> AgentState:
        """
        Execute the compiled LangGraph workflow natively.

        :param inputs: Either user query string or initialized AgentState dictionary.
        :return: Final AgentState produced by graph execution.
        """
        if isinstance(inputs, str):
            initial_state: AgentState = {
                "query": inputs,
                "retrieved_context": [],
                "current_draft": "",
                "evaluation_score": 0.0,
                "evaluation_feedback": "",
                "retry_count": 0,
                "verified": False,
                "final_answer": "",
                "history": [],
            }
        elif isinstance(inputs, dict):
            initial_state = {
                "query": inputs.get("query", ""),
                "retrieved_context": inputs.get("retrieved_context", []),
                "current_draft": inputs.get("current_draft", ""),
                "evaluation_score": inputs.get("evaluation_score", 0.0),
                "evaluation_feedback": inputs.get("evaluation_feedback", ""),
                "retry_count": inputs.get("retry_count", 0),
                "verified": inputs.get("verified", False),
                "final_answer": inputs.get("final_answer", ""),
                "history": inputs.get("history", []),
            }
        else:
            raise ValueError(f"Unsupported input type for LangGraph workflow: {type(inputs)}")

        return self.app.invoke(initial_state)


def create_evaluator_optimizer_workflow(
    vector_store: VectorStoreRepository,
    llm_client: LLMClient,
    embedding_generator: EmbeddingGenerator,
    top_k: int = DEFAULT_TOP_K,
    distance_threshold: float = DISTANCE_THRESHOLD,
    max_retries: int = MAX_RETRIES,
    trust_threshold: float = TRUST_THRESHOLD,
) -> LangGraphRAGWorkflow:
    """
    Factory function to initialize and compile the native LangGraph workflow.
    """
    return LangGraphRAGWorkflow(
        vector_store=vector_store,
        llm_client=llm_client,
        embedding_generator=embedding_generator,
        top_k=top_k,
        distance_threshold=distance_threshold,
        max_retries=max_retries,
        trust_threshold=trust_threshold,
    )

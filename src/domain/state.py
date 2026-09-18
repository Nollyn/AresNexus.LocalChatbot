"""
Native LangGraph Agent State definition for Ares-Nexus Evaluator-Optimizer workflow.
"""
from typing import TypedDict, List, Dict, Any, Optional
from src.domain.models import RetrievedContextChunk


class AgentState(TypedDict, total=False):
    """
    Single source of truth tracking state across LangGraph execution cycles:
    - query: Original user prompt / query string.
    - retrieved_context: List of retrieved context chunks from vector repository.
    - current_draft: Current generated answer draft produced by Optimizer.
    - drafts: Accumulative list of all generated draft versions across iterations.
    - evaluation_score: Floating-point evaluation metric (0.0 to 1.0) produced by Evaluator.
    - evaluation_feedback: Detailed textual critique / grounding assessment from Evaluator.
    - feedback_history: Accumulative list of all critique feedbacks across iterations.
    - retry_count: Number of optimization revision cycles completed.
    - verified: Boolean flag indicating if trust threshold was met.
    - stagnated: Boolean flag indicating if convergence stagnated triggering circuit breaker.
    - final_answer: Final vetted answer string or safety fallback.
    - history: Structured audit trace of all intermediate iterations.
    """
    query: str
    retrieved_context: List[Any]
    current_draft: str
    drafts: List[str]
    evaluation_score: float
    evaluation_feedback: str
    feedback_history: List[str]
    retry_count: int
    verified: bool
    stagnated: bool
    final_answer: str
    history: List[Dict[str, Any]]

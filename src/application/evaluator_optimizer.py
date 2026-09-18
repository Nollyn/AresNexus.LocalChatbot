"""
Evaluator-Optimizer Closed-Loop Self-Correction Controller.
Coordinates two-agent iteration: Optimizer (draft generator) and Evaluator (verification judge).
"""
import json
import logging
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator

from src.domain.interfaces import LLMClient
from src.domain.models import RetrievedContextChunk, EvaluationResult
from src.config import (
    MAX_RETRIES,
    TRUST_THRESHOLD,
    SAFETY_FALLBACK_MESSAGE,
    STAGNATION_THRESHOLD,
    STAGNATION_FALLBACK_MESSAGE,
    PARSING_ERROR_FEEDBACK,
    OPTIMIZER_SYSTEM_PROMPT,
    EVALUATOR_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)
NOT_FOUND_MESSAGE = "Information not found in the baseline document."


class EvaluatorResponseSchema(BaseModel):
    """
    Pydantic schema for strictly validating and normalizing Evaluator JSON payload.
    """
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    feedback: str = Field(default="")

    @field_validator("score", mode="before")
    @classmethod
    def validate_score(cls, v: Any) -> float:
        """Validate and clamp numerical score between 0.0 and 1.0."""
        try:
            val = float(v)
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.0

    @field_validator("feedback", mode="before")
    @classmethod
    def validate_feedback(cls, v: Any) -> str:
        """Normalize feedback string."""
        if v is None:
            return ""
        return str(v).strip()


def extract_evaluator_json(raw_text: str) -> EvaluationResult:
    """
    Defensive JSON extractor for Evaluator responses with regex extraction and Pydantic validation.
    Handles raw JSON, markdown-wrapped JSON (```json ... ```), conversational preambles/postscripts,
    and regex boundary extraction between outermost curly braces {}.
    Executes a graceful fallback to score=0.0 and PARSING_ERROR_FEEDBACK on unrecoverable malformed text.

    :param raw_text: Raw string output from Evaluator LLM.
    :return: Validated EvaluationResult object.
    """
    if not raw_text or not str(raw_text).strip():
        return EvaluationResult(
            score=0.0,
            feedback=PARSING_ERROR_FEEDBACK,
            verified=False,
        )

    text = str(raw_text).strip()

    # Outer try-except to guarantee zero graph execution crashes
    try:
        # 1. Direct JSON parse + Pydantic validation
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                model = EvaluatorResponseSchema.model_validate(data)
                return EvaluationResult(
                    score=model.score,
                    feedback=model.feedback or PARSING_ERROR_FEEDBACK,
                    verified=model.score >= TRUST_THRESHOLD,
                )
        except Exception:
            pass

        # 2. Extract from markdown code blocks ```(?:json)? ... ```
        code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if code_block_match:
            block_content = code_block_match.group(1).strip()
            try:
                data = json.loads(block_content)
                if isinstance(data, dict):
                    model = EvaluatorResponseSchema.model_validate(data)
                    return EvaluationResult(
                        score=model.score,
                        feedback=model.feedback or PARSING_ERROR_FEEDBACK,
                        verified=model.score >= TRUST_THRESHOLD,
                    )
            except Exception:
                pass

        # 3. Defensive regex extraction of outermost curly braces { ... }
        json_object_match = re.search(r"(\{[\s\S]*\})", text)
        if json_object_match:
            try:
                data = json.loads(json_object_match.group(1))
                if isinstance(data, dict):
                    model = EvaluatorResponseSchema.model_validate(data)
                    return EvaluationResult(
                        score=model.score,
                        feedback=model.feedback or PARSING_ERROR_FEEDBACK,
                        verified=model.score >= TRUST_THRESHOLD,
                    )
            except Exception:
                pass

        # 4. Fallback regex field extraction for partial/malformed key-values
        score_match = re.search(r'"?score"?\s*[:=]\s*([0-9]*\.?[0-9]+)', text, re.IGNORECASE)
        feedback_match = re.search(
            r'"?feedback"?\s*[:=]\s*["\']?(.*?)["\']?(?:,|\n|\}|$)',
            text,
            re.IGNORECASE | re.MULTILINE,
        )

        if score_match:
            try:
                raw_score = float(score_match.group(1))
                raw_feedback = feedback_match.group(1).strip() if feedback_match else text
                model = EvaluatorResponseSchema(score=raw_score, feedback=raw_feedback)
                return EvaluationResult(
                    score=model.score,
                    feedback=model.feedback or PARSING_ERROR_FEEDBACK,
                    verified=model.score >= TRUST_THRESHOLD,
                )
            except Exception:
                pass

        # Programmatic graceful fallback if text cannot be parsed
        return EvaluationResult(
            score=0.0,
            feedback=PARSING_ERROR_FEEDBACK,
            verified=False,
        )

    except Exception as exc:
        logger.warning("Defensive evaluator JSON parsing encountered exception: %s", exc)
        return EvaluationResult(
            score=0.0,
            feedback=PARSING_ERROR_FEEDBACK,
            verified=False,
        )


def _normalize_evaluation(data: Dict[str, Any]) -> EvaluationResult:
    """Normalize extracted dictionary into EvaluationResult via Pydantic."""
    try:
        model = EvaluatorResponseSchema.model_validate(data)
        return EvaluationResult(
            score=model.score,
            feedback=model.feedback or PARSING_ERROR_FEEDBACK,
            verified=model.score >= TRUST_THRESHOLD,
        )
    except Exception:
        return EvaluationResult(
            score=0.0,
            feedback=PARSING_ERROR_FEEDBACK,
            verified=False,
        )


class EvaluatorOptimizerController:
    """
    Coordinates the iterative generation, evaluation, critique feedback loop,
    and trust threshold decision gate for RAG inferences.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        max_retries: int = MAX_RETRIES,
        trust_threshold: float = TRUST_THRESHOLD,
        optimizer_prompt_template: str = OPTIMIZER_SYSTEM_PROMPT,
        evaluator_prompt_template: str = EVALUATOR_SYSTEM_PROMPT,
    ):
        """
        :param llm_client: Abstract LLMClient for model generations.
        :param max_retries: Maximum revision iterations allowed.
        :param trust_threshold: Minimum evaluation score required for acceptance.
        :param optimizer_prompt_template: System prompt for Optimizer agent.
        :param evaluator_prompt_template: System prompt for Evaluator agent.
        """
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.trust_threshold = trust_threshold
        self.optimizer_prompt_template = optimizer_prompt_template
        self.evaluator_prompt_template = evaluator_prompt_template

    def construct_optimizer_prompt(
        self,
        query: str,
        retrieved_chunks: List[RetrievedContextChunk],
        previous_draft: Optional[str] = None,
        feedback: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Construct structured prompt for Optimizer draft generation with contrastive historical context.

        :param query: User query string.
        :param retrieved_chunks: List of retrieved context chunks.
        :param previous_draft: Optional single previous draft.
        :param feedback: Optional single feedback critique.
        :param history: Optional accumulative history of previous iterations.
        :return: Structured prompt string.
        """
        if not retrieved_chunks:
            context_section = "NO RELEVANT CONTEXT FOUND."
        else:
            formatted_chunks = []
            for i, chunk in enumerate(retrieved_chunks, 1):
                meta = chunk.metadata
                source = meta.get("filename", "unknown_source")
                date = meta.get("timestamp", "unknown_date")
                section = meta.get("section_id", f"section_{i}")
                similarity = chunk.similarity

                formatted_chunk = (
                    f"--- CONTEXT CHUNK {i} ---\n"
                    f"[Source: {source} | Section: {section} | Date: {date} | Relevance Score: {similarity}]\n"
                    f"{chunk.text}\n"
                    f"------------------------"
                )
                formatted_chunks.append(formatted_chunk)
            context_section = "\n\n".join(formatted_chunks)

        prompt_parts = [
            f"CONTEXT:\n{context_section}",
            f"USER QUERY:\n{query}",
        ]

        if history and len(history) > 0:
            history_blocks = []
            for item in history:
                iter_num = item.get("iteration", len(history_blocks) + 1)
                iter_draft = item.get("draft", "")
                iter_score = item.get("score", 0.0)
                iter_feedback = item.get("feedback", "")
                history_blocks.append(
                    f"--- ITERATION {iter_num} (Score: {iter_score:.2f} - REJECTED) ---\n"
                    f"Previous Draft:\n{iter_draft}\n\n"
                    f"Evaluator Critique / Error Diagnosis:\n{iter_feedback}\n"
                    f"------------------------------------------------"
                )
            contrastive_history = "\n\n".join(history_blocks)
            prompt_parts.append(
                f"HISTORICAL REVISION LOG & PREVIOUS FAILURES:\n{contrastive_history}\n\n"
                f"CONTRASTIVE LEARNING & REVISION DIRECTIVES:\n"
                f"- Carefully review ALL previous iterations and the evaluator critiques listed above.\n"
                f"- Do NOT repeat the previous errors, hallucinations, or unsupported claims.\n"
                f"- Explicitly contrast your new draft with the prior rejected attempts to ensure all identified faults are fixed.\n"
                f"- Ensure every claim is strictly grounded in the CONTEXT above.\n"
                f"- Maintain accurate citations."
            )
        elif feedback and previous_draft:
            prompt_parts.append(
                f"PREVIOUS DRAFT (REJECTED BY EVALUATOR):\n{previous_draft}\n\n"
                f"CRITIQUE & FEEDBACK FROM EVALUATOR:\n{feedback}\n\n"
                f"REVISION DIRECTIVES:\n"
                f"- Analyze the critique above and explicitly rewrite the draft to eliminate all identified errors, hallucinations, or missing facts.\n"
                f"- Ensure all claims are strictly grounded in the provided CONTEXT above.\n"
                f"- Maintain accurate citations."
            )

        prompt_parts.append(
            f"INSTRUCTIONS:\n"
            f"- Strictly act as an AI Architect (Optimizer).\n"
            f"- Base your answer ONLY on the context provided above.\n"
            f"- Cite your sources directly in the answer using metadata tags [Source: <filename>, Section: <section_id>, Date: <date>].\n"
            f"- If the context does not contain sufficient facts to answer the question, or if no relevant context is present, respond ONLY with:\n"
            f"  \"{NOT_FOUND_MESSAGE}\""
        )

        return "\n\n".join(prompt_parts)

    def construct_evaluator_prompt(
        self,
        query: str,
        retrieved_chunks: List[RetrievedContextChunk],
        draft: str,
    ) -> str:
        """Construct structured prompt for Evaluator verification."""
        if not retrieved_chunks:
            context_section = "NO RELEVANT CONTEXT PROVIDED."
        else:
            formatted_chunks = []
            for i, chunk in enumerate(retrieved_chunks, 1):
                meta = chunk.metadata
                source = meta.get("filename", "unknown_source")
                date = meta.get("timestamp", "unknown_date")
                section = meta.get("section_id", f"section_{i}")
                similarity = chunk.similarity

                formatted_chunk = (
                    f"--- RAW CONTEXT CHUNK {i} ---\n"
                    f"[Source: {source} | Section: {section} | Date: {date} | Relevance Score: {similarity}]\n"
                    f"{chunk.text}\n"
                    f"------------------------"
                )
                formatted_chunks.append(formatted_chunk)
            context_section = "\n\n".join(formatted_chunks)

        prompt = (
            f"RAW RETRIEVED DATABASE CONTEXT:\n"
            f"{context_section}\n\n"
            f"ORIGINAL USER QUERY:\n"
            f"{query}\n\n"
            f"GENERATED DRAFT TO AUDIT:\n"
            f"{draft}\n\n"
            f"AUDIT INSTRUCTIONS:\n"
            f"1. Audit the draft claim-by-claim strictly against the RAW RETRIEVED DATABASE CONTEXT.\n"
            f"2. Check for hallucinations, extrapolations, or claims not explicitly supported by context.\n"
            f"3. Verify citations match source metadata accurately.\n"
            f"4. SPECIAL RULE: If the draft states '{NOT_FOUND_MESSAGE}' (or that information is absent/not found), and the RAW CONTEXT indeed does NOT contain facts answering the query, this is the expected and correct behavior. You MUST assign score 1.0.\n"
            f"5. Return ONLY a valid JSON object matching this schema:\n"
            f"   {{\"score\": <float between 0.0 and 1.0>, \"feedback\": \"<detailed critique explaining hallucinations/omissions/contradictions or confirming grounding>\"}}"
        )
        return prompt

    def generate_draft(
        self,
        query: str,
        retrieved_chunks: List[RetrievedContextChunk],
        previous_draft: Optional[str] = None,
        feedback: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Invoke Optimizer agent to generate draft with contrastive historical awareness."""
        prompt = self.construct_optimizer_prompt(
            query=query,
            retrieved_chunks=retrieved_chunks,
            previous_draft=previous_draft,
            feedback=feedback,
            history=history,
        )
        return self.llm_client.generate(
            prompt=prompt,
            system_instruction=self.optimizer_prompt_template,
            temperature=0.0,
        )

    def evaluate_draft(
        self,
        query: str,
        retrieved_chunks: List[RetrievedContextChunk],
        draft: str,
    ) -> EvaluationResult:
        """Invoke Evaluator agent to audit draft and return EvaluationResult."""
        try:
            prompt = self.construct_evaluator_prompt(
                query=query,
                retrieved_chunks=retrieved_chunks,
                draft=draft,
            )
            raw_eval = self.llm_client.generate(
                prompt=prompt,
                system_instruction=self.evaluator_prompt_template,
                temperature=0.0,
                format="json",
            )
            return extract_evaluator_json(raw_eval)
        except Exception as e:
            logger.error("Evaluator LLM invocation failed: %s", e)
            return EvaluationResult(
                score=0.0,
                feedback=PARSING_ERROR_FEEDBACK,
                verified=False,
            )

    def execute_loop(
        self,
        query_text: str,
        chunks: List[RetrievedContextChunk],
    ) -> Dict[str, Any]:
        """
        Execute the closed-loop Evaluator-Optimizer lifecycle with stagnation circuit breaker.

        :param query_text: Original user query.
        :param chunks: List of retrieved context chunks.
        :return: Dictionary containing final answer, verified flag, score, iterations, history.
        """
        history: List[Dict[str, Any]] = []
        current_draft: Optional[str] = None
        last_feedback: Optional[str] = None
        verified = False
        stagnated = False
        final_score = 0.0
        final_feedback = ""
        previous_score: Optional[float] = None

        print(f"\n{'='*70}")
        print(f" [*] [EVALUATOR-OPTIMIZER CONTROLLER] Starting Closed-Loop Inference")
        print(f"     Query: {query_text}")
        print(f"     Max Retries: {self.max_retries} | Trust Threshold: {self.trust_threshold:.2f}")
        print(f"{'='*70}")

        for iteration in range(1, self.max_retries + 1):
            draft_name = f"Draft v{iteration}"
            logger.info("Starting Evaluator-Optimizer iteration %d/%d", iteration, self.max_retries)

            # 1. Borrador / Optimizer Generation
            print(f"\n[1. BORRADOR / DRAFT] Optimizer generating {draft_name} (Iteration {iteration}/{self.max_retries})...")
            if history:
                print(f"[*] Applying accumulative history ({len(history)} prior iterations) for contrastive refinement...")
            elif last_feedback:
                print(f"[*] Applying previous feedback to refine response...")

            current_draft = self.generate_draft(
                query=query_text,
                retrieved_chunks=chunks,
                previous_draft=current_draft,
                feedback=last_feedback,
                history=history,
            )
            print(f"[+] [{draft_name} Generated]:\n{current_draft}\n")

            # 2. Crítica / Evaluator Audit
            print(f"[2. CRITICA / EVALUATOR] Auditing {draft_name} against retrieved context...")
            eval_result = self.evaluate_draft(
                query=query_text,
                retrieved_chunks=chunks,
                draft=current_draft,
            )

            score = eval_result.score
            feedback = eval_result.feedback
            final_score = score
            final_feedback = feedback

            history.append({
                "iteration": iteration,
                "draft": current_draft,
                "score": score,
                "feedback": feedback,
            })

            # 3. Decision Gate & Circuit Breakers
            if score >= self.trust_threshold:
                print(f"[3. ACEPTACION / ACCEPTED] Score: {score:.2f} >= Threshold ({self.trust_threshold:.2f}) -> ACCEPTED")
                print(f"    Evaluator Feedback: {feedback}")
                logger.info("Draft v%d accepted with score %.2f", iteration, score)
                verified = True
                break

            print(f"[!] [RECHAZO / REJECTED] Score: {score:.2f} < Threshold ({self.trust_threshold:.2f}) -> REJECTED")
            print(f"    Evaluator Critique: {feedback}")
            logger.warning(
                "Draft v%d failed evaluation (score=%.2f < %.2f): %s",
                iteration, score, self.trust_threshold, feedback
            )

            # Hardware-Frugality Stagnation Circuit Breaker Check
            if previous_score is not None:
                delta = score - previous_score
                if delta < STAGNATION_THRESHOLD:
                    print(
                        f"[!] [CIRCUIT BREAKER] Score delta ({delta:+.2f}) < threshold ({STAGNATION_THRESHOLD:.2f}). "
                        f"Stagnation detected."
                    )
                    stagnated = True
                    break

            previous_score = score

            if iteration < self.max_retries:
                print(f"[*] [REINTENTO / RETRY] Queueing retry with corrective feedback for Draft v{iteration + 1}...")
                last_feedback = feedback
            else:
                print(f"[!] [LIMITE ALCANZADO / LIMIT REACHED] Max retries ({self.max_retries}) reached without meeting trust threshold.")

        if verified:
            final_answer = current_draft or ""
        elif stagnated:
            logger.warning("Refinement stagnated. Triggering stagnation circuit breaker fallback.")
            final_answer = STAGNATION_FALLBACK_MESSAGE
        else:
            logger.warning(
                "Max retries (%d) exhausted without reaching trust threshold (%.2f). Returning safety fallback.",
                self.max_retries, self.trust_threshold
            )
            final_answer = SAFETY_FALLBACK_MESSAGE

        print(f"\n{'='*70}")
        print(f" [*] [EVALUATOR-OPTIMIZER COMPLETE] Status: {'VERIFIED' if verified else 'UNVERIFIED FALLBACK'}")
        print(f"     Final Score: {final_score:.2f} | Iterations: {len(history)}")
        print(f"{'='*70}\n")

        return {
            "query": query_text,
            "answer": final_answer,
            "retrieved_chunks": chunks,
            "iterations": len(history),
            "verified": verified,
            "score": final_score,
            "feedback": final_feedback,
            "history": history,
        }

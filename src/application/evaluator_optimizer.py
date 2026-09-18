"""
Evaluator-Optimizer Closed-Loop Self-Correction Controller.
Coordinates two-agent iteration: Optimizer (draft generator) and Evaluator (verification judge).
"""
import json
import logging
import re
from typing import List, Dict, Any, Optional

from src.domain.interfaces import LLMClient
from src.domain.models import RetrievedContextChunk, EvaluationResult
from src.config import (
    MAX_RETRIES,
    TRUST_THRESHOLD,
    SAFETY_FALLBACK_MESSAGE,
    OPTIMIZER_SYSTEM_PROMPT,
    EVALUATOR_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)
NOT_FOUND_MESSAGE = "Information not found in the baseline document."


def extract_evaluator_json(raw_text: str) -> EvaluationResult:
    """
    Robust JSON extractor for Evaluator responses.
    Handles raw JSON, markdown-wrapped JSON (```json ... ```), preamble/postscript text,
    and regex fallback for malformed responses.

    :param raw_text: Raw string output from Evaluator LLM.
    :return: Parsed EvaluationResult object.
    """
    if not raw_text or not raw_text.strip():
        return EvaluationResult(score=0.0, feedback="Empty response received from Evaluator.", verified=False)

    text = raw_text.strip()

    # 1. Direct JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "score" in data:
            return _normalize_evaluation(data)
    except Exception:
        pass

    # 2. Extract from markdown code blocks ```json ... ``` or ``` ... ```
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if code_block_match:
        block_content = code_block_match.group(1).strip()
        try:
            data = json.loads(block_content)
            if isinstance(data, dict) and "score" in data:
                return _normalize_evaluation(data)
        except Exception:
            pass

    # 3. Search for outermost JSON object { ... }
    json_object_match = re.search(r"(\{[\s\S]*\})", text)
    if json_object_match:
        try:
            data = json.loads(json_object_match.group(1))
            if isinstance(data, dict) and "score" in data:
                return _normalize_evaluation(data)
        except Exception:
            pass

    # 4. Regex fallback if JSON is partial or malformed
    score = 0.0
    feedback = text

    score_match = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', text, re.IGNORECASE)
    if not score_match:
        score_match = re.search(r'score\s*[:=]\s*([0-9]*\.?[0-9]+)', text, re.IGNORECASE)
    if score_match:
        try:
            score = float(score_match.group(1))
        except ValueError:
            score = 0.0

    feedback_match = re.search(r'"feedback"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', text, re.IGNORECASE)
    if not feedback_match:
        feedback_match = re.search(r'feedback\s*[:=]\s*["\']?(.*?)["\']?$', text, re.IGNORECASE | re.MULTILINE)
    if feedback_match:
        feedback = feedback_match.group(1).strip()

    clamped_score = max(0.0, min(1.0, float(score)))
    return EvaluationResult(
        score=clamped_score,
        feedback=str(feedback).strip() if feedback else text,
        verified=clamped_score >= TRUST_THRESHOLD,
    )


def _normalize_evaluation(data: Dict[str, Any]) -> EvaluationResult:
    """Normalize extracted dictionary into EvaluationResult."""
    raw_score = data.get("score", 0.0)
    try:
        score = float(raw_score)
    except (ValueError, TypeError):
        score = 0.0
    score = max(0.0, min(1.0, score))
    feedback = str(data.get("feedback", "")).strip()
    return EvaluationResult(
        score=score,
        feedback=feedback,
        verified=score >= TRUST_THRESHOLD,
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
    ) -> str:
        """Construct structured prompt for Optimizer draft generation."""
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

        if feedback and previous_draft:
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
    ) -> str:
        """Invoke Optimizer agent to generate draft."""
        prompt = self.construct_optimizer_prompt(
            query=query,
            retrieved_chunks=retrieved_chunks,
            previous_draft=previous_draft,
            feedback=feedback,
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
        prompt = self.construct_evaluator_prompt(
            query=query,
            retrieved_chunks=retrieved_chunks,
            draft=draft,
        )
        try:
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
                feedback=f"Evaluator execution failed: {e}",
                verified=False,
            )

    def execute_loop(
        self,
        query_text: str,
        chunks: List[RetrievedContextChunk],
    ) -> Dict[str, Any]:
        """
        Execute the closed-loop Evaluator-Optimizer lifecycle.

        :param query_text: Original user query.
        :param chunks: List of retrieved context chunks.
        :return: Dictionary containing final answer, verified flag, score, iterations, history.
        """
        history: List[Dict[str, Any]] = []
        current_draft: Optional[str] = None
        last_feedback: Optional[str] = None
        verified = False
        final_score = 0.0
        final_feedback = ""

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
            if last_feedback:
                print(f"[*] Applying previous feedback to refine response...")

            current_draft = self.generate_draft(
                query=query_text,
                retrieved_chunks=chunks,
                previous_draft=current_draft,
                feedback=last_feedback,
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

            # 3. Decision Gate
            if score >= self.trust_threshold:
                print(f"[3. ACEPTACION / ACCEPTED] Score: {score:.2f} >= Threshold ({self.trust_threshold:.2f}) -> ACCEPTED")
                print(f"    Evaluator Feedback: {feedback}")
                logger.info("Draft v%d accepted with score %.2f", iteration, score)
                verified = True
                break
            else:
                print(f"[!] [RECHAZO / REJECTED] Score: {score:.2f} < Threshold ({self.trust_threshold:.2f}) -> REJECTED")
                print(f"    Evaluator Critique: {feedback}")
                logger.warning(
                    "Draft v%d failed evaluation (score=%.2f < %.2f): %s",
                    iteration, score, self.trust_threshold, feedback
                )

                if iteration < self.max_retries:
                    print(f"[*] [REINTENTO / RETRY] Queueing retry with corrective feedback for Draft v{iteration + 1}...")
                    last_feedback = feedback
                else:
                    print(f"[!] [LIMITE ALCANZADO / LIMIT REACHED] Max retries ({self.max_retries}) reached without meeting trust threshold.")

        if verified:
            final_answer = current_draft or ""
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

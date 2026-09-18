"""
Inference Pipeline for Ares-Nexus Local RAG Chatbot.
Implements the Evaluator-Optimizer design pattern with closed-loop self-correction,
structured claim-by-claim verification, and ChromaDB retrieval.
"""
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from .config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    DISTANCE_THRESHOLD,
    EMBEDDING_MODEL,
    LLM_MODEL,
    MAX_RETRIES,
    TRUST_THRESHOLD,
    SAFETY_FALLBACK_MESSAGE,
    AI_ARCHITECT_SYSTEM_PROMPT,
    OPTIMIZER_SYSTEM_PROMPT,
    EVALUATOR_SYSTEM_PROMPT,
)
from .ollama_client import OllamaService

logger = logging.getLogger(__name__)

NOT_FOUND_MESSAGE = "Information not found in the baseline document."


def extract_evaluator_json(raw_text: str) -> Dict[str, Any]:
    """
    Robust JSON extractor for Evaluator responses.
    Handles raw JSON, markdown-wrapped JSON (```json ... ```), preamble/postscript text, and regex fallback.
    """
    if not raw_text or not raw_text.strip():
        return {"score": 0.0, "feedback": "Empty response received from Evaluator."}

    text = raw_text.strip()

    # 1. Direct JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "score" in data:
            return _normalize_eval_dict(data)
    except Exception:
        pass

    # 2. Extract from markdown code blocks ```json ... ``` or ``` ... ```
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if code_block_match:
        block_content = code_block_match.group(1).strip()
        try:
            data = json.loads(block_content)
            if isinstance(data, dict) and "score" in data:
                return _normalize_eval_dict(data)
        except Exception:
            pass

    # 3. Search for outermost JSON object { ... }
    json_object_match = re.search(r"(\{[\s\S]*\})", text)
    if json_object_match:
        try:
            data = json.loads(json_object_match.group(1))
            if isinstance(data, dict) and "score" in data:
                return _normalize_eval_dict(data)
        except Exception:
            pass

    # 4. Regex fallback if JSON was malformed or partial
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

    return {
        "score": max(0.0, min(1.0, float(score))),
        "feedback": str(feedback).strip() if feedback else text,
    }


def _normalize_eval_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to ensure score is a clamped float and feedback is a string."""
    raw_score = data.get("score", 0.0)
    try:
        score = float(raw_score)
    except (ValueError, TypeError):
        score = 0.0
    score = max(0.0, min(1.0, score))
    feedback = str(data.get("feedback", "")).strip()
    return {"score": score, "feedback": feedback}


class InferencePipeline:
    """Manages semantic search against ChromaDB and Evaluator-Optimizer closed-loop generation."""

    def __init__(
        self,
        chroma_dir: Optional[Path] = None,
        collection_name: str = COLLECTION_NAME,
        ollama_service: Optional[OllamaService] = None,
        embedding_model: str = EMBEDDING_MODEL,
        llm_model: str = LLM_MODEL,
        top_k: int = DEFAULT_TOP_K,
        distance_threshold: float = DISTANCE_THRESHOLD,
        max_retries: int = MAX_RETRIES,
        trust_threshold: float = TRUST_THRESHOLD,
    ):
        self.chroma_dir = chroma_dir or CHROMA_DIR
        self.collection_name = collection_name
        self.ollama = ollama_service or OllamaService()
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.top_k = top_k
        self.distance_threshold = distance_threshold
        self.max_retries = max_retries
        self.trust_threshold = trust_threshold

        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Embed user query and retrieve top-K most relevant chunks with metadata from ChromaDB.
        """
        k = top_k or self.top_k
        if self.collection.count() == 0:
            logger.warning("Vector collection '%s' is empty.", self.collection_name)
            return []

        # Generate query vector embedding
        query_embedding = self.ollama.get_embedding(query, model=self.embedding_model)

        # Query vector database
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved_chunks: List[Dict[str, Any]] = []
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
            ids = results["ids"][0] if results.get("ids") else [""] * len(docs)

            for doc, meta, dist, cid in zip(docs, metas, distances, ids):
                # Calculate similarity score from cosine distance (cosine sim = 1 - distance)
                similarity = 1.0 - dist if dist is not None else 1.0
                retrieved_chunks.append({
                    "id": cid,
                    "text": doc,
                    "metadata": meta,
                    "distance": dist,
                    "similarity": round(similarity, 4)
                })

        logger.info("Retrieved %d chunks for query '%s'", len(retrieved_chunks), query)
        return retrieved_chunks

    def construct_optimizer_prompt(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        previous_draft: Optional[str] = None,
        feedback: Optional[str] = None,
    ) -> str:
        """
        Construct structured prompt for the Optimizer agent.
        Injects context chunks, user query, and optional previous draft + critique feedback.
        """
        if not retrieved_chunks:
            context_section = "NO RELEVANT CONTEXT FOUND."
        else:
            formatted_chunks = []
            for i, chunk in enumerate(retrieved_chunks, 1):
                meta = chunk.get("metadata", {})
                source = meta.get("filename", "unknown_source")
                date = meta.get("timestamp", "unknown_date")
                section = meta.get("section_id", f"section_{i}")
                similarity = chunk.get("similarity", "N/A")

                formatted_chunk = (
                    f"--- CONTEXT CHUNK {i} ---\n"
                    f"[Source: {source} | Section: {section} | Date: {date} | Relevance Score: {similarity}]\n"
                    f"{chunk['text']}\n"
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

    def construct_prompt(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Backwards-compatible prompt constructor for single-turn calls."""
        return self.construct_optimizer_prompt(query, retrieved_chunks)

    def construct_evaluator_prompt(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        draft: str,
    ) -> str:
        """
        Construct structured verification prompt for the Evaluator agent (Faithfulness Judge).
        """
        if not retrieved_chunks:
            context_section = "NO RELEVANT CONTEXT PROVIDED."
        else:
            formatted_chunks = []
            for i, chunk in enumerate(retrieved_chunks, 1):
                meta = chunk.get("metadata", {})
                source = meta.get("filename", "unknown_source")
                date = meta.get("timestamp", "unknown_date")
                section = meta.get("section_id", f"section_{i}")
                similarity = chunk.get("similarity", "N/A")

                formatted_chunk = (
                    f"--- RAW CONTEXT CHUNK {i} ---\n"
                    f"[Source: {source} | Section: {section} | Date: {date} | Relevance Score: {similarity}]\n"
                    f"{chunk['text']}\n"
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

    def generate_optimizer_draft(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        previous_draft: Optional[str] = None,
        feedback: Optional[str] = None,
    ) -> str:
        """Invoke Optimizer agent to draft or rewrite the answer."""
        prompt = self.construct_optimizer_prompt(
            query=query,
            retrieved_chunks=retrieved_chunks,
            previous_draft=previous_draft,
            feedback=feedback,
        )
        try:
            raw_answer = self.ollama.generate(
                prompt=prompt,
                system=OPTIMIZER_SYSTEM_PROMPT,
                model=self.llm_model,
                temperature=0.0,
            )
            return raw_answer.strip()
        except Exception as e:
            logger.error("Optimizer LLM generation failed: %s", e)
            return f"Error generating answer from Optimizer: {e}"

    def evaluate_draft(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        draft: str,
    ) -> Dict[str, Any]:
        """Invoke Evaluator agent to audit the draft claim-by-claim and return structured JSON."""
        prompt = self.construct_evaluator_prompt(
            query=query,
            retrieved_chunks=retrieved_chunks,
            draft=draft,
        )
        try:
            raw_eval = self.ollama.generate(
                prompt=prompt,
                system=EVALUATOR_SYSTEM_PROMPT,
                model=self.llm_model,
                temperature=0.0,
                format="json",
            )
            parsed_eval = extract_evaluator_json(raw_eval)
            return parsed_eval
        except Exception as e:
            logger.error("Evaluator LLM audit failed: %s", e)
            return {
                "score": 0.0,
                "feedback": f"Evaluator execution failed: {e}",
            }

    def execute_evaluator_optimizer_loop(
        self,
        query_text: str,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Controller function wrapping the Evaluator-Optimizer closed-loop execution path.
        Enforces MAX_RETRIES and TRUST_THRESHOLD with verbose terminal auditing.
        """
        history: List[Dict[str, Any]] = []
        current_draft: Optional[str] = None
        last_feedback: Optional[str] = None
        last_optimizer_prompt = ""
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

            # Step 1: Optimizer Generation
            print(f"\n[1. DRAFT] Optimizer generating {draft_name} (Iteration {iteration}/{self.max_retries})...")
            if last_feedback:
                print(f"[*] Applying previous feedback to refine response...")

            last_optimizer_prompt = self.construct_optimizer_prompt(
                query=query_text,
                retrieved_chunks=chunks,
                previous_draft=current_draft,
                feedback=last_feedback,
            )

            current_draft = self.generate_optimizer_draft(
                query=query_text,
                retrieved_chunks=chunks,
                previous_draft=current_draft,
                feedback=last_feedback,
            )

            print(f"[+] [{draft_name} Generated]:\n{current_draft}\n")

            # Step 2: Evaluator Audit
            print(f"[2. EVALUATOR] Auditing {draft_name} against retrieved context...")
            eval_result = self.evaluate_draft(
                query=query_text,
                retrieved_chunks=chunks,
                draft=current_draft,
            )

            score = eval_result["score"]
            feedback = eval_result["feedback"]
            final_score = score
            final_feedback = feedback

            history.append({
                "iteration": iteration,
                "draft": current_draft,
                "score": score,
                "feedback": feedback,
            })

            # Step 3: Decision Gate (Aceptación vs Reintento)
            if score >= self.trust_threshold:
                print(f"[3. ACCEPTED] Score: {score:.2f} >= Threshold ({self.trust_threshold:.2f}) -> ACCEPTED")
                print(f"    Evaluator Feedback: {feedback}")
                logger.info("Draft v%d accepted with score %.2f", iteration, score)
                verified = True
                break
            else:
                print(f"[!] [REJECTED] Score: {score:.2f} < Threshold ({self.trust_threshold:.2f}) -> REJECTED")
                print(f"    Evaluator Critique: {feedback}")
                logger.warning(
                    "Draft v%d failed evaluation (score=%.2f < %.2f): %s",
                    iteration, score, self.trust_threshold, feedback
                )

                if iteration < self.max_retries:
                    print(f"[*] [RETRY] Queueing retry with corrective feedback for Draft v{iteration + 1}...")
                    last_feedback = feedback
                else:
                    print(f"[!] [LIMIT REACHED] Max retries ({self.max_retries}) reached without meeting trust threshold.")

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
            "confidence_passed": True,
            "prompt": last_optimizer_prompt,
            "score": final_score,
            "feedback": final_feedback,
            "iterations": len(history),
            "history": history,
            "verified": verified,
        }

    def query(self, query_text: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute full RAG pipeline: retrieval, confidence gating, and Evaluator-Optimizer loop.
        """
        if not query_text or not query_text.strip():
            return {
                "query": query_text,
                "answer": "Query cannot be empty.",
                "retrieved_chunks": [],
                "prompt": "",
                "confidence_passed": False,
                "score": 0.0,
                "feedback": "Empty query",
                "iterations": 0,
                "history": [],
                "verified": False,
            }

        chunks = self.retrieve(query_text, top_k=top_k)

        # Confidence gate: check if best retrieved chunk is within acceptable threshold
        best_distance = chunks[0]["distance"] if chunks and "distance" in chunks[0] else 1.0
        if not chunks or (best_distance is not None and best_distance > self.distance_threshold):
            logger.info(
                "Best distance (%s) exceeds threshold (%s). Returning baseline not found.",
                best_distance, self.distance_threshold
            )
            prompt = self.construct_optimizer_prompt(query_text, chunks)
            return {
                "query": query_text,
                "answer": NOT_FOUND_MESSAGE,
                "retrieved_chunks": chunks,
                "confidence_passed": False,
                "prompt": prompt,
                "score": 1.0,
                "feedback": "Query is out of domain or vector distance exceeds threshold.",
                "iterations": 0,
                "history": [],
                "verified": True,
            }

        return self.execute_evaluator_optimizer_loop(query_text=query_text, chunks=chunks)

"""
Comprehensive Test Suite for Ares-Nexus Local RAG Pipeline.
Verifies SOLID Principles, Clean Architecture abstractions, Factory Pattern,
Evaluator-Optimizer self-correction loop, and Integration behavior.
"""
import unittest
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock

from src.config import (
    DEFAULT_DOCUMENT_PATH,
    EMBEDDING_MODEL,
    LLM_MODEL,
    SAFETY_FALLBACK_MESSAGE,
    STAGNATION_THRESHOLD,
    STAGNATION_FALLBACK_MESSAGE,
    PARSING_ERROR_FEEDBACK,
    TRUST_THRESHOLD,
    MAX_RETRIES,
)
from src.domain.models import (
    DocumentChunk,
    RetrievedContextChunk,
    EvaluationResult,
    InferenceResponse,
    IngestionResult,
)
from src.domain.interfaces import (
    VectorStoreRepository,
    LLMClient,
    EmbeddingGenerator,
    ChunkerStrategy,
)
from src.infrastructure.chunking import ParagraphChunkerStrategy
from src.infrastructure.vector_store import ChromaVectorStoreRepository, VectorStoreFactory
from src.infrastructure.llm import (
    OllamaLLMClient,
    OllamaEmbeddingGenerator,
    OllamaService,
    LLMClientFactory,
    EmbeddingGeneratorFactory,
)
from langgraph.graph import START, END
from src.domain.state import AgentState
from src.application.evaluator_optimizer import (
    EvaluatorOptimizerController,
    extract_evaluator_json,
    NOT_FOUND_MESSAGE,
)
from src.application.ingestion_service import IngestionService, IngestionPipeline
from src.application.inference_service import InferenceService, InferencePipeline
from src.application.workflow import LangGraphRAGWorkflow, create_evaluator_optimizer_workflow
from src.ingestion import DocumentChunker


class MockVectorStoreRepository(VectorStoreRepository):
    """In-memory VectorStoreRepository mock for isolated unit testing."""

    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.embeddings: List[List[float]] = []

    def upsert(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings length mismatch.")
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)

    def query_similarity(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        distance_threshold: float = 0.85,
    ) -> List[RetrievedContextChunk]:
        results: List[RetrievedContextChunk] = []
        for i, chunk in enumerate(self.chunks[:top_k]):
            results.append(
                RetrievedContextChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    distance=0.1,
                    similarity=0.9,
                )
            )
        return results

    def count(self) -> int:
        return len(self.chunks)

    def reset_collection(self) -> None:
        self.chunks.clear()
        self.embeddings.clear()


class MockLLMClient(LLMClient):
    """Mock LLMClient returning queued responses."""

    def __init__(self, responses: Optional[List[str]] = None):
        self.responses = list(responses) if responses else ["Mock LLM Response"]
        self.call_history: List[Dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        self.call_history.append({
            "prompt": prompt,
            "system_instruction": system_instruction,
            "format": format,
        })
        if self.responses:
            return self.responses.pop(0)
        return "Default Mock Response"

    def is_healthy(self) -> bool:
        return True


class MockEmbeddingGenerator(EmbeddingGenerator):
    """Mock EmbeddingGenerator returning deterministic vectors."""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def generate_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")
        return [0.1] * self.dimension


class TestAresNexusCleanArchitecture(unittest.TestCase):
    """Test suite covering clean architecture components and SOLID principles."""

    @classmethod
    def setUpClass(cls):
        cls.ollama = OllamaService()
        cls.is_ollama_available = cls.ollama.is_healthy()
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="chroma_test_"))

    @classmethod
    def tearDownClass(cls):
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_01_knowledge_base_file_exists_and_content(self):
        """Verify data/ares_nexus_pattern.txt exists and contains foundational content."""
        self.assertTrue(DEFAULT_DOCUMENT_PATH.exists(), f"File {DEFAULT_DOCUMENT_PATH} does not exist.")
        content = DEFAULT_DOCUMENT_PATH.read_text(encoding="utf-8")
        self.assertIn("Ares-Nexus", content)
        self.assertIn("Non-Executive AI", content)
        self.assertIn("Decision Gate", content)

    def test_02_paragraph_chunker_strategy(self):
        """Verify ParagraphChunkerStrategy produces valid DocumentChunk entities."""
        strategy: ChunkerStrategy = ParagraphChunkerStrategy()
        chunks = strategy.chunk_document(DEFAULT_DOCUMENT_PATH)
        self.assertGreaterEqual(len(chunks), 3)

        for chunk in chunks:
            self.assertIsInstance(chunk, DocumentChunk)
            self.assertTrue(bool(chunk.chunk_id))
            self.assertTrue(bool(chunk.text))
            self.assertEqual(chunk.metadata["filename"], "ares_nexus_pattern.txt")
            self.assertIn("section_id", chunk.metadata)
            self.assertIn("timestamp", chunk.metadata)

    def test_03_vector_store_and_llm_factories(self):
        """Verify Factory Pattern instantiations."""
        vector_repo = VectorStoreFactory.create_vector_store(
            provider="chroma",
            persist_directory=self.temp_dir / "factory_test",
            collection_name="factory_collection",
        )
        self.assertIsInstance(vector_repo, VectorStoreRepository)

        llm_client = LLMClientFactory.create_llm_client(provider="ollama")
        self.assertIsInstance(llm_client, LLMClient)

        embed_gen = EmbeddingGeneratorFactory.create_embedding_generator(provider="ollama")
        self.assertIsInstance(embed_gen, EmbeddingGenerator)

        with self.assertRaises(ValueError):
            VectorStoreFactory.create_vector_store(provider="unknown_provider")

        with self.assertRaises(ValueError):
            LLMClientFactory.create_llm_client(provider="unknown_provider")

    def test_04_dependency_inversion_ingestion_service(self):
        """Verify IngestionService operates solely on abstract interfaces."""
        mock_repo = MockVectorStoreRepository()
        mock_embed = MockEmbeddingGenerator()
        mock_chunker = ParagraphChunkerStrategy()

        service = IngestionService(
            vector_store=mock_repo,
            embedding_generator=mock_embed,
            chunker_strategy=mock_chunker,
        )

        result = service.ingest_file(DEFAULT_DOCUMENT_PATH, reset_collection=True)
        self.assertEqual(result["status"], "success")
        self.assertGreaterEqual(result["chunks_ingested"], 3)
        self.assertEqual(mock_repo.count(), result["chunks_ingested"])

    def test_05_dependency_inversion_inference_service(self):
        """Verify InferenceService operates using injected mock repository and mock LLM."""
        mock_repo = MockVectorStoreRepository()
        mock_repo.upsert(
            chunks=[
                DocumentChunk(
                    chunk_id="chk_1",
                    text="Ares-Nexus uses Non-Executive AI.",
                    metadata={"filename": "pattern.txt", "section_id": "sec_1"},
                )
            ],
            embeddings=[[0.1] * 128],
        )
        mock_embed = MockEmbeddingGenerator()
        mock_llm = MockLLMClient([
            "Ares-Nexus uses Non-Executive AI [Source: pattern.txt, Section: sec_1, Date: 2026].",
            '{"score": 1.0, "feedback": "All claims grounded in context."}'
        ])

        controller = EvaluatorOptimizerController(
            llm_client=mock_llm,
            max_retries=MAX_RETRIES,
            trust_threshold=TRUST_THRESHOLD,
        )

        service = InferenceService(
            vector_store=mock_repo,
            llm_client=mock_llm,
            embedding_generator=mock_embed,
            evaluator_optimizer=controller,
        )

        response = service.query("What AI pattern does Ares-Nexus use?")
        self.assertTrue(response["verified"])
        self.assertEqual(response["score"], 1.0)
        self.assertEqual(response["iterations"], 1)
        self.assertIn("Non-Executive AI", response["answer"])

    def test_06_evaluator_json_extractor_resilience(self):
        """Verify JSON parsing for raw, markdown, preamble, and malformed evaluator outputs."""
        # Clean JSON
        res1 = extract_evaluator_json('{"score": 0.95, "feedback": "Accurate citations."}')
        self.assertEqual(res1.score, 0.95)
        self.assertTrue(res1.verified)

        # Markdown wrapped
        res2 = extract_evaluator_json('```json\n{"score": 0.92, "feedback": "Grounded."}\n```')
        self.assertEqual(res2.score, 0.92)
        self.assertTrue(res2.verified)

        # Surrounding conversational text
        res3 = extract_evaluator_json('Here is my audit:\n{"score": 0.85, "feedback": "Missing date"}\nEnd of review.')
        self.assertEqual(res3.score, 0.85)
        self.assertFalse(res3.verified)

        # Regex fallback for unquoted or key-value format
        res4 = extract_evaluator_json('score: 0.91\nfeedback: Well done.')
        self.assertEqual(res4.score, 0.91)
        self.assertTrue(res4.verified)

        # Empty response
        res5 = extract_evaluator_json('')
        self.assertEqual(res5.score, 0.0)
        self.assertEqual(res5.feedback, PARSING_ERROR_FEEDBACK)
        self.assertFalse(res5.verified)

        # Malformed non-JSON payload
        res6 = extract_evaluator_json('Totally unparseable evaluator response layout with no numbers.')
        self.assertEqual(res6.score, 0.0)
        self.assertEqual(res6.feedback, PARSING_ERROR_FEEDBACK)
        self.assertFalse(res6.verified)

    def test_07_evaluator_optimizer_retry_and_correction_flow(self):
        """Verify EvaluatorOptimizerController retries on rejection and accepts corrected draft."""
        mock_llm = MockLLMClient([
            # Draft 1 (contains hallucination)
            "Ares-Nexus uses MongoDB as its primary store.",
            # Evaluator 1 (rejection)
            '{"score": 0.50, "feedback": "MongoDB is never mentioned in the baseline context."}',
            # Draft 2 (corrected)
            "Ares-Nexus uses ChromaDB as its vector store [Source: ares.txt, Section: sec_1, Date: 2026].",
            # Evaluator 2 (acceptance)
            '{"score": 0.95, "feedback": "Corrected and verified against context."}',
        ])

        controller = EvaluatorOptimizerController(
            llm_client=mock_llm,
            max_retries=3,
            trust_threshold=0.90,
        )

        chunks = [
            RetrievedContextChunk(
                chunk_id="c1",
                text="Ares-Nexus indexes semantic vectors into ChromaDB.",
                metadata={"filename": "ares.txt", "section_id": "sec_1", "timestamp": "2026"},
                distance=0.1,
                similarity=0.9,
            )
        ]

        result = controller.execute_loop(
            query_text="What database does Ares-Nexus use?",
            chunks=chunks,
        )

        self.assertTrue(result["verified"])
        self.assertEqual(result["score"], 0.95)
        self.assertEqual(result["iterations"], 2)
        self.assertIn("ChromaDB", result["answer"])

    def test_08_evaluator_optimizer_max_retries_exhausted_fallback(self):
        """Verify controller triggers safety fallback when retries exceed MAX_RETRIES without acceptance."""
        mock_llm = MockLLMClient([
            # Draft 1 + Rejection
            "Hallucinated draft 1",
            '{"score": 0.40, "feedback": "Unsupported claims."}',
            # Draft 2 + Rejection
            "Hallucinated draft 2",
            '{"score": 0.50, "feedback": "Still unsupported."}',
            # Draft 3 + Rejection
            "Hallucinated draft 3",
            '{"score": 0.60, "feedback": "Still unsupported."}',
        ])

        controller = EvaluatorOptimizerController(
            llm_client=mock_llm,
            max_retries=3,
            trust_threshold=0.90,
        )

        chunks = [
            RetrievedContextChunk(
                chunk_id="c1",
                text="Ares-Nexus architecture principles.",
                metadata={"filename": "doc.txt", "section_id": "sec_1"},
                distance=0.1,
                similarity=0.9,
            )
        ]

        result = controller.execute_loop(
            query_text="Explain the architecture.",
            chunks=chunks,
        )

        self.assertFalse(result["verified"])
        self.assertEqual(result["answer"], SAFETY_FALLBACK_MESSAGE)
        self.assertEqual(result["iterations"], 3)

    def test_09_chroma_repository_live_integration(self):
        """Verify ChromaVectorStoreRepository persistence and similarity query."""
        if not self.is_ollama_available:
            self.skipTest("Ollama service not reachable on 127.0.0.1:11434")

        repo = ChromaVectorStoreRepository(
            persist_directory=self.temp_dir / "chroma_live",
            collection_name="live_test_collection",
        )
        repo.reset_collection()

        chunk = DocumentChunk(
            chunk_id="chk_live_1",
            text="Non-Executive AI prevents autonomous execution.",
            metadata={"filename": "test.txt", "section_id": "sec_neai"},
        )
        embed_gen = OllamaEmbeddingGenerator()
        vector = embed_gen.generate_embedding(chunk.text)

        repo.upsert(chunks=[chunk], embeddings=[vector])
        self.assertEqual(repo.count(), 1)

        retrieved = repo.query_similarity(query_embedding=vector, top_k=1, distance_threshold=0.85)
        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0].chunk_id, "chk_live_1")
        self.assertGreaterEqual(retrieved[0].similarity, 0.90)

    def test_10_legacy_pipeline_backward_compatibility(self):
        """Verify Legacy adapters (IngestionPipeline, InferencePipeline, DocumentChunker)."""
        chunks = DocumentChunker.chunk_document(DEFAULT_DOCUMENT_PATH)
        self.assertIsInstance(chunks, list)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertIsInstance(chunks[0], dict)
        self.assertIn("chunk_id", chunks[0])

        legacy_ingestion = IngestionPipeline(
            chroma_dir=self.temp_dir / "legacy_test",
            collection_name="legacy_coll",
            ollama_service=self.ollama if self.is_ollama_available else MockLLMClient(),
        )
        self.assertTrue(hasattr(legacy_ingestion, "ingest_file"))

        legacy_inference = InferencePipeline(
            chroma_dir=self.temp_dir / "legacy_test",
            collection_name="legacy_coll",
            ollama_service=self.ollama if self.is_ollama_available else MockLLMClient(),
        )
        self.assertTrue(hasattr(legacy_inference, "query"))
        self.assertTrue(hasattr(legacy_inference, "retrieve"))

    def test_11_langgraph_agent_state_schema(self):
        """Verify AgentState TypedDict schema fields and initialization."""
        state: AgentState = {
            "query": "What is Non-Executive AI?",
            "retrieved_context": [],
            "current_draft": "Draft v1 text",
            "drafts": ["Draft v1 text"],
            "evaluation_score": 0.95,
            "evaluation_feedback": "Faithful citation.",
            "feedback_history": ["Faithful citation."],
            "retry_count": 1,
            "verified": True,
            "stagnated": False,
            "final_answer": "Draft v1 text",
            "history": [],
        }
        self.assertEqual(state["query"], "What is Non-Executive AI?")
        self.assertEqual(state["evaluation_score"], 0.95)
        self.assertEqual(state["retry_count"], 1)
        self.assertTrue(state["verified"])
        self.assertFalse(state["stagnated"])
        self.assertEqual(len(state["drafts"]), 1)
        self.assertEqual(len(state["feedback_history"]), 1)
        self.assertIn("retrieved_context", state)
        self.assertIn("current_draft", state)
        self.assertIn("evaluation_feedback", state)

    def test_12_langgraph_nodes_execution_and_state_mutation(self):
        """Verify explicit LangGraph nodes (retrieve, optimize, evaluate) execute and mutate state."""
        mock_repo = MockVectorStoreRepository()
        mock_repo.upsert(
            chunks=[
                DocumentChunk(
                    chunk_id="c_lg_1",
                    text="Decision Gates validate transaction requests before state commits.",
                    metadata={"filename": "pattern.txt", "section_id": "sec_dg"},
                )
            ],
            embeddings=[[0.1] * 128],
        )
        mock_embed = MockEmbeddingGenerator()
        mock_llm = MockLLMClient([
            "Decision Gates act as security proxies [Source: pattern.txt, Section: sec_dg, Date: 2026].",
            '{"score": 0.98, "feedback": "Accurately grounded in context."}'
        ])

        workflow = LangGraphRAGWorkflow(
            vector_store=mock_repo,
            llm_client=mock_llm,
            embedding_generator=mock_embed,
            max_retries=3,
            trust_threshold=0.90,
        )

        # 1. Test node_retrieve
        initial_state: AgentState = {"query": "How do Decision Gates work?"}
        retrieval_output = workflow.node_retrieve(initial_state)
        self.assertIn("retrieved_context", retrieval_output)
        self.assertEqual(len(retrieval_output["retrieved_context"]), 1)

        # 2. Test node_optimize
        optimize_state: AgentState = {
            "query": "How do Decision Gates work?",
            "retrieved_context": retrieval_output["retrieved_context"],
            "retry_count": 0,
        }
        optimize_output = workflow.node_optimize(optimize_state)
        self.assertIn("current_draft", optimize_output)
        self.assertIn("drafts", optimize_output)
        self.assertEqual(optimize_output["retry_count"], 1)
        self.assertIn("Decision Gates", optimize_output["current_draft"])

        # 3. Test node_evaluate
        evaluate_state: AgentState = {
            "query": "How do Decision Gates work?",
            "retrieved_context": retrieval_output["retrieved_context"],
            "current_draft": optimize_output["current_draft"],
            "retry_count": optimize_output["retry_count"],
        }
        evaluate_output = workflow.node_evaluate(evaluate_state)
        self.assertEqual(evaluate_output["evaluation_score"], 0.98)
        self.assertTrue(evaluate_output["verified"])
        self.assertFalse(evaluate_output["stagnated"])
        self.assertEqual(len(evaluate_output["history"]), 1)
        self.assertEqual(len(evaluate_output["feedback_history"]), 1)

    def test_13_langgraph_conditional_edge_routing(self):
        """Verify conditional router directs flow back to node_optimize or END based on score, retries, and stagnation."""
        mock_repo = MockVectorStoreRepository()
        mock_embed = MockEmbeddingGenerator()
        mock_llm = MockLLMClient([])

        workflow = LangGraphRAGWorkflow(
            vector_store=mock_repo,
            llm_client=mock_llm,
            embedding_generator=mock_embed,
            max_retries=3,
            trust_threshold=0.90,
            stagnation_threshold=0.05,
        )

        # Case 1: Low score on iteration 1 -> retry (route to node_optimize)
        state_retry_1: AgentState = {
            "evaluation_score": 0.45,
            "retry_count": 1,
            "history": [{"score": 0.45}],
        }
        self.assertEqual(workflow.should_continue(state_retry_1), "node_optimize")

        # Case 2: Low score on iteration 2 with healthy score improvement (delta = +0.15 >= 0.05) -> retry
        state_retry_2: AgentState = {
            "evaluation_score": 0.60,
            "retry_count": 2,
            "history": [{"score": 0.45}, {"score": 0.60}],
        }
        self.assertEqual(workflow.should_continue(state_retry_2), "node_optimize")

        # Case 3: Stagnation circuit breaker (delta = +0.02 < 0.05) -> early exit (route to END)
        state_stagnated: AgentState = {
            "evaluation_score": 0.47,
            "retry_count": 2,
            "history": [{"score": 0.45}, {"score": 0.47}],
            "stagnated": True,
        }
        self.assertEqual(workflow.should_continue(state_stagnated), END)

        # Case 4: Low score, max retries reached -> terminate (route to END)
        state_max_retries: AgentState = {
            "evaluation_score": 0.60,
            "retry_count": 3,
            "history": [{"score": 0.40}, {"score": 0.50}, {"score": 0.60}],
        }
        self.assertEqual(workflow.should_continue(state_max_retries), END)

        # Case 5: High score meeting trust threshold -> accept and terminate (route to END)
        state_accepted: AgentState = {
            "evaluation_score": 0.95,
            "retry_count": 1,
            "verified": True,
            "history": [{"score": 0.95}],
        }
        self.assertEqual(workflow.should_continue(state_accepted), END)

    def test_14_langgraph_compiled_workflow_e2e_retry_loop(self):
        """Verify full compiled StateGraph multi-agent loop with retry self-correction and acceptance."""
        mock_repo = MockVectorStoreRepository()
        mock_repo.upsert(
            chunks=[
                DocumentChunk(
                    chunk_id="c_e2e_1",
                    text="Ares-Nexus isolates cognitive AI from state machines.",
                    metadata={"filename": "spec.txt", "section_id": "sec_isolation", "timestamp": "2026-09-18"},
                )
            ],
            embeddings=[[0.1] * 128],
        )
        mock_embed = MockEmbeddingGenerator()
        mock_llm = MockLLMClient([
            # Draft 1 (hallucination)
            "Ares-Nexus directly executes transactions in state machines.",
            # Evaluator 1 (rejection)
            '{"score": 0.40, "feedback": "Contradicts specification: cognitive AI is isolated from state machines."}',
            # Draft 2 (self-corrected revision)
            "Ares-Nexus isolates cognitive AI from direct state machine execution [Source: spec.txt, Section: sec_isolation, Date: 2026-09-18].",
            # Evaluator 2 (acceptance)
            '{"score": 1.0, "feedback": "Fully verified and correctly cited."}',
        ])

        workflow = create_evaluator_optimizer_workflow(
            vector_store=mock_repo,
            llm_client=mock_llm,
            embedding_generator=mock_embed,
            max_retries=3,
            trust_threshold=0.90,
        )

        final_state = workflow.invoke("How does Ares-Nexus interface with state machines?")

        self.assertTrue(final_state["verified"])
        self.assertFalse(final_state["stagnated"])
        self.assertEqual(final_state["evaluation_score"], 1.0)
        self.assertEqual(final_state["retry_count"], 2)
        self.assertEqual(len(final_state["history"]), 2)
        self.assertEqual(len(final_state["drafts"]), 2)
        self.assertEqual(len(final_state["feedback_history"]), 2)
        self.assertIn("isolates cognitive AI", final_state["final_answer"])
        self.assertEqual(len(final_state["retrieved_context"]), 1)

    def test_15_langgraph_compiled_workflow_max_retries_exhausted(self):
        """Verify compiled StateGraph gracefully returns safety fallback when retries are exhausted."""
        mock_repo = MockVectorStoreRepository()
        mock_embed = MockEmbeddingGenerator()
        mock_llm = MockLLMClient([
            # Draft 1 + Rejection
            "Draft 1",
            '{"score": 0.30, "feedback": "Hallucination 1"}',
            # Draft 2 + Rejection (score improved by 0.10 >= 0.05, no stagnation)
            "Draft 2",
            '{"score": 0.40, "feedback": "Hallucination 2"}',
            # Draft 3 + Rejection (score improved by 0.10 >= 0.05, no stagnation)
            "Draft 3",
            '{"score": 0.50, "feedback": "Hallucination 3"}',
        ])

        workflow = create_evaluator_optimizer_workflow(
            vector_store=mock_repo,
            llm_client=mock_llm,
            embedding_generator=mock_embed,
            max_retries=3,
            trust_threshold=0.90,
        )

        final_state = workflow.invoke("Explain undefined protocol.")

        self.assertFalse(final_state["verified"])
        self.assertFalse(final_state["stagnated"])
        self.assertEqual(final_state["retry_count"], 3)
        self.assertEqual(final_state["final_answer"], SAFETY_FALLBACK_MESSAGE)
        self.assertEqual(len(final_state["history"]), 3)

    def test_16_langgraph_stagnation_circuit_breaker(self):
        """Verify stagnation circuit breaker halts execution when consecutive score delta < 0.05."""
        mock_repo = MockVectorStoreRepository()
        mock_embed = MockEmbeddingGenerator()
        mock_llm = MockLLMClient([
            # Draft 1 + Rejection (score 0.40)
            "Draft v1 with ungrounded extrapolations.",
            '{"score": 0.40, "feedback": "Extrapolations detected."}',
            # Draft 2 + Rejection (score 0.42 -> delta = 0.02 < 0.05 stagnation)
            "Draft v2 with minor syntax changes but still ungrounded.",
            '{"score": 0.42, "feedback": "Still contains ungrounded claims."}',
        ])

        workflow = create_evaluator_optimizer_workflow(
            vector_store=mock_repo,
            llm_client=mock_llm,
            embedding_generator=mock_embed,
            max_retries=3,
            trust_threshold=0.90,
            stagnation_threshold=0.05,
        )

        final_state = workflow.invoke("Explain speculative execution.")

        self.assertFalse(final_state["verified"])
        self.assertTrue(final_state["stagnated"])
        self.assertEqual(final_state["retry_count"], 2)
        self.assertEqual(final_state["final_answer"], STAGNATION_FALLBACK_MESSAGE)
        self.assertEqual(len(final_state["history"]), 2)

    def test_17_evaluator_pydantic_and_malformed_json_defensive_fallbacks(self):
        """Verify defensive regex and Pydantic parsing gracefully fall back without runtime errors."""
        # Unparseable gibberish
        res_malformed = extract_evaluator_json("CRITICAL SYSTEM ERROR: MODEL FELL OVER <<<NOT JSON>>>")
        self.assertEqual(res_malformed.score, 0.0)
        self.assertEqual(res_malformed.feedback, PARSING_ERROR_FEEDBACK)
        self.assertFalse(res_malformed.verified)

        # JSON with non-numeric score
        res_invalid_type = extract_evaluator_json('{"score": "invalid_number", "feedback": "Failed"}')
        self.assertEqual(res_invalid_type.score, 0.0)
        self.assertEqual(res_invalid_type.feedback, "Failed")

        # JSON with missing fields
        res_empty_dict = extract_evaluator_json('{}')
        self.assertEqual(res_empty_dict.score, 0.0)
        self.assertEqual(res_empty_dict.feedback, PARSING_ERROR_FEEDBACK)

    def test_18_optimizer_accumulative_contrastive_history_prompt(self):
        """Verify Optimizer constructs rich contrastive prompt referencing all previous failed attempts."""
        mock_llm = MockLLMClient([])
        controller = EvaluatorOptimizerController(
            llm_client=mock_llm,
            max_retries=3,
            trust_threshold=0.90,
        )

        history = [
            {
                "iteration": 1,
                "draft": "First draft claiming AI directly triggers state transactions.",
                "score": 0.35,
                "feedback": "Hallucination: AI is non-executive and cannot trigger state transactions.",
            },
            {
                "iteration": 2,
                "draft": "Second draft omitting citation metadata.",
                "score": 0.70,
                "feedback": "Missing metadata citations for section sec_134.",
            },
        ]

        chunks = [
            RetrievedContextChunk(
                chunk_id="c1",
                text="Non-Executive AI operates purely as an advisory system.",
                metadata={"filename": "ares.txt", "section_id": "sec_134", "timestamp": "2026"},
                distance=0.1,
                similarity=0.9,
            )
        ]

        prompt = controller.construct_optimizer_prompt(
            query="What is Non-Executive AI?",
            retrieved_chunks=chunks,
            history=history,
        )

        self.assertIn("HISTORICAL REVISION LOG & PREVIOUS FAILURES:", prompt)
        self.assertIn("ITERATION 1 (Score: 0.35 - REJECTED)", prompt)
        self.assertIn("First draft claiming AI directly triggers state transactions.", prompt)
        self.assertIn("ITERATION 2 (Score: 0.70 - REJECTED)", prompt)
        self.assertIn("Second draft omitting citation metadata.", prompt)
        self.assertIn("CONTRASTIVE LEARNING & REVISION DIRECTIVES:", prompt)
        self.assertIn("Do NOT repeat the previous errors", prompt)


if __name__ == "__main__":
    unittest.main()

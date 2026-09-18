"""
Test suite for Ares-Nexus Local RAG Pipeline.
"""
import unittest
import tempfile
import shutil
from pathlib import Path

from src.config import (
    DEFAULT_DOCUMENT_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    SAFETY_FALLBACK_MESSAGE,
    TRUST_THRESHOLD,
)
from src.ollama_client import OllamaService
from src.ingestion import DocumentChunker, IngestionPipeline
from src.inference import (
    InferencePipeline,
    NOT_FOUND_MESSAGE,
    extract_evaluator_json,
)


class TestAresNexusRAG(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ollama = OllamaService()
        cls.is_ollama_available = cls.ollama.is_healthy()
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="chroma_test_"))

    @classmethod
    def tearDownClass(cls):
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def _ensure_test_collection(self):
        """Helper to ensure test collection has baseline documents ingested."""
        ingestion = IngestionPipeline(
            chroma_dir=self.temp_dir,
            collection_name="test_ares_collection",
            ollama_service=self.ollama
        )
        if ingestion.collection.count() == 0:
            ingestion.ingest_file(DEFAULT_DOCUMENT_PATH, reset_collection=False)

    def test_01_knowledge_base_file_exists_and_content(self):
        """Verify data/ares_nexus_pattern.txt exists and has expected paragraphs."""
        self.assertTrue(DEFAULT_DOCUMENT_PATH.exists(), f"File {DEFAULT_DOCUMENT_PATH} does not exist.")
        content = DEFAULT_DOCUMENT_PATH.read_text(encoding="utf-8")
        self.assertIn("Ares-Nexus", content)
        self.assertIn("Non-Executive AI", content)
        self.assertIn("Decision Gate", content)
        paragraphs = [p for p in content.split("\n\n") if p.strip()]
        self.assertGreaterEqual(len(paragraphs), 4, "Document must have at least 3-4 paragraphs.")

    def test_02_chunking_strategy(self):
        """Verify paragraph/semantic chunking produces metadata and IDs."""
        chunks = DocumentChunker.chunk_document(DEFAULT_DOCUMENT_PATH)
        self.assertGreaterEqual(len(chunks), 3)
        for chunk in chunks:
            self.assertIn("chunk_id", chunk)
            self.assertIn("text", chunk)
            self.assertIn("metadata", chunk)
            meta = chunk["metadata"]
            self.assertEqual(meta["filename"], "ares_nexus_pattern.txt")
            self.assertTrue("section_id" in meta)
            self.assertTrue("timestamp" in meta)

    def test_03_ollama_connectivity_and_embeddings(self):
        """Verify Ollama service produces valid embedding vectors."""
        if not self.is_ollama_available:
            self.skipTest("Ollama service not reachable on 127.0.0.1:11434")

        test_text = "Ares-Nexus Observe Reason Recommend loop"
        emb = self.ollama.get_embedding(test_text, model=EMBEDDING_MODEL)
        self.assertIsInstance(emb, list)
        self.assertGreater(len(emb), 100, "Embedding vector length should be > 100")

    def test_04_ingestion_pipeline(self):
        """Verify ingestion into ChromaDB test collection."""
        if not self.is_ollama_available:
            self.skipTest("Ollama service not reachable on 127.0.0.1:11434")

        ingestion = IngestionPipeline(
            chroma_dir=self.temp_dir,
            collection_name="test_ares_collection",
            ollama_service=self.ollama
        )
        result = ingestion.ingest_file(DEFAULT_DOCUMENT_PATH, reset_collection=True)
        self.assertEqual(result["status"], "success")
        self.assertGreaterEqual(result["chunks_ingested"], 3)
        self.assertGreaterEqual(result["total_collection_count"], 3)

    def test_05_inference_retrieval_and_prompt(self):
        """Verify Top-K retrieval and prompt formatting."""
        if not self.is_ollama_available:
            self.skipTest("Ollama service not reachable on 127.0.0.1:11434")

        self._ensure_test_collection()
        inference = InferencePipeline(
            chroma_dir=self.temp_dir,
            collection_name="test_ares_collection",
            ollama_service=self.ollama,
            top_k=3
        )
        chunks = inference.retrieve("What are Decision Gates and how do they work?")
        self.assertLessEqual(len(chunks), 3)
        self.assertGreater(len(chunks), 0)
        
        # Verify prompt construction
        prompt = inference.construct_prompt("What are Decision Gates?", chunks)
        self.assertIn("CONTEXT:", prompt)
        self.assertIn("USER QUERY:", prompt)
        self.assertIn("ares_nexus_pattern.txt", prompt)

    def test_06_end_to_end_rag_query(self):
        """Verify end-to-end question answering with Ollama."""
        if not self.is_ollama_available:
            self.skipTest("Ollama service not reachable on 127.0.0.1:11434")

        self._ensure_test_collection()
        inference = InferencePipeline(
            chroma_dir=self.temp_dir,
            collection_name="test_ares_collection",
            ollama_service=self.ollama,
            top_k=3
        )
        result = inference.query("Explain the Non-Executive AI (NEAI) pattern.")
        self.assertTrue(result["confidence_passed"])
        self.assertTrue(result["verified"])
        self.assertGreaterEqual(result["score"], 0.90)
        self.assertGreater(len(result["answer"]), 20)
        self.assertIn("NEAI", result["answer"] + result["prompt"])

    def test_07_out_of_domain_query_fallback(self):
        """Verify out-of-domain queries return baseline not found message."""
        if not self.is_ollama_available:
            self.skipTest("Ollama service not reachable on 127.0.0.1:11434")

        self._ensure_test_collection()
        inference = InferencePipeline(
            chroma_dir=self.temp_dir,
            collection_name="test_ares_collection",
            ollama_service=self.ollama,
            top_k=3
        )
        result = inference.query("How do I bake a chocolate cake with chocolate chips?")
        self.assertIn(NOT_FOUND_MESSAGE, result["answer"])

    def test_08_evaluator_json_extraction(self):
        """Verify robust extraction of Evaluator JSON under various formats."""
        # 1. Clean JSON
        raw1 = '{"score": 0.95, "feedback": "Accurate and well grounded."}'
        res1 = extract_evaluator_json(raw1)
        self.assertEqual(res1["score"], 0.95)
        self.assertIn("Accurate", res1["feedback"])

        # 2. Markdown fenced JSON
        raw2 = "```json\n{\n  \"score\": 0.85,\n  \"feedback\": \"Missing section cite.\"\n}\n```"
        res2 = extract_evaluator_json(raw2)
        self.assertEqual(res2["score"], 0.85)
        self.assertEqual(res2["feedback"], "Missing section cite.")

        # 3. Conversational preamble + JSON
        raw3 = "Here is my evaluation:\n```\n{\"score\": 1.0, \"feedback\": \"Fully grounded.\"}\n```\nHope this helps."
        res3 = extract_evaluator_json(raw3)
        self.assertEqual(res3["score"], 1.0)
        self.assertEqual(res3["feedback"], "Fully grounded.")

        # 4. Regex fallback when malformed
        raw4 = 'Score: 0.60\nFeedback: Contradicts section 3.'
        res4 = extract_evaluator_json(raw4)
        self.assertEqual(res4["score"], 0.60)
        self.assertTrue(len(res4["feedback"]) > 0)

        # 5. Empty string fallback
        res5 = extract_evaluator_json("")
        self.assertEqual(res5["score"], 0.0)

    def test_09_evaluator_optimizer_retry_loop_mock(self):
        """Verify closed-loop retry logic: fail on iteration 1, pass on iteration 2."""
        class MockOllamaService:
            def __init__(self):
                self.calls = []

            def is_healthy(self):
                return True

            def get_embedding(self, text, model=None):
                return [0.1] * 768

            def generate(self, prompt, system=None, model=None, temperature=0.0, options=None, format=None):
                self.calls.append({"prompt": prompt, "system": system, "format": format})
                # Check whether caller is Optimizer or Evaluator
                if format == "json" or (system and "Evaluator" in system):
                    # Evaluator response: reject first draft, accept second
                    if len([c for c in self.calls if c.get("format") == "json"]) == 1:
                        return '{"score": 0.50, "feedback": "Hallucinated database name."}'
                    else:
                        return '{"score": 0.95, "feedback": "All claims verified against context."}'
                else:
                    # Optimizer response
                    if "CRITIQUE & FEEDBACK FROM EVALUATOR" in prompt:
                        return "Draft v2: Ares-Nexus uses ChromaDB properly cited [Source: ares_nexus_pattern.txt, Section: 1]."
                    else:
                        return "Draft v1: Ares-Nexus uses SQLite with magic tables."

        mock_ollama = MockOllamaService()
        pipeline = InferencePipeline(
            chroma_dir=self.temp_dir,
            collection_name="test_ares_collection",
            ollama_service=mock_ollama,
            max_retries=3,
            trust_threshold=0.90,
        )

        dummy_chunks = [{
            "id": "1",
            "text": "Ares-Nexus uses ChromaDB vector store.",
            "metadata": {"filename": "ares_nexus_pattern.txt", "section_id": "1", "timestamp": "2026-09-18"},
            "distance": 0.1,
            "similarity": 0.9,
        }]

        result = pipeline.execute_evaluator_optimizer_loop("What database does Ares-Nexus use?", dummy_chunks)
        self.assertTrue(result["verified"])
        self.assertEqual(result["iterations"], 2)
        self.assertEqual(result["score"], 0.95)
        self.assertIn("Draft v2", result["answer"])
        self.assertEqual(len(result["history"]), 2)

    def test_10_evaluator_optimizer_max_retries_fallback_mock(self):
        """Verify fallback when max retries are reached without passing trust threshold."""
        class MockRejectingOllamaService:
            def __init__(self):
                self.optimizer_count = 0

            def is_healthy(self):
                return True

            def get_embedding(self, text, model=None):
                return [0.1] * 768

            def generate(self, prompt, system=None, model=None, temperature=0.0, options=None, format=None):
                if format == "json" or (system and "Evaluator" in system):
                    return '{"score": 0.40, "feedback": "Completely hallucinated facts."}'
                else:
                    self.optimizer_count += 1
                    return f"Unverified Draft v{self.optimizer_count}"

        mock_ollama = MockRejectingOllamaService()
        pipeline = InferencePipeline(
            chroma_dir=self.temp_dir,
            collection_name="test_ares_collection",
            ollama_service=mock_ollama,
            max_retries=3,
            trust_threshold=0.90,
        )

        dummy_chunks = [{
            "id": "1",
            "text": "Ares-Nexus architecture principles.",
            "metadata": {"filename": "ares_nexus_pattern.txt", "section_id": "1", "timestamp": "2026-09-18"},
            "distance": 0.1,
            "similarity": 0.9,
        }]

        result = pipeline.execute_evaluator_optimizer_loop("Explain the architecture.", dummy_chunks)
        self.assertFalse(result["verified"])
        self.assertEqual(result["iterations"], 3)
        self.assertEqual(result["answer"], SAFETY_FALLBACK_MESSAGE)


if __name__ == "__main__":
    unittest.main()

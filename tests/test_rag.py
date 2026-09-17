"""
Test suite for Ares-Nexus Local RAG Pipeline.
"""
import unittest
import tempfile
import shutil
from pathlib import Path

from src.config import DEFAULT_DOCUMENT_PATH, COLLECTION_NAME, EMBEDDING_MODEL, LLM_MODEL
from src.ollama_client import OllamaService
from src.ingestion import DocumentChunker, IngestionPipeline
from src.inference import InferencePipeline, NOT_FOUND_MESSAGE


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

        inference = InferencePipeline(
            chroma_dir=self.temp_dir,
            collection_name="test_ares_collection",
            ollama_service=self.ollama,
            top_k=3
        )
        result = inference.query("Explain the Non-Executive AI (NEAI) pattern and Decision Gate.")
        self.assertTrue(result["confidence_passed"])
        self.assertGreater(len(result["answer"]), 20)
        self.assertIn("NEAI", result["answer"] + result["prompt"])

    def test_07_out_of_domain_query_fallback(self):
        """Verify out-of-domain queries return baseline not found message."""
        if not self.is_ollama_available:
            self.skipTest("Ollama service not reachable on 127.0.0.1:11434")

        inference = InferencePipeline(
            chroma_dir=self.temp_dir,
            collection_name="test_ares_collection",
            ollama_service=self.ollama,
            top_k=3
        )
        result = inference.query("How do I bake a chocolate cake with chocolate chips?")
        self.assertIn(NOT_FOUND_MESSAGE, result["answer"])


if __name__ == "__main__":
    unittest.main()

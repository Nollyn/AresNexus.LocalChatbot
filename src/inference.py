"""
Inference Pipeline for Ares-Nexus Local RAG Chatbot.
Performs semantic retrieval, structured prompt assembly with metadata, and LLM generation.
"""
import logging
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
    AI_ARCHITECT_SYSTEM_PROMPT,
)
from .ollama_client import OllamaService

logger = logging.getLogger(__name__)

NOT_FOUND_MESSAGE = "Information not found in the baseline document."


class InferencePipeline:
    """Manages semantic search against ChromaDB and cited generation via Ollama."""

    def __init__(
        self,
        chroma_dir: Optional[Path] = None,
        collection_name: str = COLLECTION_NAME,
        ollama_service: Optional[OllamaService] = None,
        embedding_model: str = EMBEDDING_MODEL,
        llm_model: str = LLM_MODEL,
        top_k: int = DEFAULT_TOP_K,
        distance_threshold: float = DISTANCE_THRESHOLD,
    ):
        self.chroma_dir = chroma_dir or CHROMA_DIR
        self.collection_name = collection_name
        self.ollama = ollama_service or OllamaService()
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.top_k = top_k
        self.distance_threshold = distance_threshold

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

    def construct_prompt(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Construct structured prompt with injected context chunks, metadata citations, and query.
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

        prompt = (
            f"CONTEXT:\n"
            f"{context_section}\n\n"
            f"USER QUERY:\n"
            f"{query}\n\n"
            f"INSTRUCTIONS:\n"
            f"- Strictly act as an AI Architect.\n"
            f"- Base your answer ONLY on the context provided above.\n"
            f"- Cite your sources directly in the answer using metadata tags [Source: <filename>, Section: <section_id>, Date: <date>].\n"
            f"- If the context does not contain sufficient facts to answer the question, or if no relevant context is present, respond ONLY with:\n"
            f"  \"{NOT_FOUND_MESSAGE}\""
        )
        return prompt

    def query(self, query_text: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute full RAG pipeline: retrieval, prompt formatting, LLM generation.
        """
        if not query_text or not query_text.strip():
            return {
                "query": query_text,
                "answer": "Query cannot be empty.",
                "retrieved_chunks": [],
                "prompt": "",
            }

        chunks = self.retrieve(query_text, top_k=top_k)

        # Confidence gate: check if best retrieved chunk is within acceptable threshold
        best_distance = chunks[0]["distance"] if chunks and "distance" in chunks[0] else 1.0
        if not chunks or (best_distance is not None and best_distance > self.distance_threshold):
            logger.info("Best distance (%s) exceeds threshold (%s). Returning baseline not found.",
                        best_distance, self.distance_threshold)
            return {
                "query": query_text,
                "answer": NOT_FOUND_MESSAGE,
                "retrieved_chunks": chunks,
                "confidence_passed": False,
                "prompt": self.construct_prompt(query_text, chunks),
            }

        full_prompt = self.construct_prompt(query_text, chunks)
        
        try:
            raw_answer = self.ollama.generate(
                prompt=full_prompt,
                system=AI_ARCHITECT_SYSTEM_PROMPT,
                model=self.llm_model,
                temperature=0.0
            )
            answer = raw_answer.strip()
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            answer = f"Error generating answer from LLM: {e}"

        return {
            "query": query_text,
            "answer": answer,
            "retrieved_chunks": chunks,
            "confidence_passed": True,
            "prompt": full_prompt,
        }

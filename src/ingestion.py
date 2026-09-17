"""
Ingestion Pipeline for Ares-Nexus Knowledge Base.
Performs semantic/paragraph-based chunking and indexes vectors into ChromaDB.
"""
import os
import re
import datetime
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from .config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_DOCUMENT_PATH,
    EMBEDDING_MODEL,
)
from .ollama_client import OllamaService

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Paragraph and semantic chunker that preserves sentence and section integrity."""

    @staticmethod
    def chunk_document(file_path: Path) -> List[Dict[str, Any]]:
        if not file_path.exists():
            raise FileNotFoundError(f"Document file not found at: {file_path}")

        filename = file_path.name
        file_mtime = datetime.datetime.fromtimestamp(
            file_path.stat().st_mtime, tz=datetime.timezone.utc
        ).isoformat()

        content = file_path.read_text(encoding="utf-8")
        
        # Split document by double newlines or section headings
        raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
        
        chunks: List[Dict[str, Any]] = []
        doc_header_info = ""

        # Check if first paragraph contains document metadata / title
        if raw_paragraphs and ("#" in raw_paragraphs[0] or "Specification" in raw_paragraphs[0]):
            doc_header_info = raw_paragraphs[0]
            start_idx = 1
        else:
            start_idx = 0

        for idx, paragraph in enumerate(raw_paragraphs[start_idx:], start=start_idx):
            # Extract section title/id if present
            lines = paragraph.split("\n")
            first_line = lines[0].strip()
            
            # Identify section header like "1. Overview..." or "# Section..."
            section_match = re.match(r"^(\d+\.|\#+)\s*([^:\n\.]+)", first_line)
            if section_match:
                section_id = f"sec_{idx}_{section_match.group(2).strip().lower().replace(' ', '_')}"
            else:
                section_id = f"sec_{idx}_paragraph"

            # Create clean chunk text
            chunk_text = paragraph.strip()
            if len(chunk_text) < 30:
                continue

            chunk_id = hashlib.sha256(f"{filename}_{section_id}_{chunk_text}".encode("utf-8")).hexdigest()[:16]

            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "filename": filename,
                    "timestamp": file_mtime,
                    "section_id": section_id,
                    "chunk_index": idx,
                    "doc_header": doc_header_info.replace("\n", " | ") if doc_header_info else "Ares-Nexus Spec"
                }
            })

        logger.info("Generated %d semantic chunks from %s", len(chunks), filename)
        return chunks


class IngestionPipeline:
    """Manages document chunking, embedding generation, and ChromaDB vector indexing."""

    def __init__(
        self,
        chroma_dir: Optional[Path] = None,
        collection_name: str = COLLECTION_NAME,
        ollama_service: Optional[OllamaService] = None,
    ):
        self.chroma_dir = chroma_dir or CHROMA_DIR
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.ollama = ollama_service or OllamaService()

        # Initialize persistent Chroma client
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def ingest_file(self, file_path: Optional[Path] = None, reset_collection: bool = False) -> Dict[str, Any]:
        """
        Ingest a text document into the vector store.
        """
        target_file = file_path or DEFAULT_DOCUMENT_PATH
        logger.info("Starting ingestion of: %s", target_file)

        if reset_collection:
            logger.info("Resetting collection '%s'...", self.collection_name)
            try:
                self.chroma_client.delete_collection(name=self.collection_name)
            except Exception:
                pass
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

        chunks = DocumentChunker.chunk_document(target_file)
        if not chunks:
            raise ValueError(f"No valid chunks extracted from {target_file}")

        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            chunk_text = chunk["text"]
            chunk_meta = chunk["metadata"]

            logger.info("Computing embedding for chunk [%s]...", chunk_meta["section_id"])
            vector = self.ollama.get_embedding(chunk_text, model=EMBEDDING_MODEL)

            ids.append(chunk_id)
            documents.append(chunk_text)
            metadatas.append(chunk_meta)
            embeddings.append(vector)

        # Upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

        total_indexed = self.collection.count()
        logger.info("Ingestion completed. Total documents in collection: %d", total_indexed)
        return {
            "status": "success",
            "file": str(target_file),
            "chunks_ingested": len(chunks),
            "total_collection_count": total_indexed,
        }

"""
Concrete chunking strategy implementing paragraph and semantic boundary segmentation.
"""
import re
import hashlib
import datetime
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.domain.interfaces import ChunkerStrategy
from src.domain.models import DocumentChunk

logger = logging.getLogger(__name__)


class ParagraphChunkerStrategy(ChunkerStrategy):
    """
    Splits text documents into coherent paragraph/section units while generating
    cryptographic SHA-256 IDs and rich metadata (section headers, timestamps, filenames).
    """

    def __init__(self, min_chunk_length: int = 30):
        """
        :param min_chunk_length: Minimum character length for a chunk to be preserved.
        """
        self.min_chunk_length = min_chunk_length

    def chunk_document(self, file_path: Path) -> List[DocumentChunk]:
        """
        Read and chunk a document from the local filesystem.

        :param file_path: Path to the target file.
        :return: List of structured DocumentChunk entities.
        :raises FileNotFoundError: If file_path does not exist.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found at: {file_path}")

        filename = file_path.name
        file_mtime = datetime.datetime.fromtimestamp(
            file_path.stat().st_mtime, tz=datetime.timezone.utc
        ).isoformat()
        content = file_path.read_text(encoding="utf-8")

        return self.chunk_text(
            text=content,
            filename=filename,
            timestamp=file_mtime,
        )

    def chunk_text(
        self,
        text: str,
        filename: str = "document.txt",
        timestamp: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """
        Segment raw text into structured DocumentChunk entities.

        :param text: Raw text to segment.
        :param filename: Filename for metadata tracking.
        :param timestamp: Document timestamp or current UTC time.
        :return: List of DocumentChunk entities.
        """
        if not text or not text.strip():
            return []

        doc_timestamp = timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat()
        raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        chunks: List[DocumentChunk] = []
        doc_header_info = ""

        # Check if first paragraph contains document title or specification header
        if raw_paragraphs and ("#" in raw_paragraphs[0] or "Specification" in raw_paragraphs[0]):
            doc_header_info = raw_paragraphs[0]
            start_index = 1
        else:
            start_index = 0

        for index, paragraph in enumerate(raw_paragraphs[start_index:], start=start_index):
            lines = paragraph.split("\n")
            first_line = lines[0].strip()

            # Identify section header pattern
            section_match = re.match(r"^(\d+\.|\#+)\s*([^:\n\.]+)", first_line)
            if section_match:
                section_title = section_match.group(2).strip().lower().replace(" ", "_")
                section_id = f"sec_{index}_{section_title}"
            else:
                section_id = f"sec_{index}_paragraph"

            chunk_text = paragraph.strip()
            if len(chunk_text) < self.min_chunk_length:
                continue

            # Deterministic SHA-256 chunk ID
            hash_input = f"{filename}_{section_id}_{chunk_text}".encode("utf-8")
            chunk_id = hashlib.sha256(hash_input).hexdigest()[:16]

            metadata: Dict[str, Any] = {
                "filename": filename,
                "timestamp": doc_timestamp,
                "section_id": section_id,
                "chunk_index": index,
                "doc_header": doc_header_info.replace("\n", " | ") if doc_header_info else "Ares-Nexus Spec",
            }

            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                metadata=metadata,
            ))

        logger.info("Extracted %d chunks for document '%s'", len(chunks), filename)
        return chunks

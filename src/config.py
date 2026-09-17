"""
Configuration constants for Ares-Nexus Local RAG Pipeline.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
DEFAULT_DOCUMENT_PATH = DATA_DIR / "ares_nexus_pattern.txt"

# ChromaDB Settings
COLLECTION_NAME = "ares_nexus_collection"

# Ollama Settings
OLLAMA_HOST = "http://127.0.0.1:11434"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"

# Retrieval Settings
DEFAULT_TOP_K = 3
# Cosine distance threshold in ChromaDB (cosine distance: 0 = identical, >1.0 = distant)
DISTANCE_THRESHOLD = 0.85

# System Prompt Template
AI_ARCHITECT_SYSTEM_PROMPT = (
    "You are an expert AI Architect. Your duty is to provide accurate, rigorous, and "
    "well-structured architectural and technical guidance based strictly on the provided context.\n\n"
    "STRICT GUIDELINES:\n"
    "1. Base your response ONLY on the provided context. Do NOT extrapolate or assume external knowledge.\n"
    "2. Explicitly cite your sources using the metadata provided for each context chunk (e.g. [Source: <filename>, Section: <section_id>, Date: <timestamp>]).\n"
    "3. If the context does not contain sufficient facts to answer the user query, or if the question is unrelated to the context, you MUST explicitly output: 'Information not found in the baseline document.'\n"
    "4. Maintain professional, clear, and unambiguous technical language."
)

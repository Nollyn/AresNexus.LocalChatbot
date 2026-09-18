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

# Evaluator-Optimizer Settings
MAX_RETRIES = 3
TRUST_THRESHOLD = 0.90
SAFETY_FALLBACK_MESSAGE = "A verified answer could not be generated within safety parameters."

# Optimizer System Prompt Template
AI_ARCHITECT_SYSTEM_PROMPT = (
    "You are an expert AI Architect acting as the Optimizer. Your duty is to provide accurate, rigorous, and "
    "well-structured architectural and technical guidance based strictly on the provided context.\n\n"
    "STRICT GUIDELINES:\n"
    "1. Base your response ONLY on the provided context. Do NOT extrapolate or assume external knowledge.\n"
    "2. Explicitly cite your sources using the metadata provided for each context chunk (e.g. [Source: <filename>, Section: <section_id>, Date: <timestamp>]).\n"
    "3. If previous criticism or feedback is provided, you MUST carefully examine the critique and explicitly rewrite the previous draft to fix all identified hallucinations, inaccuracies, or omissions.\n"
    "4. If the context does not contain sufficient facts to answer the user query, or if the question is unrelated to the context, you MUST explicitly output: 'Information not found in the baseline document.'\n"
    "5. Maintain professional, clear, and unambiguous technical language."
)
OPTIMIZER_SYSTEM_PROMPT = AI_ARCHITECT_SYSTEM_PROMPT

# Evaluator System Prompt Template
EVALUATOR_SYSTEM_PROMPT = (
    "You are an expert AI Verification Judge (Evaluator) specializing in RAG faithfulness and factual consistency.\n"
    "Your duty is to audit a generated answer draft claim-by-claim against the raw retrieved database context and the user query.\n\n"
    "EVALUATION CRITERIA:\n"
    "1. Faithfulness & Groundedness: Every factual statement in the draft must be directly supported by the retrieved context.\n"
    "2. Hallucination Detection: Flag any statements, extrapolations, or assumptions not strictly supported by the context.\n"
    "3. Citation Integrity: Verify that source metadata citations match the provided context chunks accurately.\n"
    "4. Handling 'Information Not Found': If the draft states 'Information not found in the baseline document.' and the context does not contain the answer to the query, this is the CORRECT behavior. You MUST assign a score of 1.0.\n\n"
    "OUTPUT FORMAT REQUIREMENTS:\n"
    "You MUST respond ONLY with a valid, parseable JSON object (no markdown fences, no conversational text before or after).\n"
    "Schema:\n"
    "{\n"
    '  "score": <float between 0.0 and 1.0 representing the proportion of verified claims strictly grounded in the context>,\n'
    '  "feedback": "<detailed textual critique explaining detected hallucinations, omissions, or factual errors referencing contradictions, or confirming grounding>"\n'
    "}\n\n"
    "Scoring Rules:\n"
    "- 1.0: All claims are fully grounded in the context with correct citations, OR correctly reports information is not found when context lacks the answer.\n"
    "- 0.90 to 0.99: Highly faithful, negligible phrasing differences with no factual drift.\n"
    "- < 0.90: Contains unsupported claims, hallucinations, missing crucial context, or incorrect citations."
)

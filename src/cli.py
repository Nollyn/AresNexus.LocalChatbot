"""
Command-Line Interface (CLI) for Ares-Nexus Local RAG Chatbot.
Enforces Clean Architecture, SOLID principles, and Dependency Injection.
"""
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

from src.config import DEFAULT_DOCUMENT_PATH, COLLECTION_NAME, EMBEDDING_MODEL, LLM_MODEL
from src.infrastructure.vector_store import VectorStoreFactory
from src.infrastructure.llm import LLMClientFactory, EmbeddingGeneratorFactory
from src.infrastructure.chunking import ParagraphChunkerStrategy
from src.application.ingestion_service import IngestionService
from src.application.inference_service import InferenceService

logger = logging.getLogger("ares_nexus")


def print_banner() -> None:
    """Print application startup banner."""
    print("=" * 70)
    print("   ARES-NEXUS LOCAL RAG CHATBOT (Ollama + ChromaDB)")
    print("=" * 70)


def create_ingestion_service(
    collection_name: str = COLLECTION_NAME,
) -> IngestionService:
    """
    Composition root factory helper for IngestionService.
    """
    vector_store = VectorStoreFactory.create_vector_store(
        provider="chroma",
        collection_name=collection_name,
    )
    embedding_generator = EmbeddingGeneratorFactory.create_embedding_generator(
        provider="ollama",
        model_name=EMBEDDING_MODEL,
    )
    chunker_strategy = ParagraphChunkerStrategy()
    return IngestionService(
        vector_store=vector_store,
        embedding_generator=embedding_generator,
        chunker_strategy=chunker_strategy,
    )


def create_inference_service(
    collection_name: str = COLLECTION_NAME,
) -> InferenceService:
    """
    Composition root factory helper for InferenceService.
    """
    vector_store = VectorStoreFactory.create_vector_store(
        provider="chroma",
        collection_name=collection_name,
    )
    llm_client = LLMClientFactory.create_llm_client(
        provider="ollama",
        model_name=LLM_MODEL,
    )
    embedding_generator = EmbeddingGeneratorFactory.create_embedding_generator(
        provider="ollama",
        model_name=EMBEDDING_MODEL,
    )
    return InferenceService(
        vector_store=vector_store,
        llm_client=llm_client,
        embedding_generator=embedding_generator,
    )


def check_health(llm_client) -> bool:
    """Validate connectivity with Ollama server."""
    if not llm_client.is_healthy():
        print("[!] Error: Ollama server is not accessible at http://127.0.0.1:11434.")
        print("    Please ensure Ollama is running and required models are pulled.")
        return False
    return True


def run_ingestion(file_path: Optional[str] = None, reset: bool = False) -> bool:
    """Run document ingestion lifecycle."""
    path = Path(file_path) if file_path else DEFAULT_DOCUMENT_PATH
    print(f"\n[*] Starting ingestion pipeline for: {path}")

    llm_client = LLMClientFactory.create_llm_client(provider="ollama")
    if not check_health(llm_client):
        return False

    try:
        service = create_ingestion_service()
        result = service.ingest_file(file_path=path, reset_collection=reset)
        print("[+] Ingestion Succeeded!")
        print(f"    - File: {result['file']}")
        print(f"    - Chunks Indexed: {result['chunks_ingested']}")
        print(f"    - Total Documents in Vector Store: {result['total_collection_count']}\n")
        return True
    except Exception as err:
        print(f"[!] Ingestion failed with error: {err}")
        logger.exception("Ingestion failed")
        return False


def run_single_query(query_text: str, show_context: bool = False) -> None:
    """Execute a single query against the RAG inference pipeline."""
    llm_client = LLMClientFactory.create_llm_client(provider="ollama")
    if not check_health(llm_client):
        return

    try:
        service = create_inference_service()
        print(f"\n[?] Query: {query_text}")
        print("[*] Retrieving relevant knowledge and generating response...\n")

        result = service.query(query_text)

        if show_context and result.get("retrieved_chunks"):
            print("--- RETRIEVED CHUNKS ---")
            for idx, chunk in enumerate(result["retrieved_chunks"], 1):
                meta = chunk.get("metadata", {})
                print(
                    f"  [{idx}] Source: {meta.get('filename')} | "
                    f"Section: {meta.get('section_id')} | "
                    f"Sim: {chunk.get('similarity')}"
                )
            print("-" * 24 + "\n")

        status_str = "VERIFIED" if result.get("verified") else "UNVERIFIED"
        score_val = result.get("score", 0.0)
        iters = result.get("iterations", 1)
        print(f"--- AI ARCHITECT ANSWER [{status_str} | Score: {score_val:.2f} | Iterations: {iters}] ---")
        print(result["answer"])
        print("-" * 65 + "\n")

    except Exception as err:
        print(f"[!] Inference failed: {err}")
        logger.exception("Inference failed")


def run_interactive_loop() -> None:
    """Run interactive CLI chat loop."""
    print_banner()
    print("Type your questions below. Enter 'exit' or 'quit' to end session.")
    print("Enter ':context on' or ':context off' to toggle chunk inspection.\n")

    llm_client = LLMClientFactory.create_llm_client(provider="ollama")
    if not check_health(llm_client):
        return

    service = create_inference_service()
    show_context = False

    while True:
        try:
            user_input = input("\nAres-Nexus >> ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("[*] Exiting Ares-Nexus Chatbot. Goodbye!")
                break

            if user_input.lower() == ":context on":
                show_context = True
                print("[*] Context inspection enabled.")
                continue
            elif user_input.lower() == ":context off":
                show_context = False
                print("[*] Context inspection disabled.")
                continue

            result = service.query(user_input)

            if show_context and result.get("retrieved_chunks"):
                print("\n[Retrieved Context]")
                for idx, chunk in enumerate(result["retrieved_chunks"], 1):
                    meta = chunk.get("metadata", {})
                    print(
                        f"  Chunk {idx} -> File: {meta.get('filename')} | "
                        f"Section: {meta.get('section_id')} | "
                        f"Score: {chunk.get('similarity')}"
                    )

            status_str = "VERIFIED" if result.get("verified") else "UNVERIFIED"
            score_val = result.get("score", 0.0)
            iters = result.get("iterations", 1)
            print(
                f"\n[AI Architect Response | {status_str} "
                f"(Score: {score_val:.2f}, Iterations: {iters})]\n{result['answer']}"
            )

        except (KeyboardInterrupt, EOFError):
            print("\n[*] Exiting Ares-Nexus Chatbot.")
            break
        except Exception as err:
            print(f"[!] Error processing query: {err}")


def run_demo() -> None:
    """Run automated demonstration of ingestion, domain queries, and out-of-domain fallback."""
    print_banner()
    print("[*] Initiating Automated End-to-End RAG Demonstration...")

    # Step 1: Ingest
    print("\n--- STEP 1: Ingesting Knowledge Base ---")
    if not run_ingestion(reset=True):
        print("[!] Demo aborted due to ingestion failure.")
        return

    # Step 2: Queries
    test_queries = [
        "What are the main stages of the Ares-Nexus cognitive loop and what does each stage do?",
        "How do Decision Gates act as security proxies in the Ares-Nexus architecture?",
        "What fail-safe telemetry and governance mechanisms are implemented in Ares-Nexus?",
        "What is the standard recipe for baking sourdough bread?",  # Out-of-domain test
    ]

    print("\n--- STEP 2: Executing Test Queries ---")
    for q in test_queries:
        run_single_query(q, show_context=True)

    print("[+] Automated Demonstration Finished Successfully!")


def main() -> None:
    """Entrypoint parsing CLI arguments."""
    parser = argparse.ArgumentParser(description="Ares-Nexus Local RAG Chatbot")
    parser.add_argument("--ingest", action="store_true", help="Ingest knowledge base text into ChromaDB")
    parser.add_argument("--file", type=str, default=None, help="Custom document path to ingest")
    parser.add_argument("--reset", action="store_true", help="Reset ChromaDB collection before ingestion")
    parser.add_argument("--query", "-q", type=str, default=None, help="Single query to ask the chatbot")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run interactive CLI chat loop")
    parser.add_argument("--demo", action="store_true", help="Run automated end-to-end verification demo")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.ingest:
        run_ingestion(file_path=args.file, reset=args.reset)
    elif args.query:
        run_single_query(args.query, show_context=True)
    elif args.demo:
        run_demo()
    elif args.interactive or len(sys.argv) == 1:
        run_interactive_loop()

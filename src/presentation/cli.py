"""
Command-Line Interface (CLI) Presentation Layer for Ares-Nexus Local RAG Chatbot.
Invokes the compiled native LangGraph StateGraph workflow for multi-agent RAG verification.
"""
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from src.config import DEFAULT_DOCUMENT_PATH, COLLECTION_NAME, EMBEDDING_MODEL, LLM_MODEL
from src.domain.state import AgentState
from src.infrastructure.vector_store import VectorStoreFactory
from src.infrastructure.llm import LLMClientFactory, EmbeddingGeneratorFactory
from src.infrastructure.chunking import ParagraphChunkerStrategy
from src.application.ingestion_service import IngestionService
from src.application.workflow import LangGraphRAGWorkflow, create_evaluator_optimizer_workflow

logger = logging.getLogger("ares_nexus")


def print_banner() -> None:
    """Print application startup banner."""
    print("=" * 70)
    print("   ARES-NEXUS LOCAL RAG CHATBOT (LangGraph + Ollama + ChromaDB)")
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


def create_workflow(
    collection_name: str = COLLECTION_NAME,
) -> LangGraphRAGWorkflow:
    """
    Composition root factory helper for the compiled LangGraph StateGraph workflow.
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
    return create_evaluator_optimizer_workflow(
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
    """Execute a single query natively against the compiled LangGraph StateGraph workflow."""
    llm_client = LLMClientFactory.create_llm_client(provider="ollama")
    if not check_health(llm_client):
        return

    try:
        workflow = create_workflow()
        print(f"\n[?] Query: {query_text}")
        print("[*] Invoking LangGraph StateGraph workflow (Retrieval -> Optimization -> Evaluation)...\n")

        # Native LangGraph .invoke()
        final_state: AgentState = workflow.invoke(query_text)

        retrieved_chunks = final_state.get("retrieved_context", [])
        if show_context and retrieved_chunks:
            print("--- RETRIEVED CHUNKS ---")
            for idx, chunk in enumerate(retrieved_chunks, 1):
                if hasattr(chunk, "metadata"):
                    meta = chunk.metadata
                    sim = getattr(chunk, "similarity", 0.0)
                elif isinstance(chunk, dict):
                    meta = chunk.get("metadata", {})
                    sim = chunk.get("similarity", 0.0)
                else:
                    meta = {}
                    sim = 0.0
                print(
                    f"  [{idx}] Source: {meta.get('filename')} | "
                    f"Section: {meta.get('section_id')} | "
                    f"Sim: {sim}"
                )
            print("-" * 24 + "\n")

        verified = final_state.get("verified", False)
        status_str = "VERIFIED" if verified else "UNVERIFIED"
        score_val = final_state.get("evaluation_score", 0.0)
        history = final_state.get("history", [])
        iters = len(history) or final_state.get("retry_count", 1)
        answer = final_state.get("final_answer") or final_state.get("current_draft", "")

        print(f"--- AI ARCHITECT ANSWER [{status_str} | Score: {score_val:.2f} | Iterations: {iters}] ---")
        print(answer)
        print("-" * 65 + "\n")

    except Exception as err:
        print(f"[!] Inference failed: {err}")
        logger.exception("Inference failed")


def run_interactive_loop() -> None:
    """Run interactive CLI chat loop invoking LangGraph StateGraph workflow."""
    print_banner()
    print("Type your questions below. Enter 'exit' or 'quit' to end session.")
    print("Enter ':context on' or ':context off' to toggle chunk inspection.\n")

    llm_client = LLMClientFactory.create_llm_client(provider="ollama")
    if not check_health(llm_client):
        return

    workflow = create_workflow()
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

            final_state: AgentState = workflow.invoke(user_input)
            retrieved_chunks = final_state.get("retrieved_context", [])

            if show_context and retrieved_chunks:
                print("\n[Retrieved Context]")
                for idx, chunk in enumerate(retrieved_chunks, 1):
                    if hasattr(chunk, "metadata"):
                        meta = chunk.metadata
                        sim = getattr(chunk, "similarity", 0.0)
                    elif isinstance(chunk, dict):
                        meta = chunk.get("metadata", {})
                        sim = chunk.get("similarity", 0.0)
                    else:
                        meta = {}
                        sim = 0.0
                    print(
                        f"  Chunk {idx} -> File: {meta.get('filename')} | "
                        f"Section: {meta.get('section_id')} | "
                        f"Score: {sim}"
                    )

            verified = final_state.get("verified", False)
            status_str = "VERIFIED" if verified else "UNVERIFIED"
            score_val = final_state.get("evaluation_score", 0.0)
            history = final_state.get("history", [])
            iters = len(history) or final_state.get("retry_count", 1)
            answer = final_state.get("final_answer") or final_state.get("current_draft", "")

            print(
                f"\n[AI Architect Response | {status_str} "
                f"(Score: {score_val:.2f}, Iterations: {iters})]\n{answer}"
            )

        except (KeyboardInterrupt, EOFError):
            print("\n[*] Exiting Ares-Nexus Chatbot.")
            break
        except Exception as err:
            print(f"[!] Error processing query: {err}")


def run_demo() -> None:
    """Run automated demonstration of ingestion, domain queries, and out-of-domain fallback."""
    print_banner()
    print("[*] Initiating Automated End-to-End LangGraph RAG Demonstration...")

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

    print("\n--- STEP 2: Executing Test Queries with LangGraph StateGraph ---")
    for q in test_queries:
        run_single_query(q, show_context=True)

    print("[+] Automated Demonstration Finished Successfully!")


def main() -> None:
    """Entrypoint parsing CLI arguments."""
    parser = argparse.ArgumentParser(description="Ares-Nexus Local RAG Chatbot (LangGraph)")
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


if __name__ == "__main__":
    main()

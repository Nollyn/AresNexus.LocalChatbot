"""
Command-Line Interface (CLI) for Ares-Nexus Local RAG Chatbot.
"""
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

from .config import DEFAULT_DOCUMENT_PATH
from .ollama_client import OllamaService
from .ingestion import IngestionPipeline
from .inference import InferencePipeline, NOT_FOUND_MESSAGE

logger = logging.getLogger("ares_nexus")


def print_banner():
    print("=" * 70)
    print("   ARES-NEXUS LOCAL RAG CHATBOT (Ollama + ChromaDB)")
    print("=" * 70)


def run_ingestion(file_path: Optional[str] = None, reset: bool = False) -> bool:
    """Run document ingestion."""
    path = Path(file_path) if file_path else DEFAULT_DOCUMENT_PATH
    print(f"\n[*] Starting ingestion pipeline for: {path}")
    
    ollama_srv = OllamaService()
    if not ollama_srv.is_healthy():
        print("[!] Error: Ollama server is not accessible at http://127.0.0.1:11434.")
        print("    Please ensure Ollama is running and required models are pulled.")
        return False

    try:
        pipeline = IngestionPipeline(ollama_service=ollama_srv)
        result = pipeline.ingest_file(file_path=path, reset_collection=reset)
        print(f"[+] Ingestion Succeeded!")
        print(f"    - File: {result['file']}")
        print(f"    - Chunks Indexed: {result['chunks_ingested']}")
        print(f"    - Total Documents in Vector Store: {result['total_collection_count']}\n")
        return True
    except Exception as e:
        print(f"[!] Ingestion failed with error: {e}")
        logger.exception("Ingestion failed")
        return False


def run_single_query(query_text: str, show_context: bool = False):
    """Execute a single query against the RAG pipeline."""
    ollama_srv = OllamaService()
    if not ollama_srv.is_healthy():
        print("[!] Error: Ollama server is not accessible at http://127.0.0.1:11434.")
        return

    try:
        pipeline = InferencePipeline(ollama_service=ollama_srv)
        print(f"\n[?] Query: {query_text}")
        print("[*] Retrieving relevant knowledge and generating response...\n")
        
        result = pipeline.query(query_text)
        
        if show_context and result.get("retrieved_chunks"):
            print("--- RETRIEVED CHUNKS ---")
            for i, chunk in enumerate(result["retrieved_chunks"], 1):
                meta = chunk.get("metadata", {})
                print(f"  [{i}] Source: {meta.get('filename')} | Section: {meta.get('section_id')} | Sim: {chunk.get('similarity')}")
            print("-" * 24 + "\n")

        status_str = "VERIFIED" if result.get("verified") else "UNVERIFIED"
        score_val = result.get("score", 0.0)
        iters = result.get("iterations", 1)
        print(f"--- AI ARCHITECT ANSWER [{status_str} | Score: {score_val:.2f} | Iterations: {iters}] ---")
        print(result["answer"])
        print("-" * 65 + "\n")

    except Exception as e:
        print(f"[!] Inference failed: {e}")
        logger.exception("Inference failed")


def run_interactive_loop():
    """Run interactive CLI chat loop."""
    print_banner()
    print("Type your questions below. Enter 'exit' or 'quit' to end session.")
    print("Enter ':context on' or ':context off' to toggle chunk inspection.\n")

    ollama_srv = OllamaService()
    if not ollama_srv.is_healthy():
        print("[!] Error: Ollama is not accessible at http://127.0.0.1:11434. Exiting.")
        return

    pipeline = InferencePipeline(ollama_service=ollama_srv)
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

            result = pipeline.query(user_input)

            if show_context and result.get("retrieved_chunks"):
                print("\n[Retrieved Context]")
                for i, chunk in enumerate(result["retrieved_chunks"], 1):
                    meta = chunk.get("metadata", {})
                    print(f"  Chunk {i} -> File: {meta.get('filename')} | Section: {meta.get('section_id')} | Score: {chunk.get('similarity')}")

            status_str = "VERIFIED" if result.get("verified") else "UNVERIFIED"
            score_val = result.get("score", 0.0)
            iters = result.get("iterations", 1)
            print(f"\n[AI Architect Response | {status_str} (Score: {score_val:.2f}, Iterations: {iters})]\n{result['answer']}")

        except (KeyboardInterrupt, EOFError):
            print("\n[*] Exiting Ares-Nexus Chatbot.")
            break
        except Exception as e:
            print(f"[!] Error processing query: {e}")


def run_demo():
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


def main():
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

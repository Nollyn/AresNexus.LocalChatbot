"""
Command-Line Interface (CLI) entrypoint for Ares-Nexus Local RAG Chatbot.
Preserves backward compatibility while delegating to src.presentation.cli.
"""
from src.presentation.cli import (
    main,
    run_single_query,
    run_interactive_loop,
    run_demo,
    run_ingestion,
    create_ingestion_service,
    create_workflow,
    check_health,
    print_banner,
)

# Backward-compatibility alias
create_inference_service = create_workflow

if __name__ == "__main__":
    main()

# Ares-Nexus: Local GenAI Sandbox

<!-- Architectural Badges -->
![Architecture](https://shields.io/badge/Architecture-Clean%20Architecture%20%7C%20SOLID%20%7C%20DDD-darkgreen)
![Design-Patterns](https://shields.io/badge/Patterns-Evaluator--Optimizer%20%7C%20Strategy%20%7C%20Factory-blueviolet)
![AI-Engine](https://shields.io/badge/AI%20Engine-Ollama%20%7C%20Llama%203-blue)
![Vector-DB](https://shields.io/badge/Vector%20DB-ChromaDB%20%7C%20Local-orange)
![Security](https://shields.io/badge/Security-Strict%20Anti%2D%2DHallucination-darkred)

## Executive Summary
This repository serves as a production-grade local **Retrieval-Augmented Generation (RAG)** sandbox designed to evaluate the deployment and behavioral constraints of the **Ares-Nexus** architectural pattern within highly regulated environments.

The codebase strictly enforces **SOLID principles**, **Clean Architecture**, and the **Evaluator-Optimizer** Multi-Agent design pattern, establishing a closed-loop iterative audit mechanism to guarantee context grounding and verifiable inference.

---

## Clean Architecture & Enterprise Design Patterns

```text
src/
├── domain/                  # Domain Layer (Entities & Abstract Interfaces)
│   ├── models.py            # DocumentChunk, RetrievedContextChunk, EvaluationResult, InferenceResponse
│   └── interfaces.py        # VectorStoreRepository, LLMClient, EmbeddingGenerator, ChunkerStrategy
├── infrastructure/          # Infrastructure Layer (Concrete Adapters & DB Engines)
│   ├── chunking/            # ParagraphChunkerStrategy (Strategy Pattern)
│   ├── vector_store/        # ChromaVectorStoreRepository & VectorStoreFactory (Factory Pattern)
│   └── llm/                 # OllamaLLMClient, OllamaEmbeddingGenerator & LLMClientFactory
├── application/             # Application Layer (Use Cases & Orchestrators)
│   ├── ingestion_service.py # IngestionService (Single Responsibility orchestration)
│   ├── inference_service.py # InferenceService (Dependency Inversion consumer)
│   └── evaluator_optimizer.py # EvaluatorOptimizerController (Closed-loop multi-agent audit)
├── config.py                # Hyperparameters, settings, and system prompt templates
└── cli.py                   # Presentation Layer (CLI entrypoint with Composition Root)
```

### 1. SOLID Principles Implementation
* **Single Responsibility Principle (SRP):** Distinct classes for persistence (`ChromaVectorStoreRepository`), LLM generation (`OllamaLLMClient`), embeddings (`OllamaEmbeddingGenerator`), chunking (`ParagraphChunkerStrategy`), ingestion orchestration (`IngestionService`), and inference orchestration (`InferenceService`).
* **Open/Closed Principle (OCP) & Interface Segregation:** Abstract base classes (`VectorStoreRepository`, `LLMClient`, `EmbeddingGenerator`, `ChunkerStrategy`) defined in `src/domain/interfaces.py`. New backends (e.g., Amazon Bedrock, OpenSearch) can be plugged in without changing core business logic.
* **Dependency Inversion Principle (DIP):** High-level services (`InferenceService`, `IngestionService`, `EvaluatorOptimizerController`) depend strictly on domain interfaces via constructor injection (`__init__`).

### 2. Applied Design Patterns
* **Evaluator-Optimizer Multi-Agent Pattern:** Two-agent loop with Optimizer (draft generator) and Evaluator (verification judge), claim-by-claim context auditing, and trust threshold gating.
* **Strategy Pattern:** Semantic text chunking encapsulated in `ParagraphChunkerStrategy` adhering to `ChunkerStrategy`.
* **Factory Pattern:** Extensible factories `VectorStoreFactory`, `LLMClientFactory`, and `EmbeddingGeneratorFactory`.

---

## Evaluator-Optimizer Closed-Loop Pipeline

```text
[User Query] 
     │
     ▼
[Semantic Retrieval (VectorStoreRepository)]
     │
     ▼
[Optimizer Drafts v1] ◄────────────────────────────────────────┐
     │                                                         │
     ▼                                                         │
[Evaluator Audits JSON] ──(if score < 0.90)──► [Retry Loop with Corrective Critique]
     │
     │ (if score >= 0.90)
     ▼
[Final Verified Answer]
```

---

## Core SLA & Operational Matrix

| Dimension | Sandbox Target (Local) | Enterprise Production Mapping (AWS) | Verification Method |
| :--- | :--- | :--- | :--- |
| **Orchestration** | Clean Architecture Python Services | **Amazon Bedrock Agents (ReAct)** | Unit & E2E Traces |
| **Data Vectorization**| Ollama `nomic-embed-text` | **Amazon Titan Embeddings v2** | Vector pipeline execution |
| **Vector Storage** | Local ChromaDB Partition | **Amazon OpenSearch Serverless / Aurora** | Cosine Similarity (< 0.85) |
| **Security & Guardrails** | Evaluator-Optimizer with 0.90 Trust Threshold | **Amazon Bedrock Guardrails** | Claim-by-claim Grounding Audit |
| **Hallucination Rate**| 0.0% (Deterministic Fallback) | 0.0% via Guardrail Interception | RAGAS Framework Testing |

---

## Getting Started (Local Setup)

### Prerequisites
Ensure you have **Ollama** installed locally and the corresponding models downloaded:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### Installation
1. Initialize virtual environment and install dependencies:
```bash
python -m venv venv
.\venv\Scripts\activate  # On Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

2. Execute test suite:
```bash
python -m unittest discover tests
```

3. Run ingestion and single-turn query:
```bash
python main.py --ingest --reset
python main.py --query "How do Decision Gates act as security proxies in Ares-Nexus?"
```

# Ares-Nexus: Local GenAI Sandbox

<!-- Architectural Badges -->
![Architecture](https://shields.io/badge/Architecture-Clean%20Architecture%20%7C%20RAG-darkgreen)
![AI-Engine](https://shields.io/badge/AI%20Engine-Ollama%20%7C%20Llama%203-blue)
![Vector-DB](https://shields.io/badge/Vector%20DB-ChromaDB%20%7C%20Local-orange)
![Security](https://shields.io/badge/Security-Strict%20Anti%2D%2DHallucination-darkred)
![Compliance](https://shields.io/badge/Compliance-HIPAA%20%7C%20GDPR%20Ready-blueviolet)

## Executive Summary
This repository serves as a production-grade local **Retrieval-Augmented Generation (RAG)** sandbox designed to evaluate the deployment and behavioral constraints of the **Ares-Nexus** architectural pattern within highly regulated environments. 

The sandbox officially enforces the **Evaluator-Optimizer** Multi-Agent design pattern, establishing a closed-loop iterative audit mechanism to guarantee context grounding and verifiable inference. By leveraging an offline, zero-cost stack, this environment demonstrates deterministic context isolation, strict boundary enforcement, and precise citation tracking before scaling to enterprise AWS infrastructures (**Amazon Bedrock**, **Amazon OpenSearch Serverless**, and **AWS Lambda**).

---

## System Architecture & Data Lifecycle

The application enforces a rigid separation of concerns, splitting data processing into two decoupled pipelines to optimize compute costs and latency:

### 1. The Ingestion Pipeline (Asynchronous Background Layer)
*   **Data Lake Ingestion:** Source texts (clinical patterns and technical whitepapers) are securely pooled into the `data/` directory.
*   **Semantic Fragmenting:** Implements paragraph-driven chunking boundaries to protect logical sentence coherence, preventing the fragmentation of critical clinical instructions.
*   **Vectorization Engine:** Text chunks are processed locally via Ollama utilizing the `nomic-embed-text` multidimensional spatial model.
*   **Vector Indexing:** High-dimensional embeddings and structural metadata (source origins, structural identifiers, and generation timestamps) are committed directly to a localized **ChromaDB** partition.

### 2. The Inference Pipeline (Evaluator-Optimizer Closed Loop)

```text
[User Query] 
     │
     ▼
[Retrieval] 
     │
     ▼
[Optimizer Drafts v1] ◄────────────────────────────────────────┐
     │                                                         │
     ▼                                                         │
[Evaluator Judges JSON] ──(if < 0.90)──► [Retry Loop with Feedback]
     │
 (if >= 0.90)
     │
     ▼
[Final Verified Answer]
```

---

## Core SLA & Operational Matrix

| Dimension | Sandbox Target (Local) | Enterprise Production Mapping (AWS) | Verification Method |
| :--- | :--- | :--- | :--- |
| **Orchestration** | Python CLI Backend | **Amazon Bedrock Agents (ReAct)** | End-to-end trace logs |
| **Data Vectorization**| Ollama `nomic-embed-text` | **Amazon Titan Embeddings v2** | Vector pipeline execution |
| **Vector Storage** | Local ChromaDB Partition | **Amazon OpenSearch Serverless / Aurora** | KNN / Cosine Similarity |
| **Security & Guardrails** | Evaluator-Optimizer Dynamic Loop with a 0.90 Trust Threshold | **Amazon Bedrock Guardrails** | Automated Contextual Grounding Checks |
| **Hallucination Rate**| 0.0% (Deterministic Fallback) | 0.0% via Guardrail Interception | RAGAS Framework Testing |

---

## Strict Anti-Hallucination Guardrails
To satisfy the stringent safety mandates of clinical diagnostic workflows, the underlying inference pipeline rejects out-of-domain knowledge retrieval. 

*   **Context Isolation:** The language model is structurally prohibited from utilizing pre-trained weights if the semantic lookup fails to pass confidence thresholds.
*   **Deterministic Fallback:** Out-of-bounds inputs (e.g., asking general knowledge queries unrelated to the knowledge base) are actively caught by backend filters, triggering a hard validation failure: *"Information not found in the baseline document."*

---

## Getting Started (Local Setup)

### Prerequisites
Ensure you have **Ollama** installed locally and the corresponding models downloaded:
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### Installation
1. Initialize your isolated virtual environment and pull project requirements:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

2. Execute the end-to-end ingestion and chat inference pipeline:
```bash
python main.py --ingest --reset
python main.py --query "How do Decision Gates act as security proxies in Ares-Nexus?"
```

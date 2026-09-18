# ADR 0001: Local RAG Baseline Architecture

## Status
Approved

## Context
Evaluating the Ares-Nexus architectural pattern and research paper requires rigorous testing in clinical and regulated domains. Conducting early-stage experimentation and verification directly on cloud-native infrastructure introduces recurring compute/storage expenses and potential compliance risks related to data privacy and governance before production hardening.

To address these challenges, there is a clear need to evaluate the Ares-Nexus research paper without incurring cloud infrastructure costs or violating privacy data standards before scaling to AWS enterprise services (such as Amazon Bedrock, Amazon OpenSearch Serverless, and AWS Lambda).

## Decision
We have decided to implement a decoupled local Retrieval-Augmented Generation (RAG) baseline architecture. The architecture separates data ingestion from inference and consists of the following components:

1. **Local LLM Engine:** Ollama running open-weights models (`llama3` / `llama3.2`) locally for zero-egress generative inference.
2. **Spatial Embedding Model:** Ollama running `nomic-embed-text` to generate multidimensional vector embeddings locally.
3. **Vector Database & Metadata Store:** ChromaDB running in localized persistent client mode, backed by SQLite for metadata storage and indexing.
4. **Decoupled Ingestion Pipeline:** Paragraph-driven semantic chunking that ingests technical whitepapers and clinical pattern documentation while preserving contextual integrity.

## Consequences

### Positive
* **Zero Operational Cost:** Eliminates cloud API consumption charges and per-token vector/model hosting fees during early-stage prototyping and validation.
* **Absolute Data Containment & Privacy:** Guarantees complete data locality with zero external egress, satisfying strict privacy data standards and compliance prerequisites (e.g., HIPAA and GDPR ready).
* **Semantic Boundary Preservation:** Semantic-paragraph chunking preserves context boundaries and protects clinical instructional integrity.
* **Production-Ready Abstraction:** Decoupled interfaces allow seamless transition to enterprise AWS services (Amazon OpenSearch Serverless, Amazon Titan Embeddings v2, and Amazon Bedrock).

### Negative
* **Initialization Latency:** High initialization latency and cold-start overhead when loading local model weights into consumer hardware memory.
* **Local Compute Constraints:** Throughput and concurrent query scalability are constrained by local GPU/CPU resources and memory limits compared to managed cloud services.
* **Eventual Consistency Lag:** Local embedded vector persistence and indexing may introduce synchronization lags during rapid document re-indexing compared to cloud-native alternatives.

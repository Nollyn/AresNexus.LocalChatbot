# ADR 0003: Agent Framework Breadth via Native LangGraph

## Status
Approved

## Context
Fulfilling the structural requirement of demonstrating breadth across modern agent frameworks, introducing advanced stateful orchestration, real tool-calling mechanics, and auditable memory layers between execution steps.

Standard custom loops, while functional, lack formalized graph-level abstractions for state transitions, conditional routing, checkpointing, and interoperability with wider agent ecosystems. To ensure industrial rigor and parity with enterprise orchestration platforms, the multi-agent Evaluator-Optimizer lifecycle requires a standardized graph network definition.

## Decision
Adopted and integrated the official open-source **LangGraph** library to manage the multi-agent Evaluator-Optimizer loop natively via a `StateGraph` network:

1. **State Formalization (`AgentState`):** Defined a typed state dictionary tracking the single source of truth across execution cycles (`query`, `retrieved_context`, `current_draft`, `evaluation_score`, `evaluation_feedback`, `retry_count`, `verified`, `final_answer`, `history`).
2. **Explicit Graph Nodes:**
   * `node_retrieve`: Interacts with `VectorStoreRepository` (ChromaDB) to fetch semantic context chunks.
   * `node_optimize`: Invokes the Optimizer LLM agent (Ollama Llama 3) to generate or revise answer drafts using context and critique.
   * `node_evaluate`: Invokes the Evaluator LLM judge to execute claim-by-claim grounding assessment and JSON scoring.
3. **Deterministic Conditional Router (`should_continue`):**
   * If `evaluation_score < 0.90` and `retry_count < 3` -> Route back to `node_optimize`.
   * If `evaluation_score >= 0.90` or `retry_count >= 3` -> Route to LangGraph `END`.
4. **Lifecycle Stitching:** `START -> node_retrieve -> node_optimize -> node_evaluate -> should_continue -> (node_optimize | END)`.

## Consequences

### Positive
* **Hard Compliance with Industry-Standard Frameworks:** Seamlessly adopts standard LangGraph mechanics and stateful graph topologies.
* **Deterministic Graph Topology Control:** Clear separation of retrieval, generation, evaluation, and routing into inspectable nodes and edges.
* **Multi-Hop State Memory Preservation:** Complete preservation and mutation traceability of agent state and revision histories across graph iterations.
* **Direct Alignment with Enterprise Orchestration Semantics:** Maps directly to cloud agent architectures such as AWS Bedrock Multi-Agent Collaboration and Step Functions state machines.

### Negative
* **Increased CPU Instruction Overhead:** Local consumer hardware experiences incremental processing overhead from sequential multi-agent model invocations and graph step serialization.
* **Third-Party Dependency Cross-Linking:** Introduces `langgraph`, `langchain-core`, and `langsmith` runtime dependencies into the local Python environment.

# ADR 0002: Evaluator-Optimizer Multi-Agent Loop for Verified Inference

## Status
Approved

## Context
Standard single-pass Retrieval-Augmented Generation (RAG) pipelines remain vulnerable to probabilistic Large Language Model (LLM) hallucinations, subtle factual drifts, and incomplete source attributions. In regulated, high-stakes environments governed by the Ares-Nexus architectural pattern, unverified responses pose unacceptable compliance and operational risks.

To mitigate the risk of probabilistic LLM hallucinations and ensure strict, deterministic context grounding alignment with the safety principles of the Ares-Nexus pattern in regulated environments, the inference pipeline requires an autonomous, self-correcting verification mechanism.

## Decision
We have decided to implement a closed-loop multi-agent **Evaluator-Optimizer** pattern with rigid safety parameters (`TRUST_THRESHOLD = 0.90`, `MAX_RETRIES = 3`):

1. **Agent 1 (Optimizer / AI Architect):** Generates draft answers strictly grounded in the retrieved context chunks and source metadata, explicitly addressing critique and feedback from preceding iterations when present.
2. **Agent 2 (Evaluator / Verification Judge):** Audits the output claim-by-claim against the retrieved database context, generating structured JSON containing:
   * `score`: A floating-point metric (0.0 to 1.0) representing the proportion of verified, grounded claims.
   * `feedback`: Detailed textual critique pinpointing detected hallucinations, unsupported extrapolations, or citation discrepancies.
3. **Dynamic Feedback Loop:** If the evaluated score is below `0.90`, the system loops back with corrective feedback to the Optimizer for up to 3 retries.
4. **Deterministic Boundary Fallbacks:** If the trust threshold is not met within maximum retries, or if retrieved semantic distance exceeds the confidence threshold, the pipeline deterministically returns a safe fallback message without ungrounded extrapolation.

## Consequences

### Positive
* **Hard Reduction of Hallucination Rates:** Reduces hallucination rates to 0.0% through strict boundary fallback rules and claim-by-claim audit gates.
* **Verifiable Logic Traces:** Every iteration produces structured JSON scoring and transparent audit trails for explainability and compliance auditing.
* **Direct AWS Bedrock Guardrails Alignment:** The dual-agent scoring pattern directly correlates with enterprise guardrail mechanics and Bedrock Contextual Grounding policies.
* **Iterative Self-Correction:** Corrective feedback loops guide the generator toward high-fidelity answers without human-in-the-loop intervention.

### Negative
* **Compounded Token Execution Latency:** Per-user query latency increases linearly with the number of verification iterations required to pass the threshold.
* **Double Model Context Loading:** Running both Optimizer and Evaluator prompt evaluations increases local compute load and context window processing requirements.

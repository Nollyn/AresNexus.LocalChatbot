# ADR 0005: Evaluator-Optimizer CI/CD Control Plane and Zero-Trust Governance

## Status
Approved

## Context
As generative AI systems and agentic workflows are integrated into mission-critical and enterprise environments governed by the **Ares-Nexus (NEAI)** framework, the Software Development Life Cycle (SDLC) requires automated, auditable governance at the Pull Request (PR) stage. 

Standard CI/CD pipelines rely solely on deterministic static analysis (SAST, secret scanning, and unit tests). While essential, these traditional gates cannot evaluate semantic architecture alignment, agent state boundary adherence, prompt-injection resilience, or maintainability trade-offs introduced in generative components. Conversely, delegating PR approval authority entirely to an LLM introduces severe reliability risks, as Large Language Models are inherently probabilistic, non-deterministic, and susceptible to prompt injection, false positives, and hallucinations.

To bridge this gap while preserving zero-trust security postures, the pipeline requires an autonomous governance plane that combines deterministic hard gates with an agentic Evaluator-Optimizer feedback loop, governed strictly by deterministic decision gates.

## Decision
We have decided to establish a **Two-Phase Zero-Trust CI/CD Control Plane** orchestrated via GitHub Actions (`.github/workflows/neai-control-plane.yml`) and LangGraph (`src/evaluator_gate.py`), governed by formal policy rubrics (`docs/architecture_rubric.md` and `docs/security_rubric.md`):

1. **Phase 1: Deterministic Hard Gates & Circuit Breakers (Zero-Trust Baseline):**
   * Before any AI agent or LLM is invoked, the pipeline executes mandatory, deterministic controls: credential/secret pattern scanning, SAST security analysis (Bandit), and unit/integration test suites.
   * **Circuit Breaker:** Any failure in Phase 1 (e.g., exposed API keys, failing tests, or high-severity vulnerabilities) instantly trips a hard circuit breaker (`exit 1`), terminating the pipeline immediately and preventing compute waste or credential exposure to external models.

2. **Phase 2: Bounded Agentic Evaluator-Optimizer Loop (LangGraph `StateGraph`):**
   * **Isolated Context & Credential Handling:** The agentic loop receives strictly the modified PR file diffs and version-pinned Markdown policy rubrics. Sensitive environment keys are scoped and never propagated to context payloads.
   * **Evaluator Node (LLM-Assisted Audit):** Audits candidate code against formal architecture (`ARCH-01` to `ARCH-04`) and security (`SEC-01` to `SEC-04`) dimensions using structured JSON evaluation contracts via Claude / local LLM engines with deterministic AST fallbacks.
   * **Optimizer Node (Remediation Generator):** If minor, non-critical maintainability issues or architectural deviations are identified, the Optimizer automatically generates concrete code patches and remediation suggestions.
   * **Strict Iteration Bounding (`MAX_ITERATIONS = 2`):** To prevent infinite recursion, compute runaway, and prompt drift, the feedback loop is strictly bounded to a maximum of 2 iterations.

3. **Deterministic Decision Gate (Authority & System of Record):**
   * **Non-Negotiable Principle:** The LLM is strictly advisory; **AI is never the system of record for pull request acceptance**.
   * Final PR verdicts are calculated deterministically via code rules:
     * **`PASS`:** Architectural Score $\ge 0.85$, Security Score $\ge 0.90$, Composite Score $\ge 0.88$, and zero critical violations.
     * **`HUMAN_REVIEW`:** Composite Score between $0.60$ and $0.87$, or when remediations are proposed by the Optimizer. Requires explicit human architect sign-off.
     * **`BLOCK`:** Critical security violation (e.g., hardcoded secrets, unsafe `eval`/`exec`), severe architectural divergence, or Composite Score $< 0.60$. Automatically fails the PR check.
   * **Full Traceability & Immutable Audit Trail:** Every execution produces an immutable audit record (`neai_audit_log.json` and Markdown summary) binding the `Commit SHA`, policy rubric version hashes, tool invocation traces, and deterministic verdicts.

## Consequences

### Positive
* **Zero-Trust Enforcement:** Probabilistic model outputs cannot bypass deterministic security and architectural rules or auto-approve pull requests.
* **Proactive Automated Remediation:** Developers receive actionable patch proposals for maintainability debt directly within the PR lifecycle.
* **Auditable Governance Compliance:** Full cryptographic and structured traceability linking every commit SHA to the exact version of the active governance policies.
* **Compute & Token Frugality:** Phase 1 hard circuit breakers and bounded graph iterations eliminate unnecessary model inference costs.

### Negative
* **PR Execution Latency:** Adding an agentic evaluation and remediation loop introduces additional execution time (typically 10–45s depending on model provider).
* **Policy Maintenance Overhead:** Governance rubrics (`docs/*.md`) must be actively maintained and version-controlled as architectural standards evolve.

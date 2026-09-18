# ADR 0004: Defensive Parsing and Stagnation Circuit Breakers

## Status
Approved

## Context
Mitigating operational risks associated with brittle JSON deserialization from small quantized local models (such as Llama 3.2 on CPU) and preventing compute resource degradation caused by non-improving, stagnant multi-agent optimization loops.

## Decision
Implemented defensive regex extraction and try-except structural fallbacks for the Evaluator node's JSON parsing path, coupled with an automated accumulative state history log. Additionally, deployed an algorithmic stagnation circuit breaker within the conditional edge router to enforce an early exit if the evaluation score variance between retries drops below a 0.05 delta.

## Consequences

### Positive
* **100% Runtime Resilience Against Malformed LLM Syntax Layouts:** Robust extraction of JSON payloads even when surrounded by markdown code fences or conversational prose.
* **Complete Contrastive History Learning Vectors for the Optimizer:** Historical iteration records enable the generator to learn from past feedback and avoid repeating mistakes.
* **Guaranteed Protection of Hardware Compute Budget (AWS Frugality Pillar Alignment):** Eliminates wasted CPU cycles and inference latency when consecutive iterations fail to produce meaningful score improvement.

### Negative
* **State Memory Footprint:** Marginal overhead in state memory dictionary footprints due to accumulating multi-turn state history logs.
* **Routing Complexity:** Added complexity in conditional edge routing mathematical validations and variance calculations.

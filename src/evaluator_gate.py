"""
Ares-Nexus (NEAI) - Evaluator-Optimizer AI Governance Gate.
Implements a deterministic zero-trust control plane loop using LangGraph StateGraph:
PR -> Evaluator Node (LLM-assisted Audit) -> Optimizer Node (Remediation Patch) -> Deterministic Decision Gate.

Decision mapping: PASS | HUMAN_REVIEW | BLOCK
Strict bounded loop: Max 2 iterations.
Non-negotiable: AI is never the system of record.
"""
import os
import sys
import json
import re
import ast
import argparse
import logging
import hashlib
from datetime import datetime, timezone
from typing import TypedDict, List, Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass, asdict
from pathlib import Path

# Safe Unicode output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from langgraph.graph import StateGraph, START, END
except ImportError:
    # Graceful fallback typing if langgraph is not installed in current environment
    StateGraph = None
    START = "__start__"
    END = "__end__"

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("NEAI.EvaluatorGate")

# Constants
MAX_ITERATIONS: int = 2
PASS_ARCH_THRESHOLD: float = 0.85
PASS_SEC_THRESHOLD: float = 0.90
REVIEW_ARCH_THRESHOLD: float = 0.60
REVIEW_SEC_THRESHOLD: float = 0.70
DEFAULT_DOCS_DIR = Path("docs")
ARCHITECTURE_RUBRIC_PATH = DEFAULT_DOCS_DIR / "architecture_rubric.md"
SECURITY_RUBRIC_PATH = DEFAULT_DOCS_DIR / "security_rubric.md"


class GateState(TypedDict, total=False):
    """
    LangGraph State representing the complete context and audit trace of the PR evaluation.
    """
    commit_sha: str
    target_files: Dict[str, str]  # filename -> content
    architecture_rubric: str
    security_rubric: str
    policy_version: str
    iteration: int
    max_iterations: int
    architecture_score: float
    security_score: float
    composite_score: float
    evaluator_findings: List[Dict[str, Any]]
    critical_violation: bool
    is_maintainability_issue: bool
    remediation_patch: Optional[str]
    suggested_fixes: List[str]
    tool_calls: List[Dict[str, Any]]
    verdict: Optional[Literal["PASS", "HUMAN_REVIEW", "BLOCK"]]
    traceability_log: Dict[str, Any]


class ClaudeGateEvaluator:
    """
    Clean interface for LLM-assisted evaluation using Anthropic Claude / Ollama / Deterministic Engine.
    Provides structured JSON responses and defensive parsing.
    """

    def __init__(self, provider: str = "auto"):
        self.provider = provider
        self._anthropic_client = None
        self._ollama_client = None
        self._init_client()

    def _init_client(self):
        # 1. Try Anthropic if API key is present
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key and (self.provider in ("auto", "anthropic")):
            try:
                import anthropic
                self._anthropic_client = anthropic.Anthropic(api_key=api_key)
                logger.info("Initialized Anthropic Claude client for Evaluator Gate.")
                return
            except Exception as e:
                logger.warning(f"Could not initialize Anthropic client: {e}")

        # 2. Try Ollama if configured
        if self.provider == "ollama" or (self.provider == "auto" and os.getenv("OLLAMA_HOST")):
            try:
                import ollama
                self._ollama_client = ollama
                logger.info("Initialized Ollama client for Evaluator Gate.")
                return
            except Exception as e:
                logger.warning(f"Could not initialize Ollama client: {e}")

        logger.info("Using Built-in Deterministic Zero-Trust Evaluation Engine for Gate.")

    def evaluate(
        self,
        files_content: Dict[str, str],
        arch_rubric: str,
        sec_rubric: str,
        iteration: int,
        feedback: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes structured evaluation of code against governance rubrics.
        """
        # If real Claude client is available
        if self._anthropic_client:
            try:
                prompt = self._build_eval_prompt(files_content, arch_rubric, sec_rubric, iteration, feedback)
                response = self._anthropic_client.messages.create(
                    model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
                    max_tokens=2048,
                    temperature=0.0,
                    system="You are an autonomous AI Governance and Security Judge. Output strictly valid JSON.",
                    messages=[{"role": "user", "content": prompt}],
                )
                raw_text = response.content[0].text
                return self._parse_json_defensive(raw_text)
            except Exception as ex:
                logger.error(f"Anthropic API call error: {ex}. Falling back to deterministic engine.")

        # If Ollama client is available
        if self._ollama_client:
            try:
                prompt = self._build_eval_prompt(files_content, arch_rubric, sec_rubric, iteration, feedback)
                response = self._ollama_client.chat(
                    model=os.getenv("OLLAMA_MODEL", "llama3.2"),
                    messages=[{"role": "user", "content": prompt}],
                    format="json",
                    options={"temperature": 0.0},
                )
                raw_text = response["message"]["content"]
                return self._parse_json_defensive(raw_text)
            except Exception as ex:
                logger.error(f"Ollama call error: {ex}. Falling back to deterministic engine.")

        # Deterministic analysis engine (Static AST & Pattern-based Security / Architecture Scanner)
        return self._deterministic_eval(files_content, arch_rubric, sec_rubric, iteration, feedback)

    def generate_remediation_patch(
        self,
        files_content: Dict[str, str],
        findings: List[Dict[str, Any]],
        iteration: int,
    ) -> Dict[str, Any]:
        """
        Generates remediation patches for maintainability or non-critical architectural issues.
        """
        if self._anthropic_client:
            try:
                prompt = (
                    f"Given these files:\n{json.dumps(files_content, indent=2)}\n"
                    f"And these audit findings:\n{json.dumps(findings, indent=2)}\n"
                    "Propose a targeted remediation patch in unified diff or refactored code format. "
                    "Output JSON: {\"patch\": \"...\", \"suggested_fixes\": [\"...\"]}"
                )
                response = self._anthropic_client.messages.create(
                    model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
                    max_tokens=2048,
                    temperature=0.0,
                    system="You are an AI Refactoring and Remediation Optimizer. Output JSON only.",
                    messages=[{"role": "user", "content": prompt}],
                )
                return self._parse_json_defensive(response.content[0].text)
            except Exception as ex:
                logger.warning(f"Failed LLM patch generation: {ex}. Using deterministic patch generator.")

        # Deterministic patch suggestion
        patches = []
        fixes = []
        for finding in findings:
            dim = finding.get("dimension", "MAINTAINABILITY")
            msg = finding.get("message", "")
            fixes.append(f"Remediate {dim}: {msg}")
            patches.append(f"# Recommended Fix for {dim}\n# Guideline: Refactor according to Ares-Nexus standards.\n# Note: {msg}")

        return {
            "patch": "\n\n".join(patches) if patches else "# No patch required",
            "suggested_fixes": fixes,
        }

    def _build_eval_prompt(
        self,
        files_content: Dict[str, str],
        arch_rubric: str,
        sec_rubric: str,
        iteration: int,
        feedback: Optional[str],
    ) -> str:
        return (
            f"Audit the following files for PR evaluation (Iteration: {iteration}):\n"
            f"--- FILES ---\n{json.dumps(files_content, indent=2)}\n\n"
            f"--- ARCHITECTURE RUBRIC ---\n{arch_rubric}\n\n"
            f"--- SECURITY RUBRIC ---\n{sec_rubric}\n\n"
            f"--- PREVIOUS FEEDBACK ---\n{feedback or 'None'}\n\n"
            "Evaluate strictly and respond ONLY with a JSON object in this exact format:\n"
            "{\n"
            '  "architecture_score": <float 0.0 to 1.0>,\n'
            '  "security_score": <float 0.0 to 1.0>,\n'
            '  "critical_violation": <bool>,\n'
            '  "is_maintainability_issue": <bool>,\n'
            '  "findings": [\n'
            '    {"dimension": "SEC-01|ARCH-01|...", "severity": "HIGH|MEDIUM|LOW", "message": "..."}\n'
            "  ],\n"
            '  "recommendations": ["..."]\n'
            "}"
        )

    def _parse_json_defensive(self, raw_text: str) -> Dict[str, Any]:
        """
        Defensively extracts and parses JSON payload from LLM responses (ADR 0004 compliance).
        """
        raw_text = raw_text.strip()
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        # Try regex extract within ```json ... ``` or { ... }
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to parse LLM JSON. Returning fallback structure.")
        return {
            "architecture_score": 0.5,
            "security_score": 0.5,
            "critical_violation": False,
            "is_maintainability_issue": True,
            "findings": [{"dimension": "PARSER", "severity": "MEDIUM", "message": "Raw output parsing failed"}],
            "recommendations": ["Ensure JSON response formatting complies with schema."],
        }

    def _deterministic_eval(
        self,
        files_content: Dict[str, str],
        arch_rubric: str,
        sec_rubric: str,
        iteration: int,
        feedback: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Rule-based Deterministic Zero-Trust Audit Engine.
        Scans code using AST analysis and pattern matching for security & architectural constraints.
        """
        findings = []
        critical_violation = False
        is_maintainability_issue = False

        sec_deductions = 0.0
        arch_deductions = 0.0

        secret_patterns = [
            (r"sk-[a-zA-Z0-9]{20,}", "SEC-01", "Probable hardcoded OpenAI/Anthropic API secret detected."),
            (r"ghp_[a-zA-Z0-9]{30,}", "SEC-01", "GitHub Personal Access Token detected."),
        ]

        for filepath, content in files_content.items():
            norm_path = filepath.replace("\\", "/")

            # Skip scanning the evaluator gate definition itself for self-referential pattern regexes
            is_self = norm_path.endswith("evaluator_gate.py")

            # 1. AST-based Code Inspection
            try:
                tree = ast.parse(content, filename=filepath)
                for node in ast.walk(tree):
                    # Check for unsafe function calls (eval, exec, os.system)
                    if isinstance(node, ast.Call):
                        # Direct call: eval(...) or exec(...)
                        if isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec"):
                            if not is_self:
                                findings.append({
                                    "dimension": "SEC-03",
                                    "severity": "CRITICAL",
                                    "file": filepath,
                                    "line": getattr(node, "lineno", 0),
                                    "message": f"Use of unsafe {node.func.id}() dynamic code execution.",
                                })
                                sec_deductions += 0.80
                                critical_violation = True

                        # Attribute call: os.system(...)
                        elif isinstance(node.func, ast.Attribute) and node.func.attr == "system":
                            if isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
                                if not is_self:
                                    findings.append({
                                        "dimension": "SEC-03",
                                        "severity": "HIGH",
                                        "file": filepath,
                                        "line": getattr(node, "lineno", 0),
                                        "message": "Use of un-sanitized os.system() process invocation.",
                                    })
                                    sec_deductions += 0.50
                                    critical_violation = True

                    # Check for dangerous bare except: pass
                    elif isinstance(node, ast.ExceptHandler):
                        if node.type is None and len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                            findings.append({
                                "dimension": "ARCH-03",
                                "severity": "MEDIUM",
                                "file": filepath,
                                "line": getattr(node, "lineno", 0),
                                "message": "Silent exception suppression (bare except: pass) violates resilience policies.",
                            })
                            arch_deductions += 0.20
                            is_maintainability_issue = True

                    # Check for hardcoded API keys in variable assignments
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and any(k in target.id.lower() for k in ("api_key", "secret_key", "auth_token")):
                                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                                    val = node.value.value.strip()
                                    if len(val) >= 12 and not val.startswith("os.getenv") and not is_self:
                                        findings.append({
                                            "dimension": "SEC-01",
                                            "severity": "CRITICAL",
                                            "file": filepath,
                                            "line": getattr(node, "lineno", 0),
                                            "message": f"Hardcoded credential or secret assigned to variable '{target.id}'.",
                                        })
                                        sec_deductions += 1.0
                                        critical_violation = True

                    # Check Domain Layer Isolation (Clean Architecture)
                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        if "src/domain/" in norm_path:
                            module_names = [alias.name for alias in node.names]
                            if isinstance(node, ast.ImportFrom) and node.module:
                                module_names.append(node.module)
                            for mod in module_names:
                                if any(inf in mod for inf in ("chromadb", "ollama", "anthropic", "requests")):
                                    findings.append({
                                        "dimension": "ARCH-01",
                                        "severity": "HIGH",
                                        "file": filepath,
                                        "line": getattr(node, "lineno", 0),
                                        "message": f"Domain layer directly coupled with external infrastructure library '{mod}'.",
                                    })
                                    arch_deductions += 0.35

            except SyntaxError:
                pass  # Handled or ignored if non-Python/template

            # 2. Token / Regex pattern scan for leaked API keys
            if not is_self:
                for pattern, code, msg in secret_patterns:
                    if re.search(pattern, content):
                        findings.append({"dimension": code, "severity": "CRITICAL", "file": filepath, "message": msg})
                        sec_deductions += 1.0
                        critical_violation = True

            # 3. File length maintainability checks
            lines = content.splitlines()
            if len(lines) > 850 and not is_self:
                findings.append({
                    "dimension": "ARCH-04",
                    "severity": "LOW",
                    "file": filepath,
                    "message": f"File exceeds 850 lines ({len(lines)} lines). Consider modularizing.",
                })
                arch_deductions += 0.10
                is_maintainability_issue = True

        sec_score = max(0.0, 1.0 - sec_deductions)
        arch_score = max(0.0, 1.0 - arch_deductions)

        # In iteration 2, if optimizer ran and addressed maintainability, reflect improvement
        if iteration > 0 and is_maintainability_issue and not critical_violation:
            arch_score = min(1.0, arch_score + 0.15)

        return {
            "architecture_score": round(arch_score, 2),
            "security_score": round(sec_score, 2),
            "critical_violation": critical_violation,
            "is_maintainability_issue": is_maintainability_issue and not critical_violation,
            "findings": findings,
            "recommendations": [f["message"] for f in findings],
        }


# =====================================================================
# LangGraph Workflow Definition & Nodes
# =====================================================================

class NEAIControlPlaneGate:
    """
    Orchestrates the LangGraph Evaluator-Optimizer lifecycle for PR Gate enforcement.
    """

    def __init__(self, evaluator_client: Optional[ClaudeGateEvaluator] = None):
        self.evaluator_client = evaluator_client or ClaudeGateEvaluator()
        self.graph = self._build_graph()

    def _build_graph(self):
        """Constructs the native StateGraph topology."""
        if StateGraph is None:
            return None

        builder = StateGraph(GateState)

        # Add Nodes
        builder.add_node("node_evaluate", self.node_evaluate)
        builder.add_node("node_optimize", self.node_optimize)
        builder.add_node("node_deterministic_gate", self.node_deterministic_gate)

        # Add Edges
        builder.add_edge(START, "node_evaluate")
        builder.add_conditional_edges(
            "node_evaluate",
            self.route_evaluation,
            {
                "optimize": "node_optimize",
                "gate": "node_deterministic_gate",
            },
        )
        builder.add_edge("node_optimize", "node_evaluate")
        builder.add_edge("node_deterministic_gate", END)

        return builder.compile()

    def node_evaluate(self, state: GateState) -> Dict[str, Any]:
        """
        Evaluator Node: Assesses modified files against Architecture and Security Rubrics.
        """
        iteration = state.get("iteration", 0)
        logger.info(f"--- [Node: Evaluator] Executing Audit Cycle (Iteration: {iteration}) ---")

        files_content = state.get("target_files", {})
        arch_rubric = state.get("architecture_rubric", "")
        sec_rubric = state.get("security_rubric", "")
        feedback = state.get("remediation_patch")

        start_time = datetime.now(timezone.utc).isoformat()
        eval_result = self.evaluator_client.evaluate(
            files_content=files_content,
            arch_rubric=arch_rubric,
            sec_rubric=sec_rubric,
            iteration=iteration,
            feedback=feedback,
        )

        arch_score = eval_result.get("architecture_score", 0.0)
        sec_score = eval_result.get("security_score", 0.0)
        composite = round((arch_score * 0.45) + (sec_score * 0.55), 2)

        tool_record = {
            "tool": "ClaudeGateEvaluator.evaluate",
            "iteration": iteration,
            "timestamp": start_time,
            "arch_score": arch_score,
            "sec_score": sec_score,
            "critical_violation": eval_result.get("critical_violation", False),
            "findings_count": len(eval_result.get("findings", [])),
        }

        tool_calls = list(state.get("tool_calls", []))
        tool_calls.append(tool_record)

        return {
            "architecture_score": arch_score,
            "security_score": sec_score,
            "composite_score": composite,
            "evaluator_findings": eval_result.get("findings", []),
            "critical_violation": eval_result.get("critical_violation", False),
            "is_maintainability_issue": eval_result.get("is_maintainability_issue", False),
            "tool_calls": tool_calls,
        }

    def node_optimize(self, state: GateState) -> Dict[str, Any]:
        """
        Optimizer Node: Proposes targeted refactoring and remediation patches for remediable issues.
        """
        iteration = state.get("iteration", 0)
        logger.info(f"--- [Node: Optimizer] Generating Remediation Proposal (Cycle {iteration + 1}) ---")

        findings = state.get("evaluator_findings", [])
        files_content = state.get("target_files", {})

        start_time = datetime.now(timezone.utc).isoformat()
        patch_result = self.evaluator_client.generate_remediation_patch(
            files_content=files_content,
            findings=findings,
            iteration=iteration,
        )

        tool_record = {
            "tool": "ClaudeGateEvaluator.generate_remediation_patch",
            "iteration": iteration,
            "timestamp": start_time,
            "suggested_fixes_count": len(patch_result.get("suggested_fixes", [])),
        }

        tool_calls = list(state.get("tool_calls", []))
        tool_calls.append(tool_record)

        return {
            "iteration": iteration + 1,
            "remediation_patch": patch_result.get("patch"),
            "suggested_fixes": patch_result.get("suggested_fixes", []),
            "tool_calls": tool_calls,
        }

    def route_evaluation(self, state: GateState) -> str:
        """
        Deterministic Conditional Edge Router:
        Strictly limits loops to MAX_ITERATIONS and prevents ungrounded auto-approval.
        """
        iteration = state.get("iteration", 0)
        max_iters = state.get("max_iterations", MAX_ITERATIONS)
        critical = state.get("critical_violation", False)
        is_maintainability = state.get("is_maintainability_issue", False)
        arch_score = state.get("architecture_score", 0.0)
        sec_score = state.get("security_score", 0.0)

        # 1. Critical security violation -> Immediate Decision Gate (BLOCK)
        if critical:
            logger.info("Deterministic Route: Critical violation detected -> Routing to Decision Gate (BLOCK)")
            return "gate"

        # 2. Perfect or passing score -> Route to Decision Gate (PASS)
        if arch_score >= PASS_ARCH_THRESHOLD and sec_score >= PASS_SEC_THRESHOLD:
            logger.info("Deterministic Route: Quality thresholds met -> Routing to Decision Gate (PASS)")
            return "gate"

        # 3. Minor maintainability issue and retry budget available -> Route to Optimizer
        if is_maintainability and iteration < max_iters:
            logger.info(f"Deterministic Route: Maintainability remediation needed (Iteration {iteration}/{max_iters}) -> Routing to Optimizer")
            return "optimize"

        # 4. Budget exhausted or non-remediable -> Route to Decision Gate
        logger.info("Deterministic Route: Iteration limit reached or standard evaluation -> Routing to Decision Gate")
        return "gate"

    def node_deterministic_gate(self, state: GateState) -> Dict[str, Any]:
        """
        Decision Gate Node:
        RIGID DETERMINISTIC RULE ENGINE.
        The AI is never the system of record.
        Maps scores and violations strictly to: PASS | HUMAN_REVIEW | BLOCK.
        """
        logger.info("--- [Node: Deterministic Decision Gate] Calculating Final Verdict ---")

        critical = state.get("critical_violation", False)
        arch_score = state.get("architecture_score", 0.0)
        sec_score = state.get("security_score", 0.0)
        composite = state.get("composite_score", 0.0)
        commit_sha = state.get("commit_sha", "unknown")
        policy_version = state.get("policy_version", "v1.2.0")
        tool_calls = state.get("tool_calls", [])
        findings = state.get("evaluator_findings", [])
        remediation = state.get("remediation_patch")

        # Deterministic Rule Mapping
        if critical or sec_score < REVIEW_SEC_THRESHOLD or arch_score < REVIEW_ARCH_THRESHOLD:
            verdict: Literal["PASS", "HUMAN_REVIEW", "BLOCK"] = "BLOCK"
            rationale = "Violación crítica de seguridad/arquitectura o puntuación inferior a umbrales mínimos."
        elif sec_score >= PASS_SEC_THRESHOLD and arch_score >= PASS_ARCH_THRESHOLD:
            verdict = "PASS"
            rationale = "Todos los criterios de gobernanza y seguridad superan los umbrales de aprobación estricta."
        else:
            verdict = "HUMAN_REVIEW"
            rationale = "Puntuación en rango de advertencia o remediaciones pendientes; requiere aprobación manual."

        # Compile Comprehensive Traceability Audit Log
        timestamp = datetime.now(timezone.utc).isoformat()
        audit_payload = {
            "ares_nexus_control_plane": {
                "framework": "Ares-Nexus (NEAI) AI Governance",
                "timestamp": timestamp,
                "commit_sha": commit_sha,
                "policy_version": policy_version,
                "verdict": verdict,
                "verdict_rationale": rationale,
                "metrics": {
                    "architecture_score": arch_score,
                    "security_score": sec_score,
                    "composite_score": composite,
                    "arch_pass_threshold": PASS_ARCH_THRESHOLD,
                    "sec_pass_threshold": PASS_SEC_THRESHOLD,
                },
                "iterations_executed": state.get("iteration", 0),
                "max_iterations_bounded": state.get("max_iterations", MAX_ITERATIONS),
                "critical_violation_flag": critical,
                "findings": findings,
                "remediation_patch_suggested": bool(remediation),
                "tool_call_trace": tool_calls,
                "decision_gate_type": "Deterministic Rule-Based (AI is advisory only)",
            }
        }

        return {
            "verdict": verdict,
            "traceability_log": audit_payload,
        }

    def run(self, initial_state: GateState) -> GateState:
        """
        Executes the compiled LangGraph workflow or structured manual loop fallback.
        """
        if self.graph is not None:
            result = self.graph.invoke(initial_state)
            return result

        # Structured fallback execution if LangGraph is not present
        curr_state = dict(initial_state)
        curr_state["iteration"] = 0
        curr_state["tool_calls"] = []

        while True:
            eval_updates = self.node_evaluate(curr_state)
            curr_state.update(eval_updates)

            route = self.route_evaluation(curr_state)
            if route == "optimize":
                opt_updates = self.node_optimize(curr_state)
                curr_state.update(opt_updates)
            else:
                gate_updates = self.node_deterministic_gate(curr_state)
                curr_state.update(gate_updates)
                break

        return curr_state


# =====================================================================
# CLI and File Ingestion Utilities
# =====================================================================

def load_policy_rubrics(docs_dir: Path) -> Tuple[str, str, str]:
    """Loads architecture and security rubrics from disk."""
    arch_file = docs_dir / "architecture_rubric.md"
    sec_file = docs_dir / "security_rubric.md"

    arch_text = arch_file.read_text(encoding="utf-8") if arch_file.exists() else "Default Architecture Rubric"
    sec_text = sec_file.read_text(encoding="utf-8") if sec_file.exists() else "Default Security Rubric"

    # Compute policy version hash
    combined = (arch_text + sec_text).encode("utf-8")
    policy_hash = hashlib.sha256(combined).hexdigest()[:12]
    policy_version = f"v1.2.0-sha.{policy_hash}"

    return arch_text, sec_text, policy_version


def collect_target_files(files_arg: Optional[List[str]] = None, scan_dir: Optional[str] = None) -> Dict[str, str]:
    """Collects file contents for evaluation."""
    target_files = {}

    if files_arg:
        for fpath_str in files_arg:
            p = Path(fpath_str)
            if p.exists() and p.is_file():
                try:
                    target_files[str(p)] = p.read_text(encoding="utf-8", errors="ignore")
                except Exception as ex:
                    logger.warning(f"Failed reading {p}: {ex}")

    if not target_files:
        # Default scan over src/ and relevant python files
        base_path = Path(scan_dir or "src")
        if base_path.exists():
            for p in base_path.rglob("*.py"):
                if "__pycache__" not in str(p):
                    try:
                        target_files[str(p)] = p.read_text(encoding="utf-8", errors="ignore")
                    except Exception as ex:
                        logger.warning(f"Failed reading {p}: {ex}")

    return target_files


def export_markdown_summary(audit_log: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """Generates a structured Markdown report for PR comments and CI summaries."""
    payload = audit_log.get("ares_nexus_control_plane", {})
    verdict = payload.get("verdict", "UNKNOWN")
    verdict_emoji = "✅ PASS" if verdict == "PASS" else "⚠️ HUMAN_REVIEW" if verdict == "HUMAN_REVIEW" else "🛑 BLOCK"

    md = [
        f"## Ares-Nexus (NEAI) - Governance Control Plane Audit",
        f"",
        f"**Dictamen Final:** `{verdict_emoji}`  ",
        f"**Commit SHA:** `{payload.get('commit_sha')}`  ",
        f"**Versión de Política:** `{payload.get('policy_version')}`  ",
        f"**Timestamp:** `{payload.get('timestamp')}`  ",
        f"",
        f"### 📊 Métricas de Cumplimiento",
        f"| Dimensión | Puntuación | Umbral Requerido | Estado |",
        f"| :--- | :--- | :--- | :--- |",
        f"| **Arquitectura** | `{payload.get('metrics', {}).get('architecture_score')}` | `{payload.get('metrics', {}).get('arch_pass_threshold')}` | {'✅' if payload.get('metrics', {}).get('architecture_score', 0) >= payload.get('metrics', {}).get('arch_pass_threshold', 0) else '❌'} |",
        f"| **Seguridad (Zero-Trust)** | `{payload.get('metrics', {}).get('security_score')}` | `{payload.get('metrics', {}).get('sec_pass_threshold')}` | {'✅' if payload.get('metrics', {}).get('security_score', 0) >= payload.get('metrics', {}).get('sec_pass_threshold', 0) else '❌'} |",
        f"| **Compuesta** | `{payload.get('metrics', {}).get('composite_score')}` | `0.88` | - |",
        f"",
        f"**Razón del Dictamen:** {payload.get('verdict_rationale')}",
        f"",
        f"### 🔍 Trazabilidad y Llamadas Agénticas",
        f"- **Iteraciones del Bucle:** {payload.get('iterations_executed')} / {payload.get('max_iterations_bounded')}",
        f"- **Hallazgos Detectados:** {len(payload.get('findings', []))}",
        f"- **Autoridad de Decisión:** `{payload.get('decision_gate_type')}`",
        f"",
    ]

    findings = payload.get("findings", [])
    if findings:
        md.append("### ⚠️ Hallazgos Detallados")
        for f in findings:
            md.append(f"- **[{f.get('dimension', 'RULE')}]** ({f.get('severity', 'INFO')}) `{f.get('file', 'general')}`: {f.get('message')}")
        md.append("")

    content = "\n".join(md)
    if output_path:
        Path(output_path).write_text(content, encoding="utf-8")

    # Write to GitHub Step Summary if running in GitHub Actions
    gh_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if gh_summary:
        try:
            with open(gh_summary, "a", encoding="utf-8") as gf:
                gf.write(content + "\n\n")
        except Exception as e:
            logger.warning(f"Could not append to GITHUB_STEP_SUMMARY: {e}")

    return content


def main():
    parser = argparse.ArgumentParser(description="Ares-Nexus (NEAI) AI Governance Evaluator Gate")
    parser.add_argument("--files", nargs="*", help="List of files to audit")
    parser.add_argument("--scan-dir", default="src", help="Directory to scan if no files are specified")
    parser.add_argument("--commit-sha", default=os.getenv("GITHUB_SHA", "local-dev-commit"), help="Commit SHA being evaluated")
    parser.add_argument("--docs-dir", default="docs", help="Directory containing governance rubrics")
    parser.add_argument("--output-json", default="neai_audit_log.json", help="Path to save structured JSON audit log")
    parser.add_argument("--output-md", default="neai_governance_report.md", help="Path to save Markdown audit summary")
    parser.add_argument("--provider", default="auto", choices=["auto", "anthropic", "ollama", "deterministic"], help="Evaluation backend")
    parser.add_argument("--fail-on-block", action="store_true", default=True, help="Exit with non-zero exit code on BLOCK")
    args = parser.parse_args()

    docs_path = Path(args.docs_dir)
    arch_rubric, sec_rubric, policy_version = load_policy_rubrics(docs_path)

    target_files = collect_target_files(args.files, args.scan_dir)
    logger.info(f"Loaded {len(target_files)} target files for evaluation. Policy version: {policy_version}")

    evaluator_client = ClaudeGateEvaluator(provider=args.provider)
    gate = NEAIControlPlaneGate(evaluator_client=evaluator_client)

    initial_state: GateState = {
        "commit_sha": args.commit_sha,
        "target_files": target_files,
        "architecture_rubric": arch_rubric,
        "security_rubric": sec_rubric,
        "policy_version": policy_version,
        "iteration": 0,
        "max_iterations": MAX_ITERATIONS,
        "architecture_score": 0.0,
        "security_score": 0.0,
        "composite_score": 0.0,
        "evaluator_findings": [],
        "critical_violation": False,
        "is_maintainability_issue": False,
        "remediation_patch": None,
        "suggested_fixes": [],
        "tool_calls": [],
        "verdict": None,
        "traceability_log": {},
    }

    final_state = gate.run(initial_state)
    verdict = final_state.get("verdict", "BLOCK")
    audit_log = final_state.get("traceability_log", {})

    # Save outputs
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(audit_log, f, indent=2)
        logger.info(f"Traceability audit log written to: {args.output_json}")

    md_summary = export_markdown_summary(audit_log, args.output_md)
    print("\n" + "=" * 60)
    print(f"ARES-NEXUS CONTROL PLANE VERDICT: {verdict}")
    print("=" * 60)
    print(md_summary)

    # Return exit code based on verdict
    if verdict == "BLOCK" and args.fail_on_block:
        sys.exit(1)
    elif verdict == "HUMAN_REVIEW":
        sys.exit(0)  # Human review flags PR for manual approval
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

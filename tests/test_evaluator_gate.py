"""
Unit and Integration Tests for Ares-Nexus (NEAI) AI Governance Evaluator Gate.
Verifies LangGraph StateGraph topology, bounded iterations, remediation proposals,
deterministic decision gate mappings, and full traceability logs.
"""
import unittest
import json
import tempfile
from pathlib import Path

from src.evaluator_gate import (
    NEAIControlPlaneGate,
    GeminiGateEvaluator,
    ClaudeGateEvaluator,
    GateState,
    load_policy_rubrics,
    collect_target_files,
    export_markdown_summary,
    MAX_ITERATIONS,
)


class TestEvaluatorGate(unittest.TestCase):
    def setUp(self):
        self.evaluator = ClaudeGateEvaluator(provider="deterministic")
        self.gate = NEAIControlPlaneGate(evaluator_client=self.evaluator)
        self.docs_dir = Path("docs")
        self.arch_rubric, self.sec_rubric, self.policy_ver = load_policy_rubrics(self.docs_dir)

    def test_clean_code_yields_pass(self):
        """Clean code meeting all rubrics should deterministically produce PASS."""
        clean_code = """
from typing import Protocol

class Repository(Protocol):
    def get_data(self) -> str:
        ...

class CleanService:
    def __init__(self, repo: Repository):
        self._repo = repo

    def execute(self) -> str:
        return self._repo.get_data()
"""
        initial_state: GateState = {
            "commit_sha": "a1b2c3d4e5f6",
            "target_files": {"src/application/clean_service.py": clean_code},
            "architecture_rubric": self.arch_rubric,
            "security_rubric": self.sec_rubric,
            "policy_version": self.policy_ver,
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

        result = self.gate.run(initial_state)

        self.assertEqual(result["verdict"], "PASS")
        self.assertGreaterEqual(result["security_score"], 0.90)
        self.assertGreaterEqual(result["architecture_score"], 0.85)
        self.assertFalse(result["critical_violation"])

        # Check Traceability Log
        audit = result["traceability_log"]["ares_nexus_control_plane"]
        self.assertEqual(audit["commit_sha"], "a1b2c3d4e5f6")
        self.assertEqual(audit["verdict"], "PASS")
        self.assertTrue(len(audit["tool_call_trace"]) >= 1)

    def test_hardcoded_secret_yields_block(self):
        """Presence of hardcoded secret must trigger critical violation and BLOCK."""
        leaky_code = """
import os

API_KEY = "sk-1234567890abcdef1234567890abcdef"

def call_ai():
    return API_KEY
"""
        initial_state: GateState = {
            "commit_sha": "deadbeef1234",
            "target_files": {"src/leaky.py": leaky_code},
            "architecture_rubric": self.arch_rubric,
            "security_rubric": self.sec_rubric,
            "policy_version": self.policy_ver,
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

        result = self.gate.run(initial_state)

        self.assertEqual(result["verdict"], "BLOCK")
        self.assertTrue(result["critical_violation"])
        audit = result["traceability_log"]["ares_nexus_control_plane"]
        self.assertEqual(audit["verdict"], "BLOCK")
        self.assertTrue(any(f["dimension"] == "SEC-01" for f in audit["findings"]))

    def test_unsafe_execution_yields_block(self):
        """Dynamic code execution with eval/exec must trigger BLOCK."""
        unsafe_code = """
def run_dynamic(user_input: str):
    return eval(user_input)
"""
        initial_state: GateState = {
            "commit_sha": "badcall9999",
            "target_files": {"src/unsafe.py": unsafe_code},
            "architecture_rubric": self.arch_rubric,
            "security_rubric": self.sec_rubric,
            "policy_version": self.policy_ver,
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

        result = self.gate.run(initial_state)

        self.assertEqual(result["verdict"], "BLOCK")
        self.assertTrue(result["critical_violation"])

    def test_maintainability_triggers_optimizer_loop(self):
        """Maintainability issue should trigger Optimizer proposal and stay bounded."""
        code_with_maintainability_issue = """
def process():
    try:
        do_something()
    except:
        pass
"""
        initial_state: GateState = {
            "commit_sha": "looptest123",
            "target_files": {"src/application/service.py": code_with_maintainability_issue},
            "architecture_rubric": self.arch_rubric,
            "security_rubric": self.sec_rubric,
            "policy_version": self.policy_ver,
            "iteration": 0,
            "max_iterations": 2,
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

        result = self.gate.run(initial_state)

        # Should execute optimization iteration and generate remediation patch
        self.assertIsNotNone(result.get("remediation_patch"))
        self.assertLessEqual(result.get("iteration", 0), 2)
        self.assertIn(result["verdict"], ["PASS", "HUMAN_REVIEW", "BLOCK"])

    def test_markdown_summary_generation(self):
        """Ensure Markdown summary renders governance metrics and audit data."""
        mock_audit_log = {
            "ares_nexus_control_plane": {
                "framework": "Ares-Nexus (NEAI)",
                "timestamp": "2026-10-02T18:00:00Z",
                "commit_sha": "abcdef123456",
                "policy_version": "v1.2.0-test",
                "verdict": "PASS",
                "verdict_rationale": "All rules passed.",
                "metrics": {
                    "architecture_score": 0.95,
                    "security_score": 1.0,
                    "composite_score": 0.98,
                    "arch_pass_threshold": 0.85,
                    "sec_pass_threshold": 0.90,
                },
                "iterations_executed": 1,
                "max_iterations_bounded": 2,
                "findings": [],
                "decision_gate_type": "Deterministic Rule-Based",
            }
        }
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tf:
            summary = export_markdown_summary(mock_audit_log, tf.name)
            self.assertIn("Ares-Nexus (NEAI) - Governance Control Plane Audit", summary)
            self.assertIn("PASS", summary)
            self.assertIn("abcdef123456", summary)

    def test_gemini_evaluator_configuration_and_mock(self):
        """Test GeminiGateEvaluator initialization and structured parsing with mock responses."""
        evaluator = GeminiGateEvaluator(provider="gemini", model="gemini-1.5-flash")
        self.assertEqual(evaluator.model, "gemini-1.5-flash")
        self.assertEqual(evaluator.provider, "gemini")

        # Mock structured LLM output from Gemini
        mock_gemini_json = json.dumps({
            "architecture_score": 0.92,
            "security_score": 0.96,
            "critical_violation": False,
            "is_maintainability_issue": False,
            "findings": [],
            "recommendations": ["Architecture meets standard."],
        })
        evaluator._call_gemini = lambda prompt, is_json=True: mock_gemini_json

        res = evaluator.evaluate(
            files_content={"src/test.py": "def foo(): pass"},
            arch_rubric=self.arch_rubric,
            sec_rubric=self.sec_rubric,
            iteration=0,
        )
        self.assertEqual(res["architecture_score"], 0.92)
        self.assertEqual(res["security_score"], 0.96)
        self.assertFalse(res["critical_violation"])

        # Test patch generation mock
        mock_patch_json = json.dumps({
            "patch": "# Remediation patch",
            "suggested_fixes": ["Fix type annotations"],
        })
        evaluator._call_gemini = lambda prompt, is_json=True: mock_patch_json
        patch_res = evaluator.generate_remediation_patch(
            files_content={"src/test.py": "def foo(): pass"},
            findings=[{"dimension": "ARCH-01", "message": "Add type hints"}],
            iteration=0,
        )
        self.assertEqual(patch_res["patch"], "# Remediation patch")
        self.assertEqual(patch_res["suggested_fixes"], ["Fix type annotations"])


if __name__ == "__main__":
    unittest.main()

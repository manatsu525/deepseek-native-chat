#!/usr/bin/env python3
"""Run the small regression set used during routine edits.

The complete unittest suite remains available with:
    .venv/bin/python -m unittest discover -s tests -p 'test_*.py'

This default set is intentionally capped at 50 cases so a normal edit does
not spend time repeating every historical regression test.
"""

from __future__ import annotations

import pathlib
import sys
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS_ROOT = PROJECT_ROOT / "tests"
MAX_SMOKE_TESTS = 50

# Keep this balanced across the persistent workspace, context/tool loop,
# custom protocol translation, web evidence, prompts, and code execution.
SMOKE_SPECS = (
    "test_workspace.FileViewRewriteTests",
    "test_workspace.WorkspaceTests",
    "test_context.CompactRequestTests.test_under_budget_is_untouched",
    "test_context.CompactRequestTests.test_old_results_become_stubs_and_recent_ones_survive",
    "test_context.CompactRequestTests.test_whole_rounds_are_dropped_only_as_a_last_resort",
    "test_context.CompactRequestTests.test_checkpoint_carries_files_plan_and_sources",
    "test_plan_execution.ExecutionPlanTests",
    "test_plan_execution.PlanLoopTests",
    "test_agent_tools.AgentToolTests",
    "test_custom_responses.CustomResponsesProtocolTests",
    "test_custom_responses.CustomResponsesConnectionTests",
    "test_web_evidence.WebEvidenceTests",
    "test_research_prompt.SystemPromptTests",
    "test_code_runner.CodeRunnerTests",
)


def build_suite() -> unittest.TestSuite:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(TESTS_ROOT))
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    for spec in SMOKE_SPECS:
        suite.addTests(loader.loadTestsFromName(spec))
    count = suite.countTestCases()
    if count > MAX_SMOKE_TESTS:
        raise RuntimeError(f"smoke suite has {count} tests; maximum is {MAX_SMOKE_TESTS}")
    return suite


def main() -> int:
    suite = build_suite()
    count = suite.countTestCases()
    print(f"Running {count} smoke tests (full suite is opt-in).")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

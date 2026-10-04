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
    "test_grok_tools.GrokToolsTests",
    "test_grok_tools.GrokLoopTests",
    "test_workspace.WorkspaceTests.test_paths_cannot_escape_or_follow_symlink",
    "test_workspace.WorkspaceTests.test_agent_shared_workspace_lists_and_resolves_only_shared_files",
    "test_workspace.WorkspaceTests.test_agent_workspace_browses_one_level_and_deletes_directories",
    "test_agent_compaction.CompactionParityTests.test_meter_utf8_native_mirrors_and_reseed",
    "test_agent_compaction.CompactionLoopTests.test_ordinary_chat_compacts_and_recovers_private_archive_all_protocols",
    "test_agent_compaction.CompactionLoopTests.test_two_successive_compactions_tool_receipts_and_all_protocols",
    "test_agent_session.AgentSessionTests",
    "test_agent_session.AgentSessionLoopTests.test_committed_tools_replay_all_protocols",
    "test_agent_session.AgentSessionLoopTests.test_main_cross_turn_and_restart_use_committed_history",
    "test_agent_session.AgentSessionLoopTests.test_restored_history_compacts_without_losing_user_requests",
    "test_plan_execution.PlanLoopTests.test_unfinished_checklist_does_not_force_extra_rounds",
    "test_plan_execution.PlanLoopTests.test_503_waits_five_seconds_and_reports_recovery",
    "test_plan_execution.PlanLoopTests.test_configured_http_status_retries_with_same_policy",
    "test_agent_tools.AgentToolTests",
    "test_custom_responses.CustomResponsesProtocolTests.test_output_token_field_matches_each_custom_protocol",
    "test_custom_responses.CustomResponsesProtocolTests.test_chat_tool_history_becomes_responses_items",
    "test_responses_compaction.ResponsesCompactionTests",
    "test_custom_responses.CustomResponsesConnectionTests",
    "test_web_evidence.WebEvidenceTests",
    "test_code_runner.CodeRunnerTests.test_rejects_non_python_and_escaping_paths",
    "test_code_runner.CodeRunnerTests.test_runs_disposable_copy_with_systemd_limits",
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

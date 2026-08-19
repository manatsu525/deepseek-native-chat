from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app import workspace
from app.workspace import ConversationWorkspace, WorkspaceError


class WorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.original_root = workspace.WORKSPACES_DIR
        workspace.WORKSPACES_DIR = Path(self.temp.name)
        self.workspace = ConversationWorkspace(7, "conversation123")

    def tearDown(self) -> None:
        workspace.WORKSPACES_DIR = self.original_root
        self.temp.cleanup()

    def test_write_read_patch_search_list_and_delete(self) -> None:
        written = self.workspace.write_file("src/app.py", "name = 'old'\nprint(name)\n")
        self.assertEqual(written["path"], "src/app.py")
        self.assertEqual(self.workspace.read_file("src/app.py"), "name = 'old'\nprint(name)\n")

        patched = self.workspace.apply_patch("src/app.py", "'old'", "'new'")
        self.assertEqual(patched["replacements"], 1)
        self.assertEqual(self.workspace.read_file("src/app.py"), "name = 'new'\nprint(name)\n")
        self.assertEqual(self.workspace.search_files("PRINT")["matches"][0]["line"], 2)
        self.assertEqual(self.workspace.list_files()[0]["path"], "src/app.py")
        self.workspace.delete_file("src/app.py")
        self.assertEqual(self.workspace.list_files(), [])

    def test_paths_cannot_escape_or_follow_symlink(self) -> None:
        for invalid in ("../secret", "/etc/passwd", "folder/../../secret", "folder\\file"):
            with self.assertRaises(WorkspaceError):
                self.workspace.write_file(invalid, "no")

        self.workspace.root.mkdir(parents=True)
        outside = Path(self.temp.name) / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        (self.workspace.root / "link").symlink_to(outside)
        with self.assertRaises(WorkspaceError):
            self.workspace.read_file("link")
        with self.assertRaises(WorkspaceError):
            self.workspace.write_file("link", "changed")
        self.assertEqual(outside.read_text(encoding="utf-8"), "secret")

    def test_patch_refuses_ambiguous_match(self) -> None:
        self.workspace.write_file("same.txt", "x\nx\n")
        with self.assertRaises(WorkspaceError):
            self.workspace.apply_patch("same.txt", "x", "y")
        result = self.workspace.apply_patch("same.txt", "x", "y", True)
        self.assertEqual(result["replacements"], 2)
        self.assertEqual(self.workspace.read_file("same.txt"), "y\ny\n")

    def test_execute_returns_model_readable_json(self) -> None:
        result = self.workspace.execute("write_file", {"path": "index.html", "content": "<h1>Hi</h1>"})
        self.assertIn('"ok": true', result)
        self.assertIn('"path": "index.html"', result)

        snapshot = self.workspace.execute("read_file", {"path": "index.html"})
        self.assertIn('"revision":', snapshot)
        self.assertIn('"numbered_content": "1|<h1>Hi</h1>"', snapshot)

    def test_tool_schema_is_stable_as_files_change(self) -> None:
        before = self.workspace.tool_definitions()
        self.workspace.write_file("index.html", "<h1>Hi</h1>")
        self.workspace.write_file("src/app.js", "start()")
        self.workspace.write_file("tests/test_app.py", "print('ok')")
        after = self.workspace.tool_definitions()
        self.assertEqual(before, after)
        names = {item["function"]["name"] for item in after}
        self.assertIn("apply_line_edits", names)
        self.assertNotIn("apply_patch", names)
        self.assertNotIn("apply_patch_batch", names)
        self.assertIn("run_python", names)
        self.assertIn("check_web_syntax", names)
        for tool in after:
            path_schema = tool["function"]["parameters"]["properties"].get("path")
            if path_schema:
                self.assertNotIn("enum", path_schema)

    def test_edit_access_can_modify_but_cannot_run_validation(self) -> None:
        names = {item["function"]["name"] for item in self.workspace.tool_definitions("edit")}
        self.assertIn("list_files", names)
        self.assertIn("read_file", names)
        self.assertIn("search_files", names)
        self.assertIn("write_file", names)
        self.assertIn("apply_line_edits", names)
        self.assertNotIn("run_python", names)
        self.assertNotIn("check_web_syntax", names)

    def test_revisioned_line_edits_are_atomic_and_do_not_match_old_text(self) -> None:
        self.workspace.write_file("app.js", "one\ntwo\nthree\nfour\n")
        snapshot = self.workspace.read_snapshot("app.js")
        self.assertEqual(snapshot["numbered_content"], "1|one\n2|two\n3|three\n4|four")
        result = self.workspace.apply_line_edits(
            "app.js",
            snapshot["revision"],
            [
                {"start_line": 2, "end_line": 2, "new_text": "TWO"},
                {"start_line": 4, "end_line": 3, "new_text": "inserted"},
            ],
        )
        self.assertNotEqual(result["revision"], snapshot["revision"])
        self.assertEqual(self.workspace.read_file("app.js"), "one\nTWO\nthree\ninserted\nfour\n")

        latest = self.workspace.read_snapshot("app.js")
        self.workspace.apply_line_edits(
            "app.js",
            latest["revision"],
            [{"start_line": 5, "end_line": 5, "new_text": "FOUR"}],
        )
        self.assertTrue(self.workspace.read_file("app.js").endswith("FOUR\n"))

    def test_snapshot_can_read_a_numbered_line_range(self) -> None:
        self.workspace.write_file("app.js", "one\ntwo\nthree\nfour\n")
        snapshot = self.workspace.read_snapshot("app.js", 2, 3)
        self.assertEqual(snapshot["returned_from_line"], 2)
        self.assertEqual(snapshot["returned_through_line"], 3)
        self.assertEqual(snapshot["line_count"], 4)
        self.assertEqual(snapshot["numbered_content"], "2|two\n3|three")
        self.assertFalse(snapshot["truncated"])

    def test_line_edits_reject_stale_revision_without_changing_file(self) -> None:
        self.workspace.write_file("app.js", "one\ntwo\n")
        stale = self.workspace.read_snapshot("app.js")
        self.workspace.write_file("app.js", "one\nchanged\n")
        before = self.workspace.read_file("app.js")
        with self.assertRaisesRegex(WorkspaceError, "文件版本已经变化"):
            self.workspace.apply_line_edits(
                "app.js",
                stale["revision"],
                [{"start_line": 2, "end_line": 2, "new_text": "TWO"}],
            )
        self.assertEqual(self.workspace.read_file("app.js"), before)

    def test_line_edits_reject_overlapping_ranges_atomically(self) -> None:
        original = "one\ntwo\nthree\n"
        self.workspace.write_file("app.js", original)
        snapshot = self.workspace.read_snapshot("app.js")
        with self.assertRaisesRegex(WorkspaceError, "范围重叠"):
            self.workspace.apply_line_edits(
                "app.js",
                snapshot["revision"],
                [
                    {"start_line": 1, "end_line": 2, "new_text": "first"},
                    {"start_line": 2, "end_line": 3, "new_text": "second"},
                ],
            )
        self.assertEqual(self.workspace.read_file("app.js"), original)

    def test_batch_patch_is_atomic_and_uses_one_snapshot(self) -> None:
        original = "alpha = 1\nbeta = 2\ngamma = 3\n"
        self.workspace.write_file("app.py", original)
        result = self.workspace.apply_patch_batch(
            "app.py",
            [
                {"old_text": "alpha = 1", "new_text": "alpha = 10"},
                {"old_text": "gamma = 3", "new_text": "gamma = 30"},
            ],
        )
        self.assertEqual(result["changes"], 2)
        self.assertEqual(self.workspace.read_file("app.py"), "alpha = 10\nbeta = 2\ngamma = 30\n")

        before_failure = self.workspace.read_file("app.py")
        with self.assertRaises(WorkspaceError):
            self.workspace.apply_patch_batch(
                "app.py",
                [
                    {"old_text": "beta = 2", "new_text": "beta = 20"},
                    {"old_text": "missing", "new_text": "value"},
                ],
            )
        self.assertEqual(self.workspace.read_file("app.py"), before_failure)


if __name__ == "__main__":
    unittest.main()

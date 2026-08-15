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

    def test_existing_file_tools_are_constrained_to_real_paths(self) -> None:
        self.workspace.write_file("index.html", "<h1>Hi</h1>")
        self.workspace.write_file("src/app.js", "start()")
        self.workspace.write_file("tests/test_app.py", "print('ok')")
        tools = {item["function"]["name"]: item for item in self.workspace.tool_definitions()}
        expected = ["index.html", "src/app.js", "tests/test_app.py"]
        for name in ("read_file", "apply_patch", "apply_patch_batch", "delete_file"):
            schema = tools[name]["function"]["parameters"]["properties"]["path"]
            self.assertEqual(schema["enum"], expected)
        self.assertEqual(
            tools["run_python"]["function"]["parameters"]["properties"]["path"]["enum"],
            ["tests/test_app.py"],
        )
        self.assertEqual(
            tools["check_web_syntax"]["function"]["parameters"]["properties"]["path"]["enum"],
            ["index.html", "src/app.js"],
        )
        self.assertNotIn("enum", tools["write_file"]["function"]["parameters"]["properties"]["path"])

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

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
        tools = {item["function"]["name"]: item for item in self.workspace.tool_definitions()}
        expected = ["index.html", "src/app.js"]
        for name in ("read_file", "apply_patch", "delete_file"):
            schema = tools[name]["function"]["parameters"]["properties"]["path"]
            self.assertEqual(schema["enum"], expected)
        self.assertNotIn("enum", tools["write_file"]["function"]["parameters"]["properties"]["path"])


if __name__ == "__main__":
    unittest.main()

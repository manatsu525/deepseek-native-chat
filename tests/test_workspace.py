from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app import workspace
from app.workspace import AgentSharedWorkspace, ConversationWorkspace, WorkspaceError


class FileViewRewriteTests(unittest.TestCase):
    def test_partial_views_become_whole_numbered_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.json").write_text("1\n2\n3\n")
            (root / "big.md").write_text("x" * 40_000)
            (root / "bin.dat").write_bytes(b"\x00\x01")
            resolve = lambda raw: root / raw  # noqa: E731
            command, notes = workspace.expand_file_views(
                "sed -n '2,3p' a.json; echo ===; head -n 1 a.json && tail -1 a.json || tail a.json\nhead -5 big.md; sed -n 1,2p bin.dat; "
                "cat a.json | head -2; grep -n x a.json | sed -n '1,2p'",
                resolve,
            )
            self.assertEqual(
                command,
                "cat -n a.json; echo ===; cat -n a.json && cat -n a.json || cat -n a.json\nhead -5 big.md; sed -n 1,2p bin.dat; "
                "cat a.json | head -2; grep -n x a.json | sed -n '1,2p'",
            )
            self.assertEqual(len(notes), 4)
            self.assertIn("sed -n '2,3p' a.json", notes[0])
            self.assertIn("3 行", notes[0])
            unchanged, none = workspace.expand_file_views("ls -la; grep -rn foo . | head -20; sed -n '1,5p' missing.txt", resolve)
            self.assertEqual(none, [])
            self.assertEqual(unchanged, "ls -la; grep -rn foo . | head -20; sed -n '1,5p' missing.txt")


class WorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.original_root = workspace.WORKSPACES_DIR
        workspace.WORKSPACES_DIR = Path(self.temp.name)
        self.workspace = ConversationWorkspace(7, "conversation123")

    def tearDown(self) -> None:
        workspace.WORKSPACES_DIR = self.original_root
        self.temp.cleanup()

    def test_write_read_edit_search_list_and_delete(self) -> None:
        written = self.workspace.write_file("src/app.py", "name = 'old'\nprint(name)\n")
        self.assertEqual(written["path"], "src/app.py")
        self.assertEqual(self.workspace.read_file("src/app.py"), "name = 'old'\nprint(name)\n")

        edited = self.workspace.edit_file("src/app.py", [{"old_text": "'old'", "new_text": "'new'"}])
        self.assertEqual(edited["replacements"], 1)
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

    def test_edit_refuses_ambiguous_match_unless_replace_all(self) -> None:
        self.workspace.write_file("same.txt", "x\nx\n")
        with self.assertRaisesRegex(WorkspaceError, "出现 2 次"):
            self.workspace.edit_file("same.txt", [{"old_text": "x", "new_text": "y"}])
        result = self.workspace.edit_file("same.txt", [{"old_text": "x", "new_text": "y", "replace_all": True}])
        self.assertEqual(result["replacements"], 2)
        self.assertEqual(self.workspace.read_file("same.txt"), "y\ny\n")

    def test_execute_returns_model_readable_json(self) -> None:
        result = self.workspace.execute("write_file", {"path": "index.html", "content": "<h1>Hi</h1>"})
        self.assertIn('"ok": true', result)
        self.assertIn('"path": "index.html"', result)

        snapshot = json.loads(self.workspace.execute("read_file", {"path": "index.html"}))
        self.assertTrue(snapshot["revision"])
        self.assertEqual(snapshot["content"], "1|<h1>Hi</h1>")
        self.assertEqual((snapshot["from_line"], snapshot["through_line"], snapshot["truncated"]), (1, 1, False))

    def test_tool_schema_is_stable_and_has_no_line_number_editing(self) -> None:
        before = self.workspace.tool_definitions()
        self.workspace.write_file("index.html", "<h1>Hi</h1>")
        self.workspace.write_file("src/app.js", "start()")
        self.workspace.write_file("tests/test_app.py", "print('ok')")
        after = self.workspace.tool_definitions()
        self.assertEqual(before, after)
        names = {item["function"]["name"] for item in after}
        self.assertIn("edit_file", names)
        for removed in ("apply_line_edits", "apply_patch", "apply_patch_batch", "replace_text"):
            self.assertNotIn(removed, names)
        self.assertIn("run_python", names)
        self.assertIn("check_web_syntax", names)
        read_schema = next(item for item in after if item["function"]["name"] == "read_file")["function"]["parameters"]
        self.assertEqual(set(read_schema["properties"]), {"path", "start_line"})
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
        self.assertIn("edit_file", names)
        self.assertNotIn("run_python", names)
        self.assertNotIn("check_web_syntax", names)
        read_only = {item["function"]["name"] for item in self.workspace.tool_definitions("read_only")}
        self.assertNotIn("edit_file", read_only)

    def test_read_returns_whole_file_even_when_a_start_line_is_given(self) -> None:
        self.workspace.write_file("app.js", "one\ntwo\nthree\nfour\n")
        snapshot = self.workspace.read_snapshot("app.js", 3)
        self.assertEqual(snapshot["content"], "1|one\n2|two\n3|three\n4|four")
        self.assertEqual((snapshot["from_line"], snapshot["through_line"], snapshot["line_count"]), (1, 4, 4))
        self.assertIn("note", snapshot)
        with self.assertRaises(WorkspaceError):
            self.workspace.read_snapshot("app.js", 9)

    def test_only_an_oversized_file_is_split_and_continued(self) -> None:
        self.workspace.write_file("big.txt", "".join(f"{'z' * 99}\n" for _ in range(1500)))
        first = self.workspace.read_snapshot("big.txt")
        self.assertTrue(first["truncated"])
        self.assertLessEqual(len(first["content"]), workspace.MAX_READ_CHARS)
        rest = self.workspace.read_snapshot("big.txt", first["next_start_line"])
        self.assertEqual(rest["from_line"], first["through_line"] + 1)
        self.assertEqual(rest["through_line"], 1500)
        self.assertFalse(rest["truncated"])

    def test_edit_is_atomic_across_all_snippets(self) -> None:
        original = "alpha = 1\nbeta = 2\ngamma = 3\n"
        self.workspace.write_file("app.py", original)
        result = self.workspace.edit_file("app.py", [
            {"old_text": "alpha = 1", "new_text": "alpha = 10"},
            {"old_text": "gamma = 3", "new_text": "gamma = 30"},
        ])
        self.assertEqual((result["edits"], result["replacements"]), (2, 2))
        self.assertEqual(self.workspace.read_file("app.py"), "alpha = 10\nbeta = 2\ngamma = 30\n")
        self.assertIn("1|alpha = 10", result["updated_excerpt"])
        self.assertIn("3|gamma = 30", result["updated_excerpt"])

        before_failure = self.workspace.read_file("app.py")
        with self.assertRaisesRegex(WorkspaceError, "第 2 处修改.*整个批次未修改"):
            self.workspace.edit_file("app.py", [
                {"old_text": "beta = 2", "new_text": "beta = 20"},
                {"old_text": "missing", "new_text": "value"},
            ])
        with self.assertRaisesRegex(WorkspaceError, "范围重叠"):
            self.workspace.edit_file("app.py", [
                {"old_text": "alpha = 10\nbeta", "new_text": "x"},
                {"old_text": "beta = 2", "new_text": "y"},
            ])
        with self.assertRaisesRegex(WorkspaceError, "old_text 不能为空"):
            self.workspace.edit_file("app.py", [{"old_text": "", "new_text": "y"}])
        self.assertEqual(self.workspace.read_file("app.py"), before_failure)

    def test_edits_do_not_depend_on_line_numbers(self) -> None:
        self.workspace.write_file("app.js", "a();\nb();\nc();\n")
        # Inserting lines above does not invalidate a later snippet edit.
        self.workspace.edit_file("app.js", [{"old_text": "a();\n", "new_text": "setup();\nmore();\na();\n"}])
        result = self.workspace.edit_file("app.js", [{"old_text": "c();", "new_text": "done();"}])
        self.assertEqual(self.workspace.read_file("app.js"), "setup();\nmore();\na();\nb();\ndone();\n")
        self.assertIn("5|done();", result["updated_excerpt"])

    def test_legacy_patch_names_run_as_edit_file(self) -> None:
        self.workspace.write_file("app.py", "a = 1\nb = 2\n")
        single = json.loads(self.workspace.execute("apply_patch", {"path": "app.py", "old_text": "a = 1", "new_text": "a = 5"}))
        self.assertEqual(single["replacements"], 1)
        batch = json.loads(self.workspace.execute("apply_patch_batch", {"path": "app.py", "patches": [
            {"old_text": "a = 5", "new_text": "a = 6"}, {"old_text": "b = 2", "new_text": "b = 7"}]}))
        self.assertEqual(batch["edits"], 2)
        self.assertEqual(self.workspace.read_file("app.py"), "a = 6\nb = 7\n")
        with self.assertRaises(WorkspaceError):
            self.workspace.execute("apply_line_edits", {"path": "app.py"})

    def test_agent_shared_workspace_lists_and_resolves_only_shared_files(self) -> None:
        shared_root = Path(self.temp.name) / "share"
        shared_root.mkdir()
        (shared_root / "nested").mkdir()
        (shared_root / "nested" / "index.html").write_text("<h1>Agent</h1>", encoding="utf-8")
        outside = Path(self.temp.name) / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        (shared_root / "outside-link").symlink_to(outside)

        agent_workspace = AgentSharedWorkspace(shared_root)
        files = agent_workspace.list_files()
        self.assertEqual(files, [{"path": "nested/index.html", "size": 14, "backend": "agent"}])
        target, relative = agent_workspace.resolve_file("nested/index.html")
        self.assertEqual(relative, "nested/index.html")
        self.assertEqual(target.read_text(encoding="utf-8"), "<h1>Agent</h1>")
        deleted = agent_workspace.delete_file("nested/index.html")
        self.assertEqual(deleted, {"ok": True, "path": "nested/index.html"})
        self.assertFalse(target.exists())
        self.assertEqual(agent_workspace.list_files(), [])
        with self.assertRaises(WorkspaceError):
            agent_workspace.resolve_file("../outside.txt")
        with self.assertRaises(WorkspaceError):
            agent_workspace.resolve_file("outside-link")
        with self.assertRaises(WorkspaceError):
            agent_workspace.delete_file("../outside.txt")
        with self.assertRaises(WorkspaceError):
            agent_workspace.delete_file("outside-link")
        self.assertEqual(outside.read_text(encoding="utf-8"), "secret")

    def test_agent_workspace_browses_one_level_and_deletes_directories(self) -> None:
        shared_root = Path(self.temp.name) / "share2"
        (shared_root / "proj" / "src").mkdir(parents=True)
        (shared_root / "proj" / "src" / "a.py").write_text("print(1)\n", encoding="utf-8")
        (shared_root / "proj" / "README.md").write_text("hi", encoding="utf-8")
        (shared_root / "notes.txt").write_text("n", encoding="utf-8")
        (shared_root / "link").symlink_to(Path(self.temp.name))
        workspace = AgentSharedWorkspace(shared_root)

        root = workspace.list_directory("")
        self.assertEqual((root["path"], root["parent"]), ("", None))
        self.assertEqual([(e["name"], e["type"]) for e in root["entries"]], [("proj", "directory"), ("notes.txt", "file")])
        self.assertEqual(root["entries"][0]["files"], 2)
        self.assertEqual(root["entries"][0]["size"], 11)
        nested = workspace.list_directory("proj/src")
        self.assertEqual((nested["path"], nested["parent"]), ("proj/src", "proj"))
        self.assertEqual(nested["entries"], [{"name": "a.py", "path": "proj/src/a.py", "type": "file", "size": 9}])
        with self.assertRaises(WorkspaceError):
            workspace.list_directory("../")
        with self.assertRaises(WorkspaceError):
            workspace.list_directory("link")

        self.assertEqual(workspace.delete_path("proj/README.md"), {"ok": True, "path": "proj/README.md", "type": "file", "files": 1})
        self.assertEqual(workspace.delete_path("proj"), {"ok": True, "path": "proj", "type": "directory", "files": 1})
        self.assertFalse((shared_root / "proj").exists())
        with self.assertRaises(WorkspaceError):
            workspace.delete_path("")
        with self.assertRaises(WorkspaceError):
            workspace.delete_path("link")
        self.assertTrue((shared_root / "link").exists())
        self.assertEqual([e["name"] for e in workspace.list_directory("")["entries"]], ["notes.txt"])


if __name__ == "__main__":
    unittest.main()

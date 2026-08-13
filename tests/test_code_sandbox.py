"""Offline checks for the coding-agent sandbox."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app import code_sandbox


class CodeSandboxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_write_and_list_files(self) -> None:
        written = code_sandbox.write_files(
            self.workspace,
            [{"path": "hello.py", "content": "VALUE = 1\n"}],
        )
        self.assertEqual(written, ["hello.py"])
        self.assertEqual(code_sandbox.list_files(self.workspace), ["hello.py"])

    def test_rejects_path_escape(self) -> None:
        with self.assertRaises(code_sandbox.SandboxError):
            code_sandbox.write_files(self.workspace, [{"path": "../x.py", "content": "x"}])

    def test_rejects_inline_and_destructive_commands(self) -> None:
        with self.assertRaises(code_sandbox.SandboxError):
            code_sandbox.validate_test_command("python3 -c 'print(1)'", self.workspace)
        with self.assertRaises(code_sandbox.SandboxError):
            code_sandbox.validate_test_command("rm -rf /", self.workspace)
        with self.assertRaises(code_sandbox.SandboxError):
            code_sandbox.validate_test_command("python3 -m pip install x", self.workspace)

    def test_runs_real_unittest_in_workspace(self) -> None:
        code_sandbox.write_files(
            self.workspace,
            [
                {"path": "mod.py", "content": "def add(a, b):\n    return a + b\n"},
                {
                    "path": "test_mod.py",
                    "content": (
                        "import unittest\nfrom mod import add\n"
                        "class T(unittest.TestCase):\n"
                        "    def test_add(self):\n"
                        "        self.assertEqual(add(1, 2), 3)\n"
                    ),
                },
            ],
        )
        results = code_sandbox.run_test_commands(["python3 -m unittest test_mod.py"], self.workspace)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["ok"], results[0])


if __name__ == "__main__":
    unittest.main()

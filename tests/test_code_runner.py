from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import code_runner
from app.code_runner import CodeRunnerError


class CodeRunnerTests(unittest.TestCase):
    def test_rejects_non_python_and_escaping_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for path in ("../bad.py", "/tmp/bad.py", "bad.js", "folder\\bad.py"):
                with self.assertRaises(CodeRunnerError):
                    code_runner.run_python(root, path)

    def test_runs_disposable_copy_with_systemd_limits(self) -> None:
        with tempfile.TemporaryDirectory() as source_temp, tempfile.TemporaryDirectory() as run_temp:
            source = Path(source_temp)
            (source / "main.py").write_text("print('hello')\n", encoding="utf-8")
            completed = subprocess.CompletedProcess([], 0, b"hello\n", b"")
            with patch.object(code_runner, "RUN_ROOT", Path(run_temp)), patch.object(
                code_runner.subprocess, "run", return_value=completed
            ) as run:
                result = code_runner.run_python(source, "main.py", ["world"])
            self.assertTrue(result["ok"])
            self.assertEqual(result["stdout"], "hello\n")
            command = run.call_args.args[0]
            self.assertIn("PrivateNetwork=yes", command)
            self.assertIn("DynamicUser=yes", command)
            self.assertEqual(command[-2:], ["main.py", "world"])
            self.assertEqual(list(Path(run_temp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()

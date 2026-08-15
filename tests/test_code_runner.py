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

    def test_html_checks_inline_handlers_and_local_scripts(self) -> None:
        html = """<!doctype html><button onclick="go()">Go</button>
<script>function go(){ return true; }</script><script src="app.js"></script>"""
        with tempfile.TemporaryDirectory() as source_temp, tempfile.TemporaryDirectory() as run_temp:
            source = Path(source_temp)
            (source / "index.html").write_text(html, encoding="utf-8")
            (source / "app.js").write_text("const ready = true;\n", encoding="utf-8")
            completed = subprocess.CompletedProcess([], 0, b"", b"")
            with patch.object(code_runner, "RUN_ROOT", Path(run_temp)), patch.object(
                code_runner.subprocess, "run", return_value=completed
            ) as run:
                result = code_runner.check_web_syntax(source, "index.html")
            self.assertTrue(result["ok"])
            self.assertTrue(result["html_parsed"])
            self.assertEqual(len(result["checks"]), 3)
            checked = [call.args[0][-1] for call in run.call_args_list]
            self.assertIn("app.js", checked)
            self.assertTrue(any("inline-1.js" in item for item in checked))
            self.assertTrue(any("handler-1.js" in item for item in checked))

    def test_unclosed_script_fails_without_invoking_node(self) -> None:
        with tempfile.TemporaryDirectory() as source_temp, tempfile.TemporaryDirectory() as run_temp:
            source = Path(source_temp)
            (source / "bad.html").write_text("<script>const broken = true", encoding="utf-8")
            with patch.object(code_runner, "RUN_ROOT", Path(run_temp)), patch.object(code_runner.subprocess, "run") as run:
                result = code_runner.check_web_syntax(source, "bad.html")
            self.assertFalse(result["ok"])
            self.assertIn("未闭合", result["error"])
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()

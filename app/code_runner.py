from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit


RUN_ROOT = Path("/run/deepseek-code-runs")
MAX_ARGUMENTS = 20
MAX_ARGUMENT_CHARS = 1000
MAX_OUTPUT_CHARS = 12_000
RUN_TIMEOUT_SECONDS = 12


class CodeRunnerError(ValueError):
    pass


class _HtmlScripts(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.inline: list[tuple[str, bool]] = []
        self.sources: list[str] = []
        self.handlers: list[str] = []
        self._script: list[str] | None = None
        self._module = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {str(key).casefold(): str(value or "") for key, value in attrs}
        for key, value in values.items():
            if key.startswith("on") and value.strip():
                self.handlers.append(value)
        if tag.casefold() != "script":
            return
        script_type = values.get("type", "").strip().casefold()
        if script_type not in {"", "module", "text/javascript", "application/javascript", "text/ecmascript", "application/ecmascript"}:
            return
        source = values.get("src", "").strip()
        if source:
            self.sources.append(source)
            return
        self._module = script_type == "module"
        self._script = []

    def handle_data(self, data: str) -> None:
        if self._script is not None:
            self._script.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "script" and self._script is not None:
            self.inline.append(("".join(self._script), self._module))
            self._script = None
            self._module = False

    @property
    def unclosed_script(self) -> bool:
        return self._script is not None


def _prepare_copy(source_root: Path) -> Path:
    RUN_ROOT.mkdir(parents=True, exist_ok=True, mode=0o711)
    RUN_ROOT.chmod(0o711)
    target_root = Path(tempfile.mkdtemp(prefix="run-", dir=RUN_ROOT))
    target_root.chmod(0o777)
    for source in source_root.rglob("*"):
        if source.is_symlink():
            continue
        relative = source.relative_to(source_root)
        target = target_root / relative
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            target.chmod(0o777)
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            target.chmod(0o666)
    return target_root


def _isolated_command(run_root: Path, executable: str, args: list[str]) -> list[str]:
    return [
        "systemd-run",
        "--quiet",
        "--pipe",
        "--wait",
        "--collect",
        "--service-type=exec",
        "-p", "DynamicUser=yes",
        "-p", "PrivateNetwork=yes",
        "-p", "ProtectSystem=strict",
        "-p", "ProtectHome=yes",
        "-p", "PrivateDevices=yes",
        "-p", "NoNewPrivileges=yes",
        "-p", "RestrictSUIDSGID=yes",
        "-p", "ProtectKernelTunables=yes",
        "-p", "ProtectKernelModules=yes",
        "-p", "ProtectControlGroups=yes",
        "-p", "LockPersonality=yes",
        "-p", "MemoryMax=128M",
        "-p", "CPUQuota=50%",
        "-p", "TasksMax=32",
        "-p", f"RuntimeMaxSec={RUN_TIMEOUT_SECONDS}",
        "-p", f"WorkingDirectory={run_root}",
        "-p", f"ReadWritePaths={run_root}",
        "-E", "PYTHONDONTWRITEBYTECODE=1",
        "-E", "PYTHONNOUSERSITE=1",
        executable,
        *args,
    ]


def _run_isolated(run_root: Path, executable: str, args: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            _isolated_command(run_root, executable, args),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=RUN_TIMEOUT_SECONDS + 8,
            check=False,
            env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
        )
    except subprocess.TimeoutExpired as exc:
        raise CodeRunnerError("代码检查或运行超时，已停止") from exc
    return {
        "ok": completed.returncode == 0,
        "exit_code": int(completed.returncode),
        "stdout": completed.stdout.decode("utf-8", errors="replace")[-MAX_OUTPUT_CHARS:],
        "stderr": completed.stderr.decode("utf-8", errors="replace")[-MAX_OUTPUT_CHARS:],
    }


def _relative_path(value: str, label: str) -> Path:
    path = Path(str(value or ""))
    if not str(value or "").strip() or "\\" in str(value) or path.is_absolute() or ".." in path.parts:
        raise CodeRunnerError(f"{label}路径无效")
    return path


def run_python(source_root: Path, relative_path: str, arguments: Any = None) -> dict[str, Any]:
    path = _relative_path(relative_path, "Python 文件")
    if path.suffix.casefold() != ".py":
        raise CodeRunnerError("受限运行目前只支持 .py 文件")
    raw_arguments = arguments if isinstance(arguments, list) else []
    if len(raw_arguments) > MAX_ARGUMENTS:
        raise CodeRunnerError(f"运行参数最多 {MAX_ARGUMENTS} 个")
    args = [str(item) for item in raw_arguments]
    if any(len(item) > MAX_ARGUMENT_CHARS or "\x00" in item for item in args):
        raise CodeRunnerError("运行参数无效或过长")

    run_root = _prepare_copy(source_root)
    try:
        target = run_root / path
        if not target.is_file() or target.is_symlink():
            raise CodeRunnerError(f"Python 文件不存在：{relative_path}")
        result = _run_isolated(run_root, "/usr/bin/python3", [path.as_posix(), *args])
        return {
            "ok": result["ok"],
            "path": path.as_posix(),
            "exit_code": result["exit_code"],
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "limits": {"network": False, "timeout_seconds": RUN_TIMEOUT_SECONDS, "memory_mb": 128},
        }
    finally:
        shutil.rmtree(run_root, ignore_errors=True)


def check_web_syntax(source_root: Path, relative_path: str) -> dict[str, Any]:
    path = _relative_path(relative_path, "待检查文件")
    if path.suffix.casefold() not in {".js", ".mjs", ".cjs", ".html", ".htm"}:
        raise CodeRunnerError("语法检查仅支持 HTML 和 JavaScript 文件")
    run_root = _prepare_copy(source_root)
    try:
        target = run_root / path
        if not target.is_file() or target.is_symlink():
            raise CodeRunnerError(f"待检查文件不存在：{relative_path}")
        checks: list[tuple[str, Path]] = []
        html_parsed = False
        if path.suffix.casefold() in {".js", ".mjs", ".cjs"}:
            checks.append((path.as_posix(), path))
        else:
            parser = _HtmlScripts()
            try:
                parser.feed(target.read_text(encoding="utf-8"))
                parser.close()
            except (UnicodeDecodeError, ValueError) as exc:
                raise CodeRunnerError(f"HTML 读取或解析失败：{exc}") from exc
            if parser.unclosed_script:
                return {"ok": False, "path": path.as_posix(), "html_parsed": False, "checks": [], "error": "HTML 中存在未闭合的 <script> 标签"}
            html_parsed = True
            generated = run_root / ".deepseek-syntax"
            generated.mkdir(mode=0o777)
            for index, (content, module) in enumerate(parser.inline, 1):
                inline_path = generated / f"inline-{index}{'.mjs' if module else '.js'}"
                inline_path.write_text(content, encoding="utf-8")
                inline_path.chmod(0o666)
                checks.append((f"内联 script #{index}", inline_path.relative_to(run_root)))
            for index, handler in enumerate(parser.handlers, 1):
                handler_path = generated / f"handler-{index}.js"
                handler_path.write_text(f"function __handler(event){{\n{handler}\n}}\n", encoding="utf-8")
                handler_path.chmod(0o666)
                checks.append((f"内联事件处理器 #{index}", handler_path.relative_to(run_root)))
            for source in parser.sources:
                parts = urlsplit(source)
                if parts.scheme or parts.netloc or source.startswith("//"):
                    continue
                local = (target.parent / unquote(parts.path)).resolve()
                if run_root not in local.parents or not local.is_file() or local.suffix.casefold() not in {".js", ".mjs", ".cjs"}:
                    continue
                relative = local.relative_to(run_root)
                checks.append((f"本地脚本 {relative.as_posix()}", relative))
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for label, check_path in checks:
            key = check_path.as_posix()
            if key in seen:
                continue
            seen.add(key)
            result = _run_isolated(run_root, "/usr/bin/node", ["--check", key])
            results.append({"label": label, **result})
        return {
            "ok": all(item["ok"] for item in results),
            "path": path.as_posix(),
            "html_parsed": html_parsed,
            "checks": results,
            "limits": {"network": False, "timeout_seconds": RUN_TIMEOUT_SECONDS, "memory_mb": 128},
        }
    finally:
        shutil.rmtree(run_root, ignore_errors=True)

"""Isolated workspace and allow-listed test runner for coding agents."""

from __future__ import annotations

import os
import pwd
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


MAX_FILES = 20
MAX_FILE_BYTES = 200_000
MAX_TOTAL_BYTES = 1_000_000
MAX_TEST_COMMANDS = 3
TEST_TIMEOUT_SECONDS = 25
WORKSPACE_TTL_SECONDS = 60 * 60
ALLOWED_INTERPRETERS = {"python": "python3", "python3": "python3", "node": "node"}
ALLOWED_PYTHON_MODULES = {"unittest"}


class SandboxError(ValueError):
    pass


def workspace_root(data_dir: Path | None = None) -> Path:
    # Keep runs in world-traversable /tmp so tests can drop to `nobody`
    # without opening the private chat data directory.
    root = Path("/tmp/deepseek-code-runs")
    root.mkdir(parents=True, exist_ok=True, mode=0o1777)
    try:
        root.chmod(0o1777)
    except OSError:
        pass
    return root


def create_workspace(data_dir: Path, run_id: str) -> Path:
    path = workspace_root(data_dir) / "".join(ch for ch in run_id if ch.isalnum())[:32]
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True, mode=0o777)
    return path


def cleanup_workspace(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def cleanup_old_workspaces(data_dir: Path, max_age_seconds: int = WORKSPACE_TTL_SECONDS) -> None:
    root = workspace_root(data_dir)
    cutoff = time.time() - max(60, int(max_age_seconds))
    for item in root.iterdir():
        try:
            if item.is_dir() and item.stat().st_mtime < cutoff:
                shutil.rmtree(item, ignore_errors=True)
        except OSError:
            continue


def safe_relpath(value: str) -> Path:
    raw = str(value or "").replace("\\", "/").strip()
    if not raw or raw.startswith("/") or ".." in Path(raw).parts:
        raise SandboxError(f"非法文件路径：{value}")
    path = Path(raw)
    if path.is_absolute() or not path.parts:
        raise SandboxError(f"非法文件路径：{value}")
    return path


def write_files(workspace: Path, files: list[dict[str, Any]]) -> list[str]:
    if not isinstance(files, list) or not files:
        raise SandboxError("至少需要提交一个文件")
    if len(files) > MAX_FILES:
        raise SandboxError(f"一次最多提交 {MAX_FILES} 个文件")
    written: list[str] = []
    total = 0
    for item in files:
        if not isinstance(item, dict):
            raise SandboxError("文件项必须是对象")
        rel = safe_relpath(str(item.get("path") or ""))
        content = str(item.get("content") or "")
        encoded = content.encode("utf-8")
        if len(encoded) > MAX_FILE_BYTES:
            raise SandboxError(f"{rel} 超过单文件大小限制")
        total += len(encoded)
        if total > MAX_TOTAL_BYTES:
            raise SandboxError("提交文件总体积过大")
        destination = workspace / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        try:
            destination.chmod(0o666)
        except OSError:
            pass
        written.append(str(rel))
    return written


def list_files(workspace: Path) -> list[str]:
    files: list[str] = []
    for item in sorted(workspace.rglob("*")):
        if item.is_file():
            files.append(str(item.relative_to(workspace)))
    return files


def read_workspace_file(workspace: Path, relative: str, limit: int = 12_000) -> str:
    path = workspace / safe_relpath(relative)
    if not path.is_file():
        raise SandboxError(f"文件不存在：{relative}")
    return path.read_text(encoding="utf-8", errors="replace")[:limit]


def _nobody() -> pwd.struct_passwd | None:
    try:
        return pwd.getpwnam("nobody")
    except KeyError:
        return None


def validate_test_command(command: str, workspace: Path) -> list[str]:
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        raise SandboxError(f"无法解析测试命令：{exc}") from exc
    if not parts:
        raise SandboxError("测试命令不能为空")
    binary_name = Path(parts[0]).name
    if binary_name not in ALLOWED_INTERPRETERS:
        raise SandboxError("测试命令只允许 python3 或 node")
    resolved = shutil.which(ALLOWED_INTERPRETERS[binary_name])
    if not resolved:
        raise SandboxError(f"服务器没有 {ALLOWED_INTERPRETERS[binary_name]}")
    args = parts[1:]
    if any(flag in args for flag in ("-c", "-e", "--eval", "--")):
        raise SandboxError("不允许内联代码或额外解释器开关")
    if "-m" in args:
        index = args.index("-m")
        module = args[index + 1] if index + 1 < len(args) else ""
        if module not in ALLOWED_PYTHON_MODULES:
            raise SandboxError("python -m 仅允许 unittest")
    root = workspace.resolve()
    for arg in args:
        if arg.startswith("-") or arg in {"unittest", "discover"}:
            continue
        if "/" not in arg and "." not in arg:
            continue
        candidate = (workspace / arg).resolve()
        if candidate != root and root not in candidate.parents:
            raise SandboxError(f"测试路径超出工作区：{arg}")
    return [resolved, *args]


def _drop_privileges(user: pwd.struct_passwd) -> None:
    os.environ["HOME"] = "/tmp"
    os.environ["TMPDIR"] = "/tmp"
    os.setgid(user.pw_gid)
    os.setuid(user.pw_uid)


def run_test_command(command: str, workspace: Path) -> dict[str, Any]:
    argv = validate_test_command(command, workspace)
    try:
        workspace.chmod(0o777)
        for item in workspace.rglob("*"):
            item.chmod(0o777 if item.is_dir() else 0o666)
    except OSError:
        pass
    limiter = shutil.which("prlimit")
    launch = argv
    if limiter:
        launch = [limiter, "--as=100663296", "--cpu=20", "--nproc=64", "--fsize=2097152", "--", *argv]
    nobody = _nobody()
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    preexec = (lambda: _drop_privileges(nobody)) if nobody and os.geteuid() == 0 else None
    try:
        completed = subprocess.run(
            launch,
            cwd=str(workspace),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=TEST_TIMEOUT_SECONDS,
            check=False,
            preexec_fn=preexec,
        )
    except subprocess.TimeoutExpired:
        return {
            "command": command,
            "ok": False,
            "exit_code": 124,
            "stdout": "",
            "stderr": f"测试超过 {TEST_TIMEOUT_SECONDS} 秒，已停止",
        }
    stdout = completed.stdout.decode("utf-8", errors="replace")[-4000:]
    stderr = completed.stderr.decode("utf-8", errors="replace")[-4000:]
    return {
        "command": command,
        "ok": completed.returncode == 0,
        "exit_code": int(completed.returncode),
        "stdout": stdout,
        "stderr": stderr,
    }


def run_test_commands(commands: list[str], workspace: Path) -> list[dict[str, Any]]:
    if not isinstance(commands, list) or not commands:
        raise SandboxError("审查必须提供至少一条可执行测试命令")
    if len(commands) > MAX_TEST_COMMANDS:
        raise SandboxError(f"一次最多运行 {MAX_TEST_COMMANDS} 条测试命令")
    return [run_test_command(str(item), workspace) for item in commands]

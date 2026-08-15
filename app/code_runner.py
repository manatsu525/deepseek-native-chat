from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


RUN_ROOT = Path("/run/deepseek-code-runs")
MAX_ARGUMENTS = 20
MAX_ARGUMENT_CHARS = 1000
MAX_OUTPUT_CHARS = 12_000
RUN_TIMEOUT_SECONDS = 12


class CodeRunnerError(ValueError):
    pass


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


def run_python(source_root: Path, relative_path: str, arguments: Any = None) -> dict[str, Any]:
    path = Path(str(relative_path or ""))
    if not str(relative_path or "").strip() or "\\" in str(relative_path) or path.is_absolute() or ".." in path.parts:
        raise CodeRunnerError("Python 文件路径无效")
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
        command = [
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
            "/usr/bin/python3",
            path.as_posix(),
            *args,
        ]
        try:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=RUN_TIMEOUT_SECONDS + 8,
                check=False,
                env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
            )
        except subprocess.TimeoutExpired as exc:
            raise CodeRunnerError("代码运行超时，已停止") from exc
        stdout = completed.stdout.decode("utf-8", errors="replace")[-MAX_OUTPUT_CHARS:]
        stderr = completed.stderr.decode("utf-8", errors="replace")[-MAX_OUTPUT_CHARS:]
        return {
            "ok": completed.returncode == 0,
            "path": path.as_posix(),
            "exit_code": int(completed.returncode),
            "stdout": stdout,
            "stderr": stderr,
            "limits": {"network": False, "timeout_seconds": RUN_TIMEOUT_SECONDS, "memory_mb": 128},
        }
    finally:
        shutil.rmtree(run_root, ignore_errors=True)

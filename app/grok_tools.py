"""Source-derived Grok Build text tools and web overflow, with application adapters.

Upstream 2bdd1d6, Apache-2.0; see third_party/grok-build/NOTICE.
"""
from __future__ import annotations

import hashlib
import fcntl
import json
import logging
import os
import re
import signal
import subprocess
import threading
import time
import uuid
import tempfile
from pathlib import Path
from typing import Any

READ_MAX_BYTES = 100_000
READ_MAX_LINES = 1_000
WEB_MAX_BYTES = 100_000
WEB_DOWNLOAD_BYTES = 10 * 1024 * 1024


def clean_web_content(content: str) -> str:
    """Grok fetch post-processing: enforce raw byte cap, strip base64 data URIs."""
    if len(content.encode()) > WEB_DOWNLOAD_BYTES:
        raise ValueError("Web content exceeds the 10MiB download limit")
    def remove(match: re.Match) -> str:
        header = match[1]
        parts = header.split(";")
        if len(header.encode()) > 120 or not any(part.casefold() == "base64" for part in parts[1:]):
            return match[0]
        return "[base64 " + (parts[0] or "unknown") + " data removed]"
    return re.sub(r"(?<![A-Za-z0-9])data:([^,\t\n\v\f\r ]{0,120}),([A-Za-z0-9+/=]{4,})", remove, content)


def utf8_prefix(text: str, size: int) -> str:
    return text.encode("utf-8")[:max(0, size)].decode("utf-8", errors="ignore")


def read_window(content: str, offset: Any = None, limit: Any = None) -> dict[str, Any]:
    start = max(1, int(offset or 1))
    count = READ_MAX_LINES if limit is None else max(0, int(limit))
    lines = content.splitlines()
    selected = lines[start - 1:start - 1 + count]
    result: list[str] = []
    size = 0
    for n, line in enumerate(selected, start):
        rendered = f"{n}→{line}" if n == start or n % 10 == 0 else line
        extra = len(rendered.encode()) + int(bool(result))
        if result and size + extra > READ_MAX_BYTES:
            break
        result.append(utf8_prefix(rendered, READ_MAX_BYTES) if not result else rendered)
        size += extra
        if size >= READ_MAX_BYTES:
            break
    through = start - 1 + len(result)
    truncated = through < len(lines)
    text = "\n".join(result)
    if truncated:
        text += f"\n[File content truncated. Continue reading with offset={through + 1} and limit.]"
    return {"content": text, "line_count": len(lines), "from_line": start,
            "through_line": through, "truncated": truncated,
            "next_start_line": through + 1 if truncated else None}


def replace_string(content: str, old: str, new: str, replace_all: bool = False) -> str:
    if not old:
        return new
    has_crlf = "\r\n" in content
    match_text = content.replace("\r\n", "\n") if has_crlf else content
    old = old.replace("\r\n", "\n") if has_crlf else old
    count = match_text.count(old)
    if not count:
        raise ValueError("old_string was not found in the file")
    if count > 1 and not replace_all:
        raise ValueError(f"old_string matches {count} locations; provide unique context or replace_all=true")
    updated = match_text.replace(old, new, -1 if replace_all else 1)
    return updated.replace("\r\n", "\n").replace("\n", "\r\n") if has_crlf else updated


def function(name: str, description: str, properties: dict, required: list) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required,
                           "additionalProperties": False}}}


READ_TOOL = function("read_file", "Read a UTF-8 text file. Defaults to 1000 lines. Line anchors appear on the first returned line and every 10th file line. Use offset and limit for large files.", {
    "target_file": {"type": "string", "description": "The file path"},
    "offset": {"type": "integer", "description": "1-based starting line", "default": 1},
    "limit": {"type": "integer", "description": "Number of lines to read"}}, ["target_file"])
EDIT_TOOL = function("search_replace", "Replace an exact string in a file. old_string must be unique unless replace_all is true. An empty old_string creates or replaces the whole file.", {
    "file_path": {"type": "string"}, "old_string": {"type": "string"},
    "new_string": {"type": "string"}, "replace_all": {"type": "boolean", "default": False}},
    ["file_path", "old_string", "new_string"])
LIST_TOOL = function("list_dir", "List the contents of a directory.", {
    "target_directory": {"type": "string"}}, ["target_directory"])
GREP_TOOL = function("grep", "Search file contents with ripgrep regular expressions. Respect ignore files; use a focused path and glob to narrow the search.", {
    "pattern": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"},
    "-A": {"type": "integer"}, "-B": {"type": "integer"}, "-C": {"type": "integer"},
    "-i": {"type": "boolean"}, "type": {"type": "string"}, "multiline": {"type": "boolean"},
    "head_limit": {"type": "integer"}, "offset": {"type": "integer"}}, ["pattern"])
SKILL_TOOL = function("skill", "Execute a skill by name. Loads its SKILL.md instructions into context. Optional args replace $ARGUMENTS, $ARGUMENTS[N] and zero-based $0, $1, ... variables.", {
    "skill": {"type": "string"}, "args": {"type": "string"}}, ["skill"])

BASH_TOOL = function("bash", "Run a bash command. block_until_ms defaults to 30000; 0 starts in the background immediately. A running command returns a task_id; use get_task_output to wait and inspect it, kill_task to stop it. Full output is saved locally.", {
    "command": {"type": "string"}, "description": {"type": "string"},
    "block_until_ms": {"type": "integer", "default": 30_000}}, ["command", "description"])
TASK_OUTPUT_TOOL = function("get_task_output", "Read output from one or more background bash tasks. Positive timeout_ms waits for all tasks; a wait timeout leaves them running.", {
    "task_ids": {"type": "array", "items": {"type": "string"}},
    "timeout_ms": {"type": "integer", "maximum": 3_600_000}}, ["task_ids"])
KILL_TASK_TOOL = function("kill_task", "Stop a background bash task owned by this conversation.", {
    "task_id": {"type": "string"}}, ["task_id"])


def grep_content(target: Path, arguments: dict[str, Any], *, cwd: Path) -> dict:
    args = ["rg", "--no-heading", "--line-number", "--color=never"]
    for flag in ("-A", "-B", "-C"):
        if arguments.get(flag) is not None:
            args.extend([flag, str(max(0, int(arguments[flag])))])
    if arguments.get("-i"):
        args.append("-i")
    if arguments.get("multiline"):
        args.extend(["--multiline", "--multiline-dotall"])
    for key, flag in (("glob", "--glob"), ("type", "--type")):
        if arguments.get(key):
            args.extend([flag, str(arguments[key])])
    args.extend(["--regexp", str(arguments.get("pattern") or ""), "--", str(target)])
    limit = min(2000, max(0, int(arguments.get("head_limit", 200))))
    # Keep rg's normal ignore semantics; do not turn patterns into literal searches.
    with subprocess.Popen(args, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, errors="replace") as process:
        output = []
        size = 0
        truncated = False
        try:
            for line in process.stdout:
                if len(output) >= limit or size + len(line.encode()) > 40_000:
                    truncated = True
                    process.terminate()
                    break
                output.append(line)
                size += len(line.encode())
            process.wait(timeout=30)
            error = process.stderr.read()
            if process.returncode not in (0, 1) and not truncated:
                raise ValueError(error.strip() or "ripgrep failed")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    return {"content": "".join(output) or "No matches found.", "truncated": truncated}


def list_directory(target: Path) -> dict:
    if not target.is_dir():
        raise ValueError("Directory not found")
    queue = [target]
    output = []
    size = 0
    while queue:
        current = queue.pop(0)
        for child in sorted(current.iterdir(), key=lambda p: p.name):
            rendered = str(child.relative_to(target)) + ("/" if child.is_dir() else "")
            if size + len(rendered.encode()) + 1 > 10_000:
                return {"content": "\n".join(output), "truncated": True}
            output.append(rendered)
            size += len(rendered.encode()) + 1
            if child.is_dir() and not child.is_symlink() and child.name not in {".git", "node_modules", ".venv", "__pycache__"}:
                queue.append(child)
    return {"content": "\n".join(output), "truncated": False}


class BashTasks:
    """Conversation-owned host processes with recoverable logs.

    The process table is shared by runtimes so a follow-up turn can inspect a task.
    Its account/conversation key prevents cross-account task access.
    """
    _tasks: dict[tuple[int, str, str], dict] = {}
    _lock = threading.RLock()

    def __init__(self, owner: tuple[int, str], directory: Path, cancelled: threading.Event):
        self.owner, self.directory, self.cancelled = owner, directory, cancelled

    def run(self, arguments: dict, cwd: Path) -> dict:
        if self.cancelled.is_set():
            return {"ok": False, "cancelled": True, "error": "Task stopped"}
        command = str(arguments.get("command") or "")
        if not command.strip():
            raise ValueError("command must not be empty")
        task_id = uuid.uuid4().hex[:12]
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.directory / (task_id + ".log")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as log:
            process = subprocess.Popen(["/bin/bash", "-lc", command], cwd=str(cwd),
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       start_new_session=True)
        key = (*self.owner, task_id)
        with self._lock:
            self._tasks[key] = {"process": process, "path": path, "command": command}
        if self.cancelled.is_set():
            return self.kill({"task_id": task_id})
        def reap():
            try:
                process.wait(timeout=24 * 3600)
            except subprocess.TimeoutExpired:
                self.kill({"task_id": task_id})
        threading.Thread(target=reap, daemon=True).start()
        wait = min(36_000_000, max(0, int(arguments.get("block_until_ms", 30_000))))
        return self.output({"task_id": task_id, "timeout_ms": wait}, max_wait=36_000_000)

    def get_output(self, arguments: dict) -> dict:
        ids = arguments.get("task_ids", arguments.get("task_id"))
        if isinstance(ids, (str, int)):
            ids = [str(ids)]
        if not isinstance(ids, list) or not ids:
            raise ValueError("Provide a non-empty task_ids list")
        ids = list(dict.fromkeys(str(value).strip() for value in ids if str(value).strip()))
        deadline = time.monotonic() + min(3_600_000, max(0, int(arguments.get("timeout_ms") or 0))) / 1000
        results = [self.output({"task_id": task_id, "timeout_ms": max(0, int((deadline - time.monotonic()) * 1000))}) for task_id in ids]
        return results[0] if len(results) == 1 else {"results": results}

    def output(self, arguments: dict, *, max_wait: int = 3_600_000) -> dict:
        key = (*self.owner, str(arguments.get("task_id") or ""))
        with self._lock:
            task = self._tasks.get(key)
        if task is None:
            raise ValueError("Task not found in this conversation")
        process = task["process"]
        deadline = time.monotonic() + min(max_wait, max(0, int(arguments.get("timeout_ms", 0)))) / 1000
        while process.poll() is None and time.monotonic() < deadline:
            if self.cancelled.wait(min(.1, max(0, deadline - time.monotonic()))):
                self.kill({"task_id": key[-1]})
                break
        with task["path"].open("rb") as stream:
            head = stream.read(20_000)
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            if size > 40_000:
                stream.seek(-20_000, os.SEEK_END)
                output = head.decode(errors="replace") + "\n[…output truncated…]\n" + stream.read().decode(errors="replace")
            else:
                stream.seek(0)
                output = stream.read().decode(errors="replace")
        code = process.poll()
        return {"ok": code in (None, 0), "task_id": key[-1], "status": "running" if code is None else "completed",
                "exit_code": code, "stdout": output, "output_file": str(task["path"])}

    def kill(self, arguments: dict) -> dict:
        with self._lock:
            task = self._tasks.get((*self.owner, str(arguments.get("task_id") or "")))
        if task is None:
            raise ValueError("Task not found in this conversation")
        process = task["process"]
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=1)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        return {"ok": True, "task_id": arguments["task_id"], "exit_code": process.poll()}

    def cancel_owned(self) -> None:
        with self._lock:
            ids = [key[-1] for key in self._tasks if key[:2] == self.owner]
        for task_id in ids:
            self.kill({"task_id": task_id})


def save_web_artifact(directory: Path, content: str, extension: str) -> Path:
    """Upstream allocation file, monotonic numbers, locked 1GiB budget, atomic save."""
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    fd = os.open(directory / ".allocation", os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(fd, "r+b") as allocation:
        fcntl.flock(allocation, fcntl.LOCK_EX)
        files = [path for path in directory.iterdir() if not path.name.startswith(".") and path.is_file()]
        last = max((int(path.stem) for path in files if path.stem.isdecimal()), default=0)
        allocation.seek(0)
        stored = allocation.read()
        reserved = int(stored) if len(stored) == 10 and stored.isdigit() else last
        number = max(last, reserved) + 1
        allocation.seek(0)
        allocation.write(f"{number:010d}".encode())
        allocation.truncate(10)
        allocation.flush()
        os.fsync(allocation.fileno())
        data = content.encode()
        if sum(path.stat().st_size for path in files) + len(data) > 1024 * 1024 * 1024:
            raise ValueError("web_fetch artifact byte budget exceeded (1GiB)")
        destination = directory / f"{number}.{extension}"
        fd, temporary = tempfile.mkstemp(prefix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return destination


def web_preview(content: str, *, root: Path, conversation_id: str,
                context_window: int = 128_000, ordinary: bool = False,
                content_type: str = "text/markdown") -> str:
    """Grok web_fetch: 3% context preview, 100K byte output, exact full artifact.

    Ordinary mode exposes the private artifact through its read-only .context mount.
    Agent mode uses the absolute session path. No second upstream request is needed.
    """
    preview_bytes = min(int(context_window * 4 * .03), WEB_MAX_BYTES)
    total = len(content.encode())
    if total <= preview_bytes:
        return content
    directory = root / hashlib.sha256(conversation_id.encode()).hexdigest() / "web_fetch"
    extension = "md" if content_type in {"markdown", "text/markdown"} else "txt"
    is_json = False
    try:
        json.loads(content)
        is_json = True
        if extension != "md":
            extension = "json"
    except (ValueError, TypeError):
        pass
    try:
        path = save_web_artifact(directory, content, extension)
    except (OSError, ValueError):
        logging.getLogger(__name__).warning("Unable to save full web_fetch artifact", exc_info=True)
        path = None
    display = (".context/web_fetch/" + path.name if ordinary else str(path)) if path else ""
    if (is_json and extension == "json") or any(len(line.encode()) > 2000 for line in content.splitlines()):
        steer = " Use `bash` to query, slice or search the saved content, for example with python3."
    else:
        steer = " Use `read_file` with offsets and limits to read it in chunks."
    def footer(shown: int) -> str:
        return (f"\n\n[web_fetch content truncated: showing first {shown} of {total} bytes."
                + (f" Full content saved to: {display}.{steer}" if path else "") + "]")
    budget = min(preview_bytes, WEB_MAX_BYTES - len(footer(preview_bytes).encode()))
    preview = utf8_prefix(content, budget)
    return preview + footer(len(preview.encode()))

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
import selectors
from collections import Counter, deque
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


def bool_arg(value: Any, default: bool = False) -> bool:
    """Upstream's lenient JSON bool arguments, including string booleans."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "false", "0", "1"}:
        return value.lower() in {"true", "1"}
    raise ValueError("Expected a boolean")


def read_window(content: str, offset: Any = None, limit: Any = None, *, path: str = "") -> dict[str, Any]:
    start = max(1, int(offset or 1))
    count = min(READ_MAX_LINES, READ_MAX_LINES if limit is None else max(0, int(limit)))
    lines = content.replace("\r\n", "\n").split("\n") if content else []
    # Source whole-read policy for instruction and skill markdown, when it fits.
    if Path(path).name in {"SKILL.md", "AGENTS.md", "CLAUDE.md"}:
        full = "\n".join(f"{n}→{line}" if n == 1 or n % 10 == 0 else line for n, line in enumerate(lines, 1))
        if len(full.encode()) <= READ_MAX_BYTES:
            start, count = 1, len(lines)
    selected = lines[start - 1:start - 1 + count]
    result: list[str] = []
    for n, line in enumerate(selected, start):
        rendered = f"{n}→{line}" if n == start or n % 10 == 0 else line
        result.append(rendered)
    through = start - 1 + len(result)
    truncated = through < len(lines)
    text = "\n".join(result)
    if len(text.encode()) > READ_MAX_BYTES:
        tokens = (len(text.encode()) + 3) // 4
        hint = "\nNote: the requested read is a single very long line, so line-based offset/limit cannot narrow it further. Use the 'bash' tool to extract the parts you need (e.g. `jq`, `python3`, or `cut -c`)." if len(selected) <= 1 else ""
        raise ValueError(f"File content ({tokens} tokens) exceeds maximum allowed tokens (25000 tokens).\nPlease use offset and limit parameters to read a shorter range, or use the 'grep' to search for specific content." + hint)
    return {"content": text, "line_count": len(lines), "from_line": start,
            "through_line": through, "truncated": truncated,
            "next_start_line": through + 1 if truncated else None}


def replace_string(content: str, old: str, new: str, replace_all: bool = False) -> str:
    if old == new:
        raise ValueError("Old string and new string are the same")
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


READ_TOOL = function("read_file", """Read a file.

Usage:
- The target_file parameter can be a relative path in the workspace or an absolute path
- By default, it reads up to 1000 lines starting from the beginning of the file (SKILL.md and AGENTS.md/CLAUDE.md files are always returned whole; offset and limit are ignored for them)
- Line numbers (1-based) appear as anchors in the format LINE_NUMBER→LINE_CONTENT on the first returned line and on every 10th line of the file; the lines in between show content only. Count from the nearest anchor when referring to a specific line
- PDF, PowerPoint and Jupyter notebook files can be read as extracted text. Image viewing is not available through this adapter.""", {
    "target_file": {"type": "string", "description": "The path of the file to read. You can use either a relative path in the workspace or an absolute path. If an absolute path is provided, it will be preserved as is."},
    "offset": {"type": "integer", "description": "The line number to start reading from. Only provide if the file is too large to read at once.", "default": 1},
    "limit": {"type": "integer", "description": "The number of lines to read. Only provide if the file is too large to read at once."},
    "pages": {"type": "string", "description": "Page range for PDF files (e.g. '1-5', '3', '10-'). Required for PDFs with more than 10 pages. Max 20 pages per call. Ignored for non-PDF files."},
    "format": {"type": "string", "description": "PDF output format. This adapter supports text extraction."}}, ["target_file"])
EDIT_TOOL = function("search_replace", """Replace an exact string in a file.

- `read_file` prefixes each line with "LINE_NUMBER→". That prefix is not part of the file: match only what comes after the →, with its exact indentation.
- `old_string` must match exactly one place in the file. If it appears more than once, add surrounding lines to make it unique, or set `replace_all` to change every occurrence (handy for renaming an identifier).
- To create a new file, set `old_string` to an empty string.""", {
    "file_path": {"type": "string", "description": "The path to the file to modify. You can use either a relative path in the workspace or an absolute path."}, "old_string": {"type": "string", "description": "The text to replace"},
    "new_string": {"type": "string", "description": "The text to replace it with (must be different from old_string)"}, "replace_all": {"type": "boolean", "description": "Replace all occurrences of old_string (default false)", "default": False}},
    ["file_path", "old_string", "new_string"])
LIST_TOOL = function("list_dir", "List the contents of a directory.", {
    "target_directory": {"type": "string"}}, ["target_directory"])
GREP_TOOL = function("grep", """Search file contents with regular expressions (ripgrep).

- Full regex syntax, so escape literal special characters: `functionCall\\(`, or `interface\\{\\}` to find interface{} in Go.
- Pass pattern as a raw regex string — no surrounding quotes.
- Respects .gitignore unless you pass a broad glob like '--glob *'.
- Only filter by 'type' or 'glob' when you are sure of the file type; import paths may not match source file types (.js vs .ts).
- Output is ripgrep-style: ':' marks match lines, '-' marks context lines, grouped by file. Large results are capped and report "at least" counts.""", {
    "pattern": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"},
    "-A": {"type": "integer"}, "-B": {"type": "integer"}, "-C": {"type": "integer"},
    "-i": {"type": "boolean"}, "type": {"type": "string"}, "multiline": {"type": "boolean"},
    "head_limit": {"type": "integer", "description": "Limit output to first N lines/entries, equivalent to | head -N. Defaults to 200 lines or 500 entries."}}, ["pattern"])
SKILL_TOOL = function("skill", "Execute a skill by name. Loads its SKILL.md instructions into context. Optional args replace $ARGUMENTS, $ARGUMENTS[N] and zero-based $0, $1, ... variables.", {
    "skill": {"type": "string"}, "args": {"type": "string"}}, ["skill"])

BASH_TOOL = function("bash", """Run a bash command and return its output.

Usage notes:
  - You can specify an optional block_until_ms in milliseconds (up to 36000000ms). A foreground command still running at block_until_ms is moved to the background instead of killed; once backgrounded it runs until it exits (background cap 24h). You will receive a task id; wait for it with get_task_output. `block_until_ms: 0` runs the command in the background immediately.
  - Background commands run until they exit, until you stop them with kill_task, or until the 24h background cap. kill_task sends SIGTERM to the process group, then SIGKILL after ~1s; processes that did not detach via setsid / nohup are killed with it.
  - If the output exceeds 40000 characters, the middle is truncated (you keep the beginning and end) and the result includes the path to a log file with the full output, which you can read or search.
  - Set `block_until_ms` to 0 to run the command in the background (e.g., dev servers, long builds): it returns a task id immediately and keeps running in the background. Use get_task_output to monitor it or wait for it to finish. You do not need to use '&' at the end of the command when using this parameter.""", {
    "command": {"type": "string"}, "description": {"type": "string"},
    "block_until_ms": {"type": "integer", "default": 30_000}}, ["command", "description"])
TASK_OUTPUT_TOOL = function("get_task_output", "Read output from one or more background bash tasks. Positive timeout_ms waits for all tasks; a wait timeout leaves them running.", {
    "task_ids": {"type": "array", "items": {"type": "string"}},
    "timeout_ms": {"type": "integer", "maximum": 3_600_000}}, ["task_ids"])
KILL_TASK_TOOL = function("kill_task", "Stop a background bash task owned by this conversation.", {
    "task_id": {"type": "string"}}, ["task_id"])


def grep_content(target: Path, arguments: dict[str, Any], *, cwd: Path, display_cwd: str = "") -> dict:
    mode = arguments.get("output_mode", "content")
    if mode not in {"content", "files_with_matches", "count"}:
        raise ValueError("Invalid output_mode")
    args = ["rg", "--heading", "--line-number", "--color=never"] if mode == "content" else ["rg", "--files-with-matches" if mode == "files_with_matches" else "--count", "--color=never"]
    for flag in ("-A", "-B", "-C"):
        if arguments.get(flag) is not None:
            args.extend([flag, str(max(0, int(arguments[flag])))])
    if bool_arg(arguments.get("-i")):
        args.append("-i")
    if bool_arg(arguments.get("multiline")):
        args.extend(["--multiline", "--multiline-dotall"])
    for key, flag in (("glob", "--glob"), ("type", "--type")):
        if arguments.get(key):
            args.extend([flag, str(arguments[key])])
    args.extend(["--regexp", str(arguments.get("pattern") or ""), "--", str(target)])
    limit = min(2000 if mode == "content" else 10000,
                max(0, int(arguments.get("head_limit", 200 if mode == "content" else 500))))
    # Keep rg's normal ignore semantics; do not turn patterns into literal searches.
    with tempfile.TemporaryFile() as stderr, subprocess.Popen(args, cwd=str(cwd), stdout=subprocess.PIPE, stderr=stderr) as process:
        output = []
        size = 0
        truncated = False
        try:
            deadline = time.monotonic() + 20
            pending = b""
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ValueError("Search timed out after 20 seconds")
                    if not selector.select(remaining):
                        raise ValueError("Search timed out after 20 seconds")
                    chunk = os.read(process.stdout.fileno(), 8192)
                    pending += chunk
                    lines = pending.split(b"\n")
                    pending = lines.pop()
                    if not chunk and pending:
                        lines.append(pending)
                        pending = b""
                    for raw in lines:
                        line = raw.decode(errors="replace")
                        if display_cwd and line.startswith(str(cwd) + "/"):
                            line = display_cwd.rstrip("/") + line[len(str(cwd)):]
                        # Source per-line truncation and bounded streamed output.
                        line = line[:1000] + ("..." if len(line) > 1000 else "")
                        if len(output) >= limit or size + len(line.encode()) + 1 > 40_000:
                            truncated = True
                            break
                        output.append(line)
                        size += len(line.encode()) + 1
                    if truncated:
                        process.terminate()
                        break
                    if not chunk:
                        break
            process.wait(timeout=max(.1, deadline - time.monotonic()))
            stderr.seek(0)
            error = stderr.read().decode(errors="replace")
            if process.returncode not in (0, 1) and not truncated:
                raise ValueError(error.strip() or "ripgrep failed")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    at_least = "at least " if truncated else ""
    if not output:
        body = "No matches found"
    else:
        if mode == "content":
            count = sum(bool(re.match(r"^\d+:", line)) for line in output)
            summary = f"Found {at_least}{count} matching lines"
        elif mode == "files_with_matches":
            summary = f"Found {at_least}{len(output)} files"
        else:
            count = sum(int(line.rsplit(":", 1)[-1]) for line in output if line.rsplit(":", 1)[-1].isdigit())
            summary = f"Found {count} across {at_least}{len(output)} files"
        body = summary + "\n" + "\n".join(output)
    return {"content": f'<workspace_result workspace_path="{display_cwd or cwd}">\n{body}\n</workspace_result>', "truncated": truncated}


def list_directory(target: Path, *, display_path: str = "") -> dict:
    if not target.is_dir():
        raise ValueError("Directory not found")
    class Node:
        def __init__(self, depth=0):
            self.depth, self.files, self.children, self.extensions = depth, [], {}, Counter()
            self.expanded = False
        def add(self, parts, directory=False):
            name, *rest = parts
            if not directory:
                self.extensions[Path(parts[-1]).suffix.lstrip(".").lower() or "no-ext"] += 1
            if rest or directory:
                child = self.children.setdefault(name + "/", Node(self.depth + 1))
                if rest:
                    child.add(rest, directory)
            elif name not in self.files:
                self.files.append(name)
        def summary(self):
            total = sum(self.extensions.values())
            if not total:
                return ""
            top = sorted(self.extensions.items(), key=lambda pair: (-pair[1], pair[0]))[:3]
            parts = [f"{count} *{'.' + ext if ext != 'no-ext' else 'no-ext'}" for ext, count in top]
            return f"[{total} {'file' if total == 1 else 'files'} in subtree: {', '.join(parts)}{', ...' if sum(n for _, n in top) < total else ''}]"
        def render(self):
            if not self.expanded:
                summary = self.summary()
                return "  " * (self.depth + 1) + summary + "\n" if summary else ""
            return "".join("  " * (self.depth + 1) + "- " + name + "\n" + (self.children[name].render() if name in self.children else "")
                           for name in sorted([*self.files, *self.children], key=str.lower))
    tree = Node()
    # rg uses the same ignore library as upstream's WalkBuilder. Seed direct
    # siblings first so a large subtree cannot hide later root entries.
    children = [p for p in target.iterdir() if not p.name.startswith(".")]
    ignored = subprocess.run(["git", "check-ignore", "--stdin", "-z"], cwd=target,
                             input=b"\0".join(os.fsencode(p.name) for p in children) + b"\0",
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False).stdout.split(b"\0")
    ignored = set(os.fsdecode(p) for p in ignored if p)
    truncated = len(children) > 100000
    for child in children[:100000]:
        if child.name not in ignored:
            if child.is_dir() and not child.is_symlink():
                tree.add([child.name], True)
    # Files are streamed, bounded by the upstream deep-walk item budget.
    with tempfile.TemporaryFile() as stderr, subprocess.Popen(["rg", "--files", "--color=never", "."], cwd=target, stdout=subprocess.PIPE, stderr=stderr) as process:
        try:
            for n, line in enumerate(process.stdout):
                if n >= 100000:
                    truncated = True
                    process.terminate()
                    break
                relative = Path(os.fsdecode(line.rstrip(b"\n")))
                tree.add(list(relative.parts))
            process.wait(timeout=20)
            if process.returncode not in (0, 1) and not truncated:
                stderr.seek(0)
                raise ValueError(stderr.read().decode(errors="replace") or "Directory walk failed")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    tree.expanded = True
    notice = "    ...\n\n    Note: this directory is too large to list fully. Try list_dir on a narrower path, or use grep / bash."
    body = tree.render()
    if len(body.encode()) > 10000:
        chunks = []
        size = 0
        for name in sorted([*tree.files, *tree.children], key=str.lower):
            chunk = "  - " + name + "\n" + (tree.children[name].render() if name in tree.children else "")
            if size + len(chunk.encode()) > 10000:
                break
            chunks.append(chunk)
            size += len(chunk.encode())
        body = "".join(chunks) + notice
        truncated = True
    else:
        remaining = 10000 - len(body.encode())
        queue = deque(tree.children.values())
        while queue:
            node = queue.popleft()
            summary_size = len(node.render().encode())
            node.expanded = True
            expanded_size = len(node.render().encode())
            if expanded_size > remaining + summary_size:
                node.expanded = False
                continue
            remaining += summary_size - expanded_size
            queue.extend(node.children[name] for name in sorted(node.children, key=str.lower))
        body = tree.render()
    if truncated:
        body += "\nNote: there are more than 100000 items in the directory, so not all files may be shown.\n" if not body.endswith(notice) else ""
    return {"content": f"- {(display_path or str(target)).rstrip('/')}/\n{body.rstrip()}", "truncated": truncated}


def model_tool_output(name: str, result: dict) -> str:
    """ToolOutput::to_prompt_format; transport/UI metadata stays out of context."""
    if name in {"read_file", "list_dir", "grep"}:
        content = result.get("content", "")
        if name == "read_file" and not content:
            if not result.get("line_count"):
                return "File is empty."
            if result.get("from_line", 1) > result["line_count"]:
                return f"(no lines returned: the requested window is past the end of the file; the file has {result['line_count']} lines)"
            return "(no lines returned)"
        return content
    if name == "search_replace":
        return result["message"]
    if name == "bash":
        if result.get("status") == "running":
            return (f"<task-id>{result['task_id']}</task-id>\n<task-type>local_bash</task-type>\n"
                    f"<output-file>{result['output_file']}</output-file>\n<status>running</status>\n"
                    f"<summary>Command is still running and has been moved to the background as task {result['task_id']}.</summary>\n"
                    "Use get_task_output to monitor it or wait for it to finish.")
        annotation = f" [truncated: full output at: {result['output_file']}]" if result.get("truncated") else ""
        return f"exit: {result['exit_code']}{annotation}\n{result.get('stdout') or '(no output)'}"
    if name == "get_task_output":
        results = result.get("results", [result])
        return "\n\n".join(f"=== Task {r['task_id']} ===\nCommand: {r.get('command', '')}\nStatus: {r['status']}\nDuration: {r.get('duration_secs', 0):.2f}s\n"
                           + (f"Exit Code: {r['exit_code']}\n" if r.get("exit_code") is not None else "")
                           + f"Output File: {r['output_file']}\n\n=== Output ===\n{r.get('stdout') or '(no output)'}" for r in results)
    if name == "kill_task":
        return f"Task {result['task_id']} stopped."
    return json.dumps(result, ensure_ascii=False)


class BashTasks:
    """Conversation-owned host processes with recoverable logs.

    The process table is shared by runtimes so a follow-up turn can inspect a task.
    Its account/conversation key prevents cross-account task access.
    """
    _tasks: dict[tuple[int, str, str], dict] = {}
    _lock = threading.RLock()

    def __init__(self, owner: tuple[int, str], directory: Path, cancelled: threading.Event,
                 *, launcher=None, stop_process=None, display_directory: str = ""):
        self.owner, self.directory, self.cancelled = owner, directory, cancelled
        self.launcher, self.stop_process, self.display_directory = launcher, stop_process, display_directory

    def run(self, arguments: dict, cwd: Path) -> dict:
        if self.cancelled.is_set():
            return {"ok": False, "cancelled": True, "error": "Task stopped"}
        command = str(arguments.get("command") or "")
        if not command.strip():
            raise ValueError("command must not be empty")
        task_id = uuid.uuid4().hex[:12]
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.directory / (task_id + ".log")
        wait = min(36_000_000, max(0, int(arguments.get("block_until_ms", 0 if bool_arg(arguments.get("is_background")) else 30_000))))
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as log:
            argv = self.launcher(command, cwd, task_id) if self.launcher else ["/bin/bash", "-lc", command]
            process = subprocess.Popen(argv, cwd=str(cwd),
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       start_new_session=True)
        key = (*self.owner, task_id)
        with self._lock:
            self._tasks[key] = {"process": process, "path": path, "command": command,
                                "started": time.monotonic(), "stop_process": self.stop_process,
                                "output_file": (self.display_directory.rstrip("/") + "/" + path.name) if self.display_directory else str(path)}
        if self.cancelled.is_set():
            return self.kill({"task_id": task_id})
        def reap():
            try:
                process.wait(timeout=24 * 3600)
            except subprocess.TimeoutExpired:
                self.kill({"task_id": task_id})
        threading.Thread(target=reap, daemon=True).start()
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
                "exit_code": code, "stdout": output, "output_file": task["output_file"],
                "command": task["command"], "duration_secs": time.monotonic() - task["started"],
                "truncated": size > 40_000}

    def kill(self, arguments: dict) -> dict:
        with self._lock:
            task = self._tasks.get((*self.owner, str(arguments.get("task_id") or "")))
        if task is None:
            raise ValueError("Task not found in this conversation")
        process = task["process"]
        if process.poll() is None:
            if task.get("stop_process"):
                task["stop_process"](str(arguments.get("task_id")))
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


def document_text(path: Path, arguments: dict) -> str:
    """Local document extraction; no model or network calls."""
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico"}:
        raise ValueError("Image viewing is not available through this text adapter; upload the image in the conversation")
    if suffix == ".pdf":
        info = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True, timeout=60, check=True).stdout
        match = re.search(r"^Pages:\s*(\d+)", info, re.M)
        total = int(match[1]) if match else 0
        pages = str(arguments.get("pages") or "")
        if total > 10 and not pages:
            raise ValueError("Page range is required for PDFs with more than 10 pages. Max 20 pages per call.")
        if pages:
            match = re.fullmatch(r"(\d+)(?:-(\d*))?", pages)
            if not match:
                raise ValueError("Invalid PDF page range")
            first = int(match[1])
            last = int(match[2]) if match[2] else (total if "-" in pages else first)
        else:
            first, last = 1, total
        if first < 1 or last < first or last > total or last - first >= 20:
            raise ValueError("Invalid PDF page range; max 20 pages per call")
        if arguments.get("format") not in (None, "text"):
            raise ValueError("This adapter supports PDF text extraction, not page images")
        return subprocess.run(["pdftotext", "-f", str(first), "-l", str(last), str(path), "-"], capture_output=True, text=True, timeout=60, check=True).stdout
    if suffix == ".pptx":
        import zipfile
        from xml.etree import ElementTree
        with zipfile.ZipFile(path) as archive:
            names = sorted((n for n in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)), key=lambda n: int(re.search(r"(\d+)\.xml", n)[1]))
            return "\n\n".join("\n".join(node.text or "" for node in ElementTree.fromstring(archive.read(name)).iter() if node.tag.endswith("}t")) for name in names)
    if suffix == ".ipynb":
        notebook = json.loads(path.read_text())
        blocks = []
        for n, cell in enumerate(notebook.get("cells", []), 1):
            source = cell.get("source", [])
            blocks.append(f"Cell {n} ({cell.get('cell_type', '')}):\n" + (source if isinstance(source, str) else "".join(source)))
            for output in cell.get("outputs", []):
                text = output.get("text", output.get("data", {}).get("text/plain", []))
                blocks.append(text if isinstance(text, str) else "".join(text))
        return "\n\n".join(blocks)
    return path.read_bytes().decode("utf-8", errors="replace")


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

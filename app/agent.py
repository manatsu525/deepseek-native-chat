"""Host-level Agent tools.

The normal chat path intentionally keeps using the existing conversation
workspace. This module is wired only into the explicit Agent mode and gives
that mode host-level file, shell, Skill, conversation, and frontend tools.
"""

from __future__ import annotations

import json
import asyncio
import os
import shlex
import signal
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from . import attachments
from .db import Database
from .skills import SkillRegistry
from .workspace import delete_conversation_workspace
from .code_runner import _HtmlScripts


# Agent mode deliberately uses one shared host workspace.  The ordinary chat
# path keeps its own per-conversation workspace under data/workspaces.
AGENT_PROJECT_ROOT = Path(os.getenv("AGENT_WORKSPACE_ROOT", os.getenv("AGENT_PROJECT_ROOT", "/home/share")))
HOST_READ_MAX_BYTES = 8 * 1024 * 1024
HOST_WRITE_MAX_BYTES = 32 * 1024 * 1024
HOST_OUTPUT_MAX_CHARS = 100_000
HOST_LIST_MAX_ENTRIES = 4_000
HOST_SEARCH_MAX_RESULTS = 500
HOST_COMMAND_TIMEOUT = 900


AGENT_SYSTEM_PROMPT = """You are the host-level Agent for this server. You have unrestricted root-level file and shell access and may install packages, edit projects, manage this application, and manage Skills when the user asks. The shared Agent workspace is /home/share; relative host paths are resolved from there. It is strictly separate from the ordinary chat per-conversation workspace. The ordinary workspace tools (list_files, read_file, write_file, apply_line_edits, search_files, delete_file, run_python, and check_web_syntax) are not available in Agent mode. Never claim that an operation happened without calling the corresponding tool and checking its result.

Use the installed Skills as working instructions, not as a replacement for the user's request. For code or frontend deliverables in Agent mode, use the host_* and frontend_* tools under /home/share; use absolute host paths when changing the real application, repositories, server configuration, or other host resources. For a new project, create the files directly; for an existing project, preserve unrelated work. You may create, rename, inspect, and delete conversations with the conversation tools. You may install or disable Skills at any time with the Skill tools. Frontend work should use the frontend tools and should include a real syntax/build check when practical.

There are exactly two Agent scheduling rules: (1) at most one web_search or fetch_webpage call is executed in each model turn; (2) all non-web tool calls emitted in a turn execute serially in the order emitted. Host access itself is not restricted by a workspace sandbox. Do not wait for permission between ordinary tool calls; act on the user's explicit request immediately."""


def _function(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


class _FrontendParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.errors: list[str] = []
        self._stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() not in {"meta", "link", "img", "input", "br", "hr", "source", "area", "base", "embed", "param", "track", "wbr"}:
            self._stack.append(tag.casefold())

    def handle_endtag(self, tag: str) -> None:
        name = tag.casefold()
        if name not in self._stack:
            self.errors.append(f"未匹配的结束标签：{tag}")
            return
        while self._stack:
            current = self._stack.pop()
            if current == name:
                break

    def close(self) -> None:
        super().close()
        if self._stack:
            self.errors.append("未闭合标签：" + ", ".join(self._stack[-10:]))


HOST_TOOLS = [
    _function(
        "host_list_files",
        "List files and directories on the real host. Relative paths use the shared Agent workspace /home/share as the base.",
        {
            "path": {"type": "string", "description": "Absolute path or path relative to /home/share; defaults to /home/share"},
            "max_depth": {"type": "integer", "minimum": 0, "maximum": 20, "description": "Directory depth to include; defaults to 3"},
        },
        [],
    ),
    _function(
        "host_read_file",
        "Read a UTF-8 text file from anywhere on the host with line numbers.",
        {
            "path": {"type": "string", "description": "Absolute path or path relative to /home/share"},
            "start_line": {"type": "integer", "minimum": 1, "description": "Optional first line"},
            "end_line": {"type": "integer", "minimum": 1, "description": "Optional last inclusive line"},
        },
        ["path"],
    ),
    _function(
        "host_write_file",
        "Create or replace a UTF-8 text file anywhere on the host. Parent directories are created automatically.",
        {
            "path": {"type": "string", "description": "Absolute path or path relative to /home/share"},
            "content": {"type": "string", "description": "Complete file contents"},
        },
        ["path", "content"],
    ),
    _function(
        "host_apply_patch",
        "Replace exact text in a host file. Use replace_all only when every match should change.",
        {
            "path": {"type": "string", "description": "Absolute path or path relative to /home/share"},
            "old_text": {"type": "string", "description": "Exact existing text"},
            "new_text": {"type": "string", "description": "Replacement text"},
            "replace_all": {"type": "boolean", "description": "Replace every match instead of requiring one match"},
        },
        ["path", "old_text", "new_text"],
    ),
    _function(
        "host_search_files",
        "Search literal text recursively on the host. Prefer a focused project or directory path.",
        {
            "query": {"type": "string", "description": "Literal case-insensitive text"},
            "path": {"type": "string", "description": "Directory or file; defaults to /home/share"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 500, "description": "Maximum matches"},
        },
        ["query"],
    ),
    _function(
        "host_run_command",
        "Run an arbitrary bash command as the application user (root on this installation) on the real host. Use the returned stdout, stderr, and exit code as evidence.",
        {
            "command": {"type": "string", "description": "Bash command to execute"},
            "cwd": {"type": "string", "description": "Working directory; defaults to /home/share"},
            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 3600, "description": "Command timeout; defaults to 900"},
            "env": {"type": "object", "additionalProperties": {"type": "string"}, "description": "Optional environment overrides"},
        },
        ["command"],
    ),
    _function(
        "host_delete_path",
        "Delete a file or directory on the real host. A directory requires recursive=true.",
        {
            "path": {"type": "string", "description": "Absolute path or path relative to /home/share"},
            "recursive": {"type": "boolean", "description": "Allow recursive directory deletion"},
        },
        ["path"],
    ),
]


CONVERSATION_TOOLS = [
    _function("conversation_list", "List this user's conversations newest first.", {"limit": {"type": "integer", "minimum": 1, "maximum": 200}}, []),
    _function("conversation_read", "Read a conversation's messages and metadata.", {"conversation_id": {"type": "string"}, "message_limit": {"type": "integer", "minimum": 1, "maximum": 500}}, ["conversation_id"]),
    _function("conversation_create", "Create a new empty conversation for the current user.", {"title": {"type": "string"}}, []),
    _function("conversation_rename", "Rename one of the current user's conversations.", {"conversation_id": {"type": "string"}, "title": {"type": "string"}}, ["conversation_id", "title"]),
    _function("conversation_delete", "Delete a conversation and its stored attachments/workspace.", {"conversation_id": {"type": "string"}}, ["conversation_id"]),
]


SKILL_TOOLS = [
    _function("skill_list", "List built-in and user-installed Skills and whether each is enabled.", {}, []),
    _function("skill_read", "Read a Skill's complete SKILL.md.", {"skill_id": {"type": "string"}}, ["skill_id"]),
    _function("skill_install", "Install a Skill from a Git URL or a local directory containing SKILL.md.", {"source": {"type": "string"}, "name": {"type": "string"}}, ["source"]),
    _function("skill_enable", "Enable or disable a Skill for future Agent turns.", {"skill_id": {"type": "string"}, "enabled": {"type": "boolean"}}, ["skill_id", "enabled"]),
    _function("skill_remove", "Remove a user-installed Skill. Built-in Skills can only be disabled.", {"skill_id": {"type": "string"}}, ["skill_id"]),
]


FRONTEND_TOOLS = [
    _function("frontend_list_pages", "Find HTML and common frontend source pages under a host directory.", {"path": {"type": "string"}, "max_depth": {"type": "integer", "minimum": 0, "maximum": 20}}, []),
    _function("frontend_read_page", "Read one frontend page or component from the host.", {"path": {"type": "string"}}, ["path"]),
    _function("frontend_write_page", "Create or replace one frontend page or component on the host.", {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    _function("frontend_validate_page", "Parse an HTML page and syntax-check JavaScript pages when Node.js is available.", {"path": {"type": "string"}}, ["path"]),
]


class AgentRuntime:
    def __init__(self, database: Database, user_id: int, conversation_id: str) -> None:
        self.db = database
        self.user_id = int(user_id)
        self.conversation_id = str(conversation_id)
        self.skills = SkillRegistry()
        self._cancelled = threading.Event()

    async def execute_async(self, name: str, arguments: dict[str, Any]) -> str:
        worker = asyncio.create_task(asyncio.to_thread(self.execute, name, arguments))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            self._cancelled.set()
            # Cancelling to_thread does not stop its worker. Keep the job slot
            # occupied until the command has exited and the tool has returned.
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    self._cancelled.set()
            worker.result()
            raise

    @property
    def tool_definitions(self) -> list[dict[str, Any]]:
        return [*HOST_TOOLS, *CONVERSATION_TOOLS, *SKILL_TOOLS, *FRONTEND_TOOLS]

    @staticmethod
    def _path(value: Any, *, required: bool = True) -> Path:
        raw = str(value or "").strip()
        if not raw:
            if required:
                raise ValueError("路径不能为空")
            return AGENT_PROJECT_ROOT
        path = Path(raw).expanduser()
        return path if path.is_absolute() else AGENT_PROJECT_ROOT / path

    def _host_list_files(self, arguments: dict[str, Any]) -> dict[str, Any]:
        root = self._path(arguments.get("path"), required=False)
        max_depth = max(0, min(20, int(arguments.get("max_depth", 3) or 0)))
        if not root.exists():
            raise ValueError(f"路径不存在：{root}")
        if root.is_file():
            return {"root": str(root), "entries": [{"path": str(root), "type": "file", "size": root.stat().st_size}]}
        entries: list[dict[str, Any]] = []
        for current, directories, files in os.walk(root, followlinks=False):
            current_path = Path(current)
            depth = len(current_path.relative_to(root).parts)
            if depth >= max_depth:
                directories[:] = []
            for directory in sorted(directories, key=str.casefold):
                entries.append({"path": str(current_path / directory), "type": "directory"})
            for filename in sorted(files, key=str.casefold):
                path = current_path / filename
                try:
                    size = path.stat().st_size
                except OSError:
                    size = 0
                entries.append({"path": str(path), "type": "file", "size": size})
            if len(entries) >= HOST_LIST_MAX_ENTRIES:
                break
        return {"root": str(root), "entries": entries[:HOST_LIST_MAX_ENTRIES], "truncated": len(entries) > HOST_LIST_MAX_ENTRIES}

    def _host_read_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._path(arguments.get("path"))
        if not path.is_file():
            raise ValueError(f"文件不存在：{path}")
        if path.stat().st_size > HOST_READ_MAX_BYTES:
            raise ValueError(f"文件过大（上限 {HOST_READ_MAX_BYTES // 1024 // 1024}MB）：{path}")
        content = path.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        first = max(1, int(arguments.get("start_line", 1) or 1))
        last = len(lines) if arguments.get("end_line") is None else int(arguments["end_line"])
        if first > len(lines) and lines:
            raise ValueError(f"start_line 超出文件范围（共 {len(lines)} 行）")
        last = min(max(0, last), len(lines))
        selected = lines[first - 1:last] if lines else []
        numbered = "\n".join(f"{first + index}|{line}" for index, line in enumerate(selected))
        return {"path": str(path), "line_count": len(lines), "start_line": first, "end_line": last, "content": numbered}

    def _host_write_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._path(arguments.get("path"))
        content = str(arguments.get("content") or "")
        encoded = content.encode("utf-8")
        if len(encoded) > HOST_WRITE_MAX_BYTES:
            raise ValueError(f"文件过大（上限 {HOST_WRITE_MAX_BYTES // 1024 // 1024}MB）")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
        return {"ok": True, "path": str(path), "size": len(encoded)}

    def _host_apply_patch(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._path(arguments.get("path"))
        if not path.is_file():
            raise ValueError(f"文件不存在：{path}")
        old = str(arguments.get("old_text") or "")
        if not old:
            raise ValueError("old_text 不能为空")
        content = path.read_text(encoding="utf-8", errors="replace")
        matches = content.count(old)
        replace_all = bool(arguments.get("replace_all", False))
        if not matches:
            raise ValueError("old_text 与当前文件不匹配")
        if matches > 1 and not replace_all:
            raise ValueError(f"old_text 出现 {matches} 次；请缩小上下文或启用 replace_all")
        updated = content.replace(old, str(arguments.get("new_text") or ""), -1 if replace_all else 1)
        path.write_text(updated, encoding="utf-8")
        return {"ok": True, "path": str(path), "replacements": matches if replace_all else 1}

    def _host_search_files(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query") or "")
        if not query:
            raise ValueError("搜索内容不能为空")
        target = self._path(arguments.get("path"), required=False)
        if not target.exists():
            raise ValueError(f"搜索路径不存在：{target}")
        limit = max(1, min(HOST_SEARCH_MAX_RESULTS, int(arguments.get("max_results", 100) or 100)))
        candidates = [target] if target.is_file() else []
        if target.is_dir():
            for current, directories, files in os.walk(target, followlinks=False):
                directories[:] = [item for item in directories if item not in {".git", "node_modules", ".venv", "__pycache__"}]
                candidates.extend(Path(current) / item for item in files)
        matches: list[dict[str, Any]] = []
        folded = query.casefold()
        for path in sorted(candidates, key=lambda item: str(item).casefold()):
            try:
                if path.stat().st_size > HOST_READ_MAX_BYTES:
                    continue
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line_number, line in enumerate(lines, 1):
                if folded in line.casefold():
                    matches.append({"path": str(path), "line": line_number, "text": line[:500]})
                    if len(matches) >= limit:
                        return {"matches": matches, "truncated": True}
        return {"matches": matches, "truncated": False}

    def _host_run_command(self, arguments: dict[str, Any]) -> dict[str, Any]:
        command = str(arguments.get("command") or "")
        if not command:
            raise ValueError("command 不能为空")
        cwd = self._path(arguments.get("cwd"), required=False)
        if not cwd.is_dir():
            raise ValueError(f"工作目录不存在：{cwd}")
        timeout = max(1, min(3600, int(arguments.get("timeout_seconds", HOST_COMMAND_TIMEOUT) or HOST_COMMAND_TIMEOUT)))
        environment = os.environ.copy()
        supplied_env = arguments.get("env")
        if isinstance(supplied_env, dict):
            environment.update({str(key): str(value) for key, value in supplied_env.items()})
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            if self._cancelled.is_set():
                return {"ok": False, "cancelled": True, "error": "任务已停止"}
            process = subprocess.Popen(
                ["/bin/bash", "-lc", command], cwd=str(cwd), env=environment,
                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                start_new_session=True,
            )
            deadline = time.monotonic() + timeout
            timed_out = False
            cancelled = False
            try:
                while True:
                    cancelled = self._cancelled.is_set()
                    timed_out = time.monotonic() >= deadline
                    if cancelled or timed_out:
                        # Kill the process group, including shell children.
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                        break
                    if process.poll() is not None:
                        break
                    self._cancelled.wait(0.1)
            finally:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            def tail(stream: Any) -> str:
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell() - HOST_OUTPUT_MAX_CHARS * 4))
                return stream.read().decode("utf-8", errors="replace")[-HOST_OUTPUT_MAX_CHARS:]
            return {
                "ok": process.returncode == 0 and not cancelled and not timed_out,
                "exit_code": int(process.returncode),
                "cwd": str(cwd),
                "stdout": tail(stdout), "stderr": tail(stderr),
                "timeout": timed_out, "cancelled": cancelled,
                "error": "任务已停止" if cancelled else "命令执行超时" if timed_out else "",
            }

    def _host_delete_path(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._path(arguments.get("path"))
        if not path.exists() and not path.is_symlink():
            raise ValueError(f"路径不存在：{path}")
        if path.is_dir() and not path.is_symlink():
            if not bool(arguments.get("recursive", False)):
                raise ValueError("删除目录必须显式设置 recursive=true")
            shutil.rmtree(path)
        else:
            path.unlink()
        return {"ok": True, "path": str(path)}

    def _conversation_list(self, arguments: dict[str, Any]) -> dict[str, Any]:
        limit = max(1, min(200, int(arguments.get("limit", 100) or 100)))
        rows = self.db.all(
            "SELECT id,title,created_at,updated_at,pinned_at FROM conversations WHERE user_id=? ORDER BY pinned_at IS NULL,pinned_at DESC,updated_at DESC LIMIT ?",
            (self.user_id, limit),
        )
        for row in rows:
            row["pinned"] = row.get("pinned_at") is not None
        return {"conversations": rows}

    def _owned_conversation(self, conversation_id: str) -> dict[str, Any]:
        row = self.db.one("SELECT * FROM conversations WHERE id=? AND user_id=?", (str(conversation_id), self.user_id))
        if not row:
            raise ValueError("对话不存在")
        return row

    def _conversation_read(self, arguments: dict[str, Any]) -> dict[str, Any]:
        conversation_id = str(arguments.get("conversation_id") or "")
        conversation = self._owned_conversation(conversation_id)
        limit = max(1, min(500, int(arguments.get("message_limit", 200) or 200)))
        rows = self.db.all(
            "SELECT id,role,content,meta_json,created_at FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
            (conversation_id, limit),
        )
        rows.reverse()
        for row in rows:
            try:
                row["meta"] = json.loads(row.pop("meta_json") or "{}")
            except ValueError:
                row["meta"] = {}
        return {"conversation": conversation, "messages": rows}

    def _conversation_create(self, arguments: dict[str, Any]) -> dict[str, Any]:
        title = " ".join(str(arguments.get("title") or "新对话").split())[:100] or "新对话"
        conversation_id = uuid.uuid4().hex
        stamp = int(time.time())
        self.db.run("INSERT INTO conversations(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)", (conversation_id, self.user_id, title, stamp, stamp))
        return {"ok": True, "conversation": self._owned_conversation(conversation_id)}

    def _conversation_rename(self, arguments: dict[str, Any]) -> dict[str, Any]:
        conversation_id = str(arguments.get("conversation_id") or "")
        self._owned_conversation(conversation_id)
        title = " ".join(str(arguments.get("title") or "").split())[:100]
        if not title:
            raise ValueError("标题不能为空")
        self.db.run("UPDATE conversations SET title=?,updated_at=? WHERE id=? AND user_id=?", (title, int(time.time()), conversation_id, self.user_id))
        return {"ok": True, "conversation": self._owned_conversation(conversation_id)}

    def _conversation_delete(self, arguments: dict[str, Any]) -> dict[str, Any]:
        conversation_id = str(arguments.get("conversation_id") or "")
        self._owned_conversation(conversation_id)
        if self.db.one("SELECT id FROM jobs WHERE conversation_id=? AND user_id=? AND status IN ('queued','running')", (conversation_id, self.user_id)):
            raise ValueError("当前对话仍有生成中的任务")
        records = self.db.all("SELECT * FROM attachments WHERE user_id=? AND (conversation_id=? OR (job_id IS NULL AND draft_id=?))", (self.user_id, conversation_id, conversation_id))
        if records:
            self.db.delete_attachments(self.user_id, [item["id"] for item in records])
        self.db.run("DELETE FROM conversations WHERE id=? AND user_id=?", (conversation_id, self.user_id))
        attachments.delete_files(records)
        delete_conversation_workspace(self.user_id, conversation_id)
        return {"ok": True, "conversation_id": conversation_id}

    def _skill_list(self, _: dict[str, Any]) -> dict[str, Any]:
        enabled = set(self.skills.enabled_ids())
        return {"skills": [{"id": item.skill_id, "name": item.name, "description": item.description, "builtin": item.builtin, "enabled": item.skill_id in enabled, "path": str(item.path)} for item in self.skills.all()]}

    def _skill_read(self, arguments: dict[str, Any]) -> dict[str, Any]:
        skill_id = str(arguments.get("skill_id") or "")
        skill = self.skills.find(skill_id)
        if skill is None:
            raise ValueError(f"Skill 不存在：{skill_id}")
        return {"id": skill.skill_id, "name": skill.name, "builtin": skill.builtin, "content": self.skills.read(skill.skill_id)}

    def _skill_install(self, arguments: dict[str, Any]) -> dict[str, Any]:
        skill = self.skills.install(str(arguments.get("source") or ""), str(arguments.get("name") or ""))
        self.skills.set_enabled(skill.skill_id, True)
        return {"ok": True, "id": skill.skill_id, "name": skill.name, "path": str(skill.path), "enabled": True}

    def _skill_enable(self, arguments: dict[str, Any]) -> dict[str, Any]:
        skill_id = str(arguments.get("skill_id") or "")
        values = self.skills.set_enabled(skill_id, bool(arguments.get("enabled")))
        return {"ok": True, "enabled": values}

    def _skill_remove(self, arguments: dict[str, Any]) -> dict[str, Any]:
        skill_id = str(arguments.get("skill_id") or "")
        self.skills.remove(skill_id)
        return {"ok": True, "removed": skill_id}

    def _frontend_list_pages(self, arguments: dict[str, Any]) -> dict[str, Any]:
        root = self._path(arguments.get("path"), required=False)
        max_depth = max(0, min(20, int(arguments.get("max_depth", 6) or 6)))
        extensions = {".html", ".htm", ".js", ".jsx", ".ts", ".tsx", ".vue", ".svelte", ".css", ".scss"}
        listing = self._host_list_files({"path": str(root), "max_depth": max_depth})
        listing["entries"] = [item for item in listing["entries"] if item.get("type") == "file" and Path(item["path"]).suffix.casefold() in extensions]
        return listing

    def _frontend_validate_page(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._path(arguments.get("path"))
        if not path.is_file():
            raise ValueError(f"文件不存在：{path}")
        suffix = path.suffix.casefold()
        if suffix in {".jsx", ".ts", ".tsx"}:
            return {"ok": False, "path": str(path), "error": "此文件需要项目的 TypeScript/JSX 编译或构建检查；node --check 不能完整验证。"}
        def check_js(script: Path) -> dict[str, Any]:
            return self._host_run_command({"command": "node --check " + shlex.quote(str(script.resolve())),
                                           "cwd": str(path.resolve().parent), "timeout_seconds": 30})
        if suffix in {".js", ".mjs", ".cjs"}:
            return {**check_js(path), "path": str(path)}
        if suffix in {".html", ".htm"}:
            content = path.read_text(encoding="utf-8", errors="replace")
            parser = _FrontendParser()
            parser.feed(content)
            parser.close()
            scripts = _HtmlScripts()
            scripts.feed(content)
            scripts.close()
            errors = list(parser.errors)
            if scripts.unclosed_script:
                errors.append("未闭合 script 标签")
            checked: list[str] = []
            skipped: list[str] = []
            with tempfile.TemporaryDirectory(prefix="agent-js-check-") as directory:
                checks: list[tuple[str, Path]] = []
                for index, (source, module) in enumerate(scripts.inline):
                    target = Path(directory) / f"inline-{index}{'.mjs' if module else '.cjs'}"
                    target.write_text(source, encoding="utf-8")
                    checks.append((f"inline-script-{index + 1}", target))
                for index, handler in enumerate(scripts.handlers):
                    target = Path(directory) / f"handler-{index}.cjs"
                    target.write_text("function handler(event) {\n" + handler + "\n}\n", encoding="utf-8")
                    checks.append((f"event-handler-{index + 1}", target))
                for source in scripts.sources:
                    parsed = urlsplit(source)
                    if parsed.scheme or parsed.netloc or parsed.path.startswith("/"):
                        skipped.append(source)
                        continue
                    target = path.parent / unquote(parsed.path)
                    if not target.is_file():
                        errors.append(f"引用的本地脚本不存在：{source}")
                    else:
                        checks.append((source, target))
                for label, target in checks:
                    result = check_js(target)
                    checked.append(label)
                    if not result["ok"]:
                        errors.append(f"{label}: {result.get('error') or result.get('stderr') or '语法检查失败'}")
                    if result.get("cancelled"):
                        break
            return {"ok": not errors, "path": str(path), "errors": errors, "checked_scripts": checked,
                    "skipped_scripts": skipped, "validation_scope": "HTML 标签与已列出的本地 JavaScript 语法；未验证浏览器行为或跳过的 URL"}
        raise ValueError("前端语法检查支持 HTML、JavaScript、TypeScript 页面")

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        if self._cancelled.is_set():
            return _json({"ok": False, "cancelled": True, "error": "任务已停止"})
        dispatch = {
            "host_list_files": self._host_list_files,
            "host_read_file": self._host_read_file,
            "host_write_file": self._host_write_file,
            "host_apply_patch": self._host_apply_patch,
            "host_search_files": self._host_search_files,
            "host_run_command": self._host_run_command,
            "host_delete_path": self._host_delete_path,
            "conversation_list": self._conversation_list,
            "conversation_read": self._conversation_read,
            "conversation_create": self._conversation_create,
            "conversation_rename": self._conversation_rename,
            "conversation_delete": self._conversation_delete,
            "skill_list": self._skill_list,
            "skill_read": self._skill_read,
            "skill_install": self._skill_install,
            "skill_enable": self._skill_enable,
            "skill_remove": self._skill_remove,
            "frontend_list_pages": self._frontend_list_pages,
            "frontend_read_page": lambda args: self._host_read_file(args),
            "frontend_write_page": lambda args: self._host_write_file(args),
            "frontend_validate_page": self._frontend_validate_page,
        }
        handler = dispatch.get(name)
        if handler is None:
            raise ValueError(f"不支持的 Agent 工具：{name}")
        try:
            return _json(handler(arguments))
        except Exception as exc:
            return _json({"ok": False, "error": str(exc)[:4_000]})


def build_agent_system_prompt() -> str:
    prompt = AGENT_SYSTEM_PROMPT
    skills = SkillRegistry().prompt()
    return f"{prompt}\n\n{skills}" if skills else prompt


def build_agent_skills_prompt() -> str:
    """Return the current enabled Skill bodies for a shared system addendum."""
    return SkillRegistry().prompt()

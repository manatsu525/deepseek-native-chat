from __future__ import annotations

import json
import re
import shutil
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

from .config import settings


WORKSPACES_DIR = settings.data_dir / "workspaces"
MAX_FILES = 200
MAX_FILE_BYTES = 512 * 1024
MAX_TOTAL_BYTES = 10 * 1024 * 1024
MAX_READ_CHARS = 60_000
MAX_SEARCH_RESULTS = 20

WORKSPACE_SYSTEM_PROMPT = """A persistent, isolated coding workspace is available for this conversation. When the user asks you to create code or a multi-file project, use the workspace tools to save the actual files instead of only printing complete files in chat. On later requests, inspect the existing workspace files and patch only what needs to change. Do not recreate or overwrite unrelated files. Every workspace tool call must include every field marked required in its JSON schema. File tools must include a non-empty workspace-relative path exactly as listed by list_files (or the intended new relative path for write_file), except when a tool's own description explicitly says its path is already bound; a pre-bound tool must not receive path. Before emitting a tool call, verify its arguments against the schema and never repeat an unchanged read_file call. When making multiple edits from one read, submit one apply_patch_batch call instead of sequential apply_patch calls, because earlier edits invalidate the old snapshot. Saved Python programs can be verified with run_python. Saved HTML and JavaScript must be checked with check_web_syntax when that tool is available; it parses HTML and runs Node.js syntax checks on inline, event-handler, and local JavaScript without executing it. These tools run without network in disposable resource-limited copies. Treat a nonzero exit or ok=false as a real failure and fix it before claiming success. Syntax success does not prove browser behavior is correct, so state that limitation. After editing, briefly summarize changed files; the UI supplies download links automatically."""


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


WORKSPACE_TOOLS = [
    _function(
        "list_files",
        "List files currently saved in this conversation's coding workspace.",
        {},
        [],
    ),
    _function(
        "read_file",
        "Read a UTF-8 text file from the workspace before modifying it.",
        {"path": {"type": "string", "description": "Workspace-relative path"}},
        ["path"],
    ),
    _function(
        "write_file",
        "Create or completely replace one UTF-8 text file. Prefer apply_patch for later modifications.",
        {
            "path": {"type": "string", "description": "Workspace-relative path"},
            "content": {"type": "string", "description": "Complete file contents"},
        },
        ["path", "content"],
    ),
    _function(
        "apply_patch",
        "Modify an existing text file with one exact replacement. Read the file first. For multiple edits based on one read, prefer apply_patch_batch.",
        {
            "path": {"type": "string", "description": "Workspace-relative path"},
            "old_text": {"type": "string", "description": "Exact existing text to replace"},
            "new_text": {"type": "string", "description": "Replacement text"},
            "replace_all": {"type": "boolean", "description": "Replace every exact match; defaults to false"},
        },
        ["path", "old_text", "new_text"],
    ),
    _function(
        "apply_patch_batch",
        "Atomically apply several non-overlapping exact replacements to one file, all matched against the same file snapshot. Prefer this over sequential apply_patch calls.",
        {
            "path": {"type": "string", "description": "Workspace-relative path"},
            "patches": {
                "type": "array",
                "minItems": 1,
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "properties": {
                        "old_text": {"type": "string", "description": "Exact text from the current file snapshot"},
                        "new_text": {"type": "string", "description": "Replacement text"},
                        "replace_all": {"type": "boolean", "description": "Replace all exact matches; defaults to false"},
                    },
                    "required": ["old_text", "new_text"],
                    "additionalProperties": False,
                },
            },
        },
        ["path", "patches"],
    ),
    _function(
        "search_files",
        "Search text across workspace files and return matching paths and line snippets.",
        {
            "query": {"type": "string", "description": "Literal case-insensitive text to find"},
            "path": {"type": "string", "description": "Optional file or directory to search; defaults to the workspace root"},
        },
        ["query"],
    ),
    _function(
        "delete_file",
        "Delete one workspace file when the user asks for it or it is genuinely obsolete.",
        {"path": {"type": "string", "description": "Workspace-relative file path"}},
        ["path"],
    ),
]

RUN_PYTHON_TOOL = _function(
    "run_python",
    "Run one saved Python file in an isolated disposable copy of the workspace. Network is disabled and time/memory/process limits apply. Use the real output to verify and fix code; never claim success when ok is false.",
    {
        "path": {"type": "string", "description": "Existing .py file to execute"},
        "arguments": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 20,
            "description": "Optional command-line arguments passed directly to the Python file",
        },
    },
    ["path"],
)

CHECK_WEB_SYNTAX_TOOL = _function(
    "check_web_syntax",
    "Check one saved HTML or JavaScript file in an isolated disposable copy. HTML parsing includes inline scripts, inline event handlers, and referenced local JS files. JavaScript is checked with node --check but not executed. Fix every reported syntax error before claiming success.",
    {"path": {"type": "string", "description": "Existing .html, .htm, .js, .mjs, or .cjs file"}},
    ["path"],
)

WORKSPACE_TOOL_NAMES = {item["function"]["name"] for item in [*WORKSPACE_TOOLS, RUN_PYTHON_TOOL, CHECK_WEB_SYNTAX_TOOL]}


class WorkspaceError(ValueError):
    pass


class ConversationWorkspace:
    def __init__(self, user_id: int, conversation_id: str) -> None:
        if not str(user_id).isdigit() or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", conversation_id or ""):
            raise WorkspaceError("无效的工作区标识")
        self.root = WORKSPACES_DIR / str(user_id) / conversation_id

    @staticmethod
    def _clean_path(value: Any, *, allow_root: bool = False) -> PurePosixPath:
        raw = str(value or "").strip()
        if allow_root and raw in {"", "."}:
            return PurePosixPath(".")
        if not raw or "\x00" in raw or "\\" in raw:
            raise WorkspaceError("文件路径无效")
        path = PurePosixPath(raw)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise WorkspaceError("只能使用工作区内的相对路径")
        if len(path.as_posix()) > 300:
            raise WorkspaceError("文件路径过长")
        return path

    def resolve(self, value: Any, *, allow_root: bool = False) -> tuple[Path, str]:
        relative = self._clean_path(value, allow_root=allow_root)
        candidate = self.root if relative == PurePosixPath(".") else self.root.joinpath(*relative.parts)
        root_resolved = self.root.resolve(strict=False)
        resolved = candidate.resolve(strict=False)
        if resolved != root_resolved and root_resolved not in resolved.parents:
            raise WorkspaceError("文件路径越过了工作区边界")
        return candidate, "" if relative == PurePosixPath(".") else relative.as_posix()

    def _files(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(
            (path for path in self.root.rglob("*") if path.is_file() and not path.is_symlink()),
            key=lambda item: item.relative_to(self.root).as_posix().casefold(),
        )

    def list_files(self) -> list[dict[str, Any]]:
        return [
            {"path": path.relative_to(self.root).as_posix(), "size": path.stat().st_size}
            for path in self._files()
        ]

    def tool_definitions(self) -> list[dict[str, Any]]:
        """Return a byte-stable schema so provider prefix caches stay reusable.

        Existing paths are runtime state, not part of a tool's contract.  Putting
        them into JSON-schema enums made the entire tool prefix change after every
        write, invalidating provider prompt caches.  ``list_files`` remains the
        authoritative way for the model to discover paths.
        """
        return deepcopy([*WORKSPACE_TOOLS, RUN_PYTHON_TOOL, CHECK_WEB_SYNTAX_TOOL])

    def _read_text(self, path: Any) -> tuple[str, str]:
        target, relative = self.resolve(path)
        if not target.is_file() or target.is_symlink():
            raise WorkspaceError(f"文件不存在：{relative}")
        if target.stat().st_size > MAX_FILE_BYTES:
            raise WorkspaceError("文件过大，无法读取")
        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceError("工作区工具只支持 UTF-8 文本文件") from exc
        return content, relative

    def read_file(self, path: Any) -> str:
        content, _ = self._read_text(path)
        if len(content) > MAX_READ_CHARS:
            return content[:MAX_READ_CHARS] + "\n\n[内容过长，已截断]"
        return content

    def _validate_write(self, target: Path, content: str) -> bytes:
        encoded = content.encode("utf-8")
        if len(encoded) > MAX_FILE_BYTES:
            raise WorkspaceError(f"单个文件不能超过 {MAX_FILE_BYTES // 1024}KB")
        files = self._files()
        if not target.exists() and len(files) >= MAX_FILES:
            raise WorkspaceError(f"一个工作区最多保存 {MAX_FILES} 个文件")
        current_size = target.stat().st_size if target.is_file() and not target.is_symlink() else 0
        total = sum(item.stat().st_size for item in files) - current_size + len(encoded)
        if total > MAX_TOTAL_BYTES:
            raise WorkspaceError(f"一个工作区最多占用 {MAX_TOTAL_BYTES // 1024 // 1024}MB")
        return encoded

    def write_file(self, path: Any, content: Any) -> dict[str, Any]:
        target, relative = self.resolve(path)
        if target.exists() and (not target.is_file() or target.is_symlink()):
            raise WorkspaceError("目标路径不是普通文件")
        text = str(content or "")
        encoded = self._validate_write(target, text)
        target.parent.mkdir(parents=True, exist_ok=True)
        for directory in (WORKSPACES_DIR, WORKSPACES_DIR / self.root.parent.name, self.root):
            if directory.exists():
                directory.chmod(0o700)
        target.write_bytes(encoded)
        target.chmod(0o600)
        return {"ok": True, "path": relative, "size": len(encoded)}

    def apply_patch(self, path: Any, old_text: Any, new_text: Any, replace_all: Any = False) -> dict[str, Any]:
        old = str(old_text or "")
        if not old:
            raise WorkspaceError("old_text 不能为空")
        content, _ = self._read_text(path)
        matches = content.count(old)
        if not matches:
            raise WorkspaceError("old_text 与当前文件不匹配；请重新读取文件后再修改")
        if matches > 1 and not bool(replace_all):
            raise WorkspaceError(f"old_text 在文件中出现 {matches} 次；请提供更精确的上下文或启用 replace_all")
        updated = content.replace(old, str(new_text or ""), -1 if bool(replace_all) else 1)
        result = self.write_file(path, updated)
        result["replacements"] = matches if bool(replace_all) else 1
        return result

    def apply_patch_batch(self, path: Any, patches: Any) -> dict[str, Any]:
        if not isinstance(patches, list) or not patches or len(patches) > 20:
            raise WorkspaceError("patches 必须是包含 1 到 20 项的数组")
        content, _ = self._read_text(path)
        replacements: list[tuple[int, int, str, int]] = []
        for patch_index, patch in enumerate(patches):
            if not isinstance(patch, dict):
                raise WorkspaceError(f"第 {patch_index + 1} 个补丁不是对象")
            old = str(patch.get("old_text") or "")
            if not old:
                raise WorkspaceError(f"第 {patch_index + 1} 个补丁的 old_text 不能为空")
            starts: list[int] = []
            offset = 0
            while True:
                found = content.find(old, offset)
                if found < 0:
                    break
                starts.append(found)
                offset = found + len(old)
            if not starts:
                raise WorkspaceError(f"第 {patch_index + 1} 个补丁的 old_text 与当前文件不匹配；整个批次未修改")
            replace_all = bool(patch.get("replace_all", False))
            if len(starts) > 1 and not replace_all:
                raise WorkspaceError(f"第 {patch_index + 1} 个补丁的 old_text 出现 {len(starts)} 次；请提供更精确上下文")
            for start in starts if replace_all else starts[:1]:
                replacements.append((start, start + len(old), str(patch.get("new_text") or ""), patch_index))
        ordered = sorted(replacements, key=lambda item: (item[0], item[1]))
        for previous, current in zip(ordered, ordered[1:]):
            if current[0] < previous[1]:
                raise WorkspaceError(f"第 {previous[3] + 1} 和第 {current[3] + 1} 个补丁范围重叠；整个批次未修改")
        updated = content
        for start, end, new_text, _ in reversed(ordered):
            updated = updated[:start] + new_text + updated[end:]
        result = self.write_file(path, updated)
        result["changes"] = len(patches)
        result["replacements"] = len(replacements)
        return result

    def run_python(self, path: Any, arguments: Any = None) -> dict[str, Any]:
        from .code_runner import run_python

        _, relative = self.resolve(path)
        return run_python(self.root, relative, arguments)

    def check_web_syntax(self, path: Any) -> dict[str, Any]:
        from .code_runner import check_web_syntax

        _, relative = self.resolve(path)
        return check_web_syntax(self.root, relative)

    def search_files(self, query: Any, path: Any = "") -> dict[str, Any]:
        needle = str(query or "")
        if not needle:
            raise WorkspaceError("搜索内容不能为空")
        target, relative = self.resolve(path, allow_root=True)
        if target.is_symlink() or not target.exists():
            raise WorkspaceError(f"搜索路径不存在：{relative or '.'}")
        candidates = [target] if target.is_file() else [item for item in target.rglob("*") if item.is_file() and not item.is_symlink()]
        matches: list[dict[str, Any]] = []
        folded = needle.casefold()
        for file_path in sorted(candidates):
            if file_path.stat().st_size > MAX_FILE_BYTES:
                continue
            try:
                lines = file_path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for number, line in enumerate(lines, 1):
                if folded in line.casefold():
                    matches.append({"path": file_path.relative_to(self.root).as_posix(), "line": number, "text": line[:300]})
                    if len(matches) >= MAX_SEARCH_RESULTS:
                        return {"matches": matches, "truncated": True}
        return {"matches": matches, "truncated": False}

    def delete_file(self, path: Any) -> dict[str, Any]:
        target, relative = self.resolve(path)
        if not target.is_file() or target.is_symlink():
            raise WorkspaceError(f"文件不存在：{relative}")
        target.unlink()
        parent = target.parent
        while parent != self.root and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
        return {"ok": True, "path": relative}

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "list_files":
            result: Any = {"files": self.list_files()}
        elif name == "read_file":
            result = {"path": str(arguments.get("path") or ""), "content": self.read_file(arguments.get("path"))}
        elif name == "write_file":
            result = self.write_file(arguments.get("path"), arguments.get("content"))
        elif name == "apply_patch":
            result = self.apply_patch(arguments.get("path"), arguments.get("old_text"), arguments.get("new_text"), arguments.get("replace_all", False))
        elif name == "apply_patch_batch":
            result = self.apply_patch_batch(arguments.get("path"), arguments.get("patches"))
        elif name == "search_files":
            result = self.search_files(arguments.get("query"), arguments.get("path", ""))
        elif name == "delete_file":
            result = self.delete_file(arguments.get("path"))
        elif name == "run_python":
            result = self.run_python(arguments.get("path"), arguments.get("arguments", []))
        elif name == "check_web_syntax":
            result = self.check_web_syntax(arguments.get("path"))
        else:
            raise WorkspaceError(f"不支持的工作区工具：{name}")
        return json.dumps(result, ensure_ascii=False)


def delete_conversation_workspace(user_id: int, conversation_id: str) -> None:
    workspace = ConversationWorkspace(user_id, conversation_id)
    if workspace.root.is_dir():
        shutil.rmtree(workspace.root)


def delete_user_workspaces(user_id: int) -> None:
    root = WORKSPACES_DIR / str(user_id)
    if root.is_dir():
        shutil.rmtree(root)

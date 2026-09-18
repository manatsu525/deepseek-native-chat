from __future__ import annotations

import bisect
import difflib
import json
import hashlib
import os
import re
import shutil
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

from .config import settings


WORKSPACES_DIR = settings.data_dir / "workspaces"
AGENT_WORKSPACE_ROOT = Path(os.getenv("AGENT_WORKSPACE_ROOT", "/home/share"))
MAX_FILES = 200
MAX_FILE_BYTES = 512 * 1024
MAX_TOTAL_BYTES = 10 * 1024 * 1024
# One read returns a whole file up to this size (typical single-file apps fit).
MAX_READ_CHARS = 100_000
MAX_SEARCH_RESULTS = 20
AGENT_MAX_FILES = 4_000




READ_FILE_DESCRIPTION = (
    "Read a whole UTF-8 file as numbered lines ('N|text') in one call. Read a file once and reuse what you saw; "
    "the file is never split unless it exceeds the response limit, in which case truncated=true and next_start_line tells where to continue."
)
READ_START_LINE_DESCRIPTION = "Only for continuing a truncated read: pass the previous next_start_line. Omit otherwise."
EDIT_FILE_DESCRIPTION = (
    "Change an existing file by replacing exact text snippets. Put every change for this file in one call; all edits are "
    "applied together or not at all. Each old_text must be copied verbatim from the file (without the 'N|' prefixes) and "
    "contain enough surrounding lines to match exactly one place, unless replace_all is true. To insert, include a nearby "
    "anchor line in old_text and repeat it in new_text. To delete, use an empty new_text. The result shows each edited "
    "region with its new line numbers; the rest of the file is unchanged, so you do not need to read it again."
)
EDITS_SCHEMA = {
    "type": "array",
    "minItems": 1,
    "maxItems": 30,
    "items": {
        "type": "object",
        "properties": {
            "old_text": {"type": "string", "description": "Exact existing text to replace"},
            "new_text": {"type": "string", "description": "Replacement text (may be empty)"},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence instead of requiring exactly one"},
        },
        "required": ["old_text", "new_text"],
        "additionalProperties": False,
    },
}


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
        READ_FILE_DESCRIPTION,
        {
            "path": {"type": "string", "description": "Workspace-relative path"},
            "start_line": {"type": "integer", "minimum": 1, "description": READ_START_LINE_DESCRIPTION},
        },
        ["path"],
    ),
    _function(
        "write_file",
        "Create a new UTF-8 text file, or completely replace a file only when a full rewrite is genuinely intended. Use edit_file for changes to an existing file.",
        {
            "path": {"type": "string", "description": "Workspace-relative path"},
            "content": {"type": "string", "description": "Complete file contents"},
        },
        ["path", "content"],
    ),
    _function("edit_file", EDIT_FILE_DESCRIPTION, {
        "path": {"type": "string", "description": "Workspace-relative path of an existing file"},
        "edits": EDITS_SCHEMA,
    }, ["path", "edits"]),
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

RUN_COMMAND_TOOL = _function(
    "run_command",
    "Run one bash command in a disposable, network-less copy of the workspace (time and memory limited). Use it to grep, list, "
    "build or test. Output is real, but files the command creates or changes are discarded; change files with write_file and edit_file.",
    {
        "command": {"type": "string", "description": "Bash command, run from the workspace root"},
        "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 60, "description": "Optional timeout; defaults to 12"},
    },
    ["command"],
)

CHECK_WEB_SYNTAX_TOOL = _function(
    "check_web_syntax",
    "Check one saved HTML or JavaScript file in an isolated disposable copy. HTML parsing includes inline scripts, inline event handlers, and referenced local JS files. JavaScript is checked with node --check but not executed. Fix every reported syntax error before claiming success.",
    {"path": {"type": "string", "description": "Existing .html, .htm, .js, .mjs, or .cjs file"}},
    ["path"],
)

# Older single/batch patch names that some models or text-markup fallbacks
# still emit. They are not advertised and are executed as edit_file.
LEGACY_PATCH_TOOL_NAMES = {"apply_patch", "apply_patch_batch", "replace_text"}
VALIDATION_TOOL_NAMES = {"run_python", "run_command", "check_web_syntax"}
WORKSPACE_TOOL_NAMES = {
    item["function"]["name"] for item in [*WORKSPACE_TOOLS, RUN_PYTHON_TOOL, RUN_COMMAND_TOOL, CHECK_WEB_SYNTAX_TOOL]
} | LEGACY_PATCH_TOOL_NAMES
READ_ONLY_WORKSPACE_TOOL_NAMES = {"list_files", "read_file", "search_files"} | VALIDATION_TOOL_NAMES
EDIT_WORKSPACE_TOOL_NAMES = WORKSPACE_TOOL_NAMES - VALIDATION_TOOL_NAMES


class WorkspaceError(ValueError):
    pass


EXCERPT_CONTEXT_LINES = 2
EXCERPT_MAX_CHARS = 6_000
_LINE_NUMBER_PREFIX = re.compile(r"^\s*\d+\|")


def _line_starts(content: str) -> list[int]:
    starts = [0]
    for line in content.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    return starts


def _line_of_offset(starts: list[int], offset: int) -> int:
    """1-based line containing ``offset``; ``starts`` comes from _line_starts."""
    return min(max(1, bisect.bisect_right(starts, offset)), max(1, len(starts) - 1))


def edited_excerpt(content: str, regions: list[tuple[int, int]]) -> dict[str, Any]:
    """Numbered lines around edited character regions of the updated content.

    Returning the edited area lets the model verify the change and see the new
    line numbers without spending another round on read_file.
    """
    lines = content.splitlines()
    if not lines:
        return {"updated_excerpt": "", "excerpt_truncated": False}
    starts = _line_starts(content)
    spans: list[list[int]] = []
    for start, end in sorted(regions):
        first = _line_of_offset(starts, start)
        last = max(first, _line_of_offset(starts, max(start, end - 1)))
        first = max(1, first - EXCERPT_CONTEXT_LINES)
        last = min(len(lines), last + EXCERPT_CONTEXT_LINES)
        if spans and first <= spans[-1][1] + 1:
            spans[-1][1] = max(spans[-1][1], last)
        else:
            spans.append([first, last])
    rendered: list[str] = []
    used = 0
    truncated = False
    for index, (first, last) in enumerate(spans):
        if index:
            rendered.append("...")
        for number in range(first, last + 1):
            text = f"{number}|{lines[number - 1]}"
            if used + len(text) > EXCERPT_MAX_CHARS:
                truncated = True
                break
            rendered.append(text)
            used += len(text) + 1
        if truncated:
            break
    return {"updated_excerpt": "\n".join(rendered), "excerpt_truncated": truncated}


def _closest_region_hint(content: str, old: str) -> str:
    old_lines = old.splitlines() or [old]
    lines = content.splitlines()
    if not lines:
        return ""
    window = max(1, min(len(old_lines), len(lines)))
    target = "\n".join(line.strip() for line in old_lines)
    best_ratio, best_start = 0.0, 0
    anchor = old_lines[0].strip()
    candidates = [index for index, line in enumerate(lines) if anchor and anchor[:20] in line] or range(0, len(lines) - window + 1)
    for index in list(candidates)[:2000]:
        chunk = "\n".join(line.strip() for line in lines[index:index + window])
        ratio = difflib.SequenceMatcher(None, target, chunk).quick_ratio()
        if ratio > best_ratio:
            best_ratio, best_start = ratio, index
    if best_ratio < 0.5:
        return ""
    shown = [f"{number + 1}|{lines[number][:100]}" for number in range(best_start, min(len(lines), best_start + min(window, 8)))]
    return "最接近的当前内容：\n" + "\n".join(shown)


def numbered_window(content: str, start_line: Any = None, max_chars: int = MAX_READ_CHARS) -> dict[str, Any]:
    """Render a file as numbered lines, whole whenever it fits in one response.

    ``start_line`` only continues a read that was truncated. If the whole file
    fits, it is returned from line 1 regardless, so a model can never turn one
    read into several overlapping partial reads.
    """
    lines = content.splitlines()
    try:
        first = int(start_line) if start_line not in (None, "") else 1
    except (TypeError, ValueError) as exc:
        raise WorkspaceError("start_line 无效") from exc
    if first < 1:
        raise WorkspaceError("start_line 必须大于等于 1")
    if lines and first > len(lines):
        raise WorkspaceError(f"start_line 超出文件范围（共 {len(lines)} 行）")
    whole_size = sum(len(str(number)) + 2 + len(line) for number, line in enumerate(lines, 1))
    note = ""
    if whole_size <= max_chars:
        if first != 1:
            note = "文件可以一次读完，已忽略 start_line 并返回全文。"
        first = 1
    rendered: list[str] = []
    used = 0
    through = first - 1 if lines else 0
    for number in range(first, len(lines) + 1):
        text = f"{number}|{lines[number - 1]}"
        if rendered and used + len(text) + 1 > max_chars:
            break
        rendered.append(text)
        used += len(text) + 1
        through = number
    result: dict[str, Any] = {
        "line_count": len(lines),
        "from_line": first if rendered else 0,
        "through_line": through,
        "truncated": through < len(lines),
        "content": "\n".join(rendered),
    }
    if result["truncated"]:
        result["next_start_line"] = through + 1
    if note:
        result["note"] = note
    return result


_PATH_ALIASES = ("file_path", "filepath", "file", "filename")
_EDITS_ALIASES = ("changes", "replacements", "patches")
_OLD_ALIASES = ("old_string", "old", "search", "find")
_NEW_ALIASES = ("new_string", "new", "replace", "replacement")
_CONTENT_ALIASES = ("contents", "text", "code")
FILE_TOOL_NAMES = {
    "read_file", "write_file", "edit_file", "delete_file", "run_python", "check_web_syntax",
    "host_read_file", "host_write_file", "host_edit_file", "host_delete_path",
    "frontend_read_page", "frontend_write_page", "frontend_validate_page",
}


def _first_alias(source: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    return next((source[key] for key in aliases if key in source), None)


def normalize_file_tool_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Accept common argument spellings from other agent tool conventions.

    Models trained on other tools often send file_path, old_string/new_string,
    a single top-level edit, or wrap everything in one object. Only missing
    canonical fields are filled in; canonical fields always win.
    """
    if name not in FILE_TOOL_NAMES or not isinstance(arguments, dict):
        return arguments
    result = dict(arguments)
    if len(result) == 1:
        (only,) = result.values()
        if isinstance(only, str):
            try:
                only = json.loads(only)
            except (TypeError, ValueError):
                pass
        if isinstance(only, dict) and ({"path", "edits", "content"} | set(_PATH_ALIASES)) & set(only):
            result = dict(only)
    if "path" not in result:
        alias = _first_alias(result, _PATH_ALIASES)
        if alias is not None:
            result["path"] = alias
    if name in {"write_file", "host_write_file", "frontend_write_page"} and "content" not in result:
        alias = _first_alias(result, _CONTENT_ALIASES)
        if alias is not None:
            result["content"] = alias
    if name in {"edit_file", "host_edit_file"}:
        edits = result.get("edits", _first_alias(result, _EDITS_ALIASES))
        if isinstance(edits, str):
            try:
                edits = json.loads(edits)
            except (TypeError, ValueError):
                pass
        if isinstance(edits, dict):
            edits = [edits]
        if edits is None and ("old_text" in result or _first_alias(result, _OLD_ALIASES) is not None):
            edits = [{key: result[key] for key in ("old_text", "new_text", "replace_all", *_OLD_ALIASES, *_NEW_ALIASES) if key in result}]
        if isinstance(edits, list):
            normalized = []
            for edit in edits:
                if isinstance(edit, dict):
                    edit = dict(edit)
                    if "old_text" not in edit and _first_alias(edit, _OLD_ALIASES) is not None:
                        edit["old_text"] = _first_alias(edit, _OLD_ALIASES)
                    if "new_text" not in edit and _first_alias(edit, _NEW_ALIASES) is not None:
                        edit["new_text"] = _first_alias(edit, _NEW_ALIASES)
                    edit = {key: edit[key] for key in ("old_text", "new_text", "replace_all") if key in edit}
                normalized.append(edit)
            result["edits"] = normalized
    return result


def edit_list(name: str, arguments: dict[str, Any]) -> Any:
    """Normalize edit_file and legacy single/batch patch arguments to an edit list."""
    if name == "apply_patch_batch":
        return arguments.get("patches")
    if name in {"apply_patch", "replace_text", "host_apply_patch"} or (
        "edits" not in arguments and "old_text" in arguments
    ):
        return [{key: arguments[key] for key in ("old_text", "new_text", "replace_all") if key in arguments}]
    return arguments.get("edits")


def apply_text_edits(content: str, edits: Any) -> tuple[str, list[tuple[int, int]], list[dict[str, Any]]]:
    """Apply several exact-snippet edits to one file atomically.

    All snippets are located in the original content, must not overlap, and
    are applied together; any failure leaves the file untouched. Returns the
    updated content, the edited character regions in it, and per-edit details.
    """
    if not isinstance(edits, list) or not edits or len(edits) > 30:
        raise WorkspaceError("edits 必须是包含 1 到 30 项的数组")
    planned: list[tuple[int, int, str, int]] = []
    details: list[dict[str, Any]] = []
    for index, edit in enumerate(edits, 1):
        if not isinstance(edit, dict):
            raise WorkspaceError(f"第 {index} 处修改不是对象；整个批次未修改")
        old = edit.get("old_text")
        new = edit.get("new_text")
        if not isinstance(old, str) or not old:
            raise WorkspaceError(f"第 {index} 处修改的 old_text 不能为空；整个批次未修改")
        if not isinstance(new, str):
            raise WorkspaceError(f"第 {index} 处修改缺少 new_text；整个批次未修改")
        try:
            spans, replacement, mode = _find_spans(content, old, new)
        except WorkspaceError as exc:
            raise WorkspaceError(f"第 {index} 处修改：{exc}；整个批次未修改") from None
        replace_all = bool(edit.get("replace_all", False))
        if len(spans) > 1 and not replace_all:
            starts = _line_starts(content)
            where = "、".join(str(_line_of_offset(starts, start)) for start, _ in spans[:10])
            raise WorkspaceError(
                f"第 {index} 处修改的 old_text 出现 {len(spans)} 次（起始行：{where}）；"
                "请加入更多上下文使其唯一，或设置 replace_all；整个批次未修改"
            )
        targets = spans if replace_all else spans[:1]
        planned.extend((start, end, replacement, index) for start, end in targets)
        details.append({"replacements": len(targets), "match": mode})
    ordered = sorted(planned, key=lambda item: (item[0], item[1]))
    for previous, current in zip(ordered, ordered[1:]):
        if current[0] < previous[1]:
            raise WorkspaceError(f"第 {previous[3]} 和第 {current[3]} 处修改的范围重叠；整个批次未修改")
    updated = content
    for start, end, replacement, _ in reversed(ordered):
        updated = updated[:start] + replacement + updated[end:]
    regions: list[tuple[int, int]] = []
    shift = 0
    for start, end, replacement, _ in ordered:
        new_start = start + shift
        regions.append((new_start, new_start + len(replacement)))
        shift += len(replacement) - (end - start)
    return updated, regions, details


def _find_spans(content: str, old: str, new: str) -> tuple[list[tuple[int, int]], str, str]:
    """Locate ``old`` with conservative recovery for common copy slips.

    Recovery is attempted only when the exact text is absent: copied ``N|``
    read_file prefixes are removed, then trailing whitespace and line endings
    are ignored line by line. Returns (spans, replacement, match mode).
    """
    attempts: list[tuple[str, str, str]] = [("exact", old, new)]
    old_lines = old.splitlines()
    if old_lines and all(_LINE_NUMBER_PREFIX.match(line) for line in old_lines):
        stripped_old = "\n".join(_LINE_NUMBER_PREFIX.sub("", line, count=1) for line in old_lines)
        new_lines = new.splitlines()
        stripped_new = (
            "\n".join(_LINE_NUMBER_PREFIX.sub("", line, count=1) for line in new_lines)
            if new_lines and all(_LINE_NUMBER_PREFIX.match(line) for line in new_lines)
            else new
        )
        if old.endswith("\n"):
            stripped_old += "\n"
        attempts.append(("line_numbers_removed", stripped_old, stripped_new))
    for mode, needle, replacement in attempts:
        starts: list[int] = []
        offset = 0
        while True:
            found = content.find(needle, offset)
            if found < 0:
                break
            starts.append(found)
            offset = found + len(needle)
        if starts:
            return [(start, start + len(needle)) for start in starts], replacement, mode
    # Line-wise match that ignores trailing whitespace and CR/LF differences.
    needle_lines = [line.rstrip() for line in attempts[-1][1].splitlines()]
    replacement = attempts[-1][2]
    if needle_lines and any(needle_lines):
        lines = content.splitlines(keepends=True)
        starts_at = _line_starts(content)
        spans: list[tuple[int, int]] = []
        size = len(needle_lines)
        index = 0
        while index + size <= len(lines):
            if all(lines[index + k].rstrip() == needle_lines[k] for k in range(size)):
                end = starts_at[index + size]
                last_line = lines[index + size - 1]
                if not attempts[-1][1].endswith(("\n", "\r")):
                    end -= len(last_line) - len(last_line.rstrip("\r\n"))
                spans.append((starts_at[index], end))
                index += size
            else:
                index += 1
        if spans:
            return spans, replacement, "whitespace_insensitive"
    hint = _closest_region_hint(content, attempts[-1][1])
    raise WorkspaceError(
        "old_text 与当前文件不匹配（已忽略行号前缀和行尾空白）"
        + (f"。{hint}" if hint else "；请确认片段与文件内容逐字一致")
    )


class ConversationWorkspace:
    def __init__(self, user_id: int, conversation_id: str) -> None:
        if not str(user_id).isdigit() or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", conversation_id or ""):
            raise WorkspaceError("无效的工作区标识")
        self.root = WORKSPACES_DIR / str(user_id) / conversation_id

    @staticmethod
    def _clean_path(value: Any, *, allow_root: bool = False) -> PurePosixPath:
        raw = str(value or "").strip()
        # Models often name the workspace root as "/" when searching it.
        if allow_root and raw in {"", ".", "/", "./"}:
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

    def tool_definitions(self, access: str = "full") -> list[dict[str, Any]]:
        """Return a byte-stable schema so provider prefix caches stay reusable.

        Existing paths are runtime state, not part of a tool's contract.  Putting
        them into JSON-schema enums made the entire tool prefix change after every
        write, invalidating provider prompt caches.  ``list_files`` remains the
        authoritative way for the model to discover paths.
        """
        tools = [*WORKSPACE_TOOLS, RUN_COMMAND_TOOL, RUN_PYTHON_TOOL, CHECK_WEB_SYNTAX_TOOL]
        if access == "read_only":
            tools = [item for item in tools if item["function"]["name"] in READ_ONLY_WORKSPACE_TOOL_NAMES]
        elif access == "edit":
            tools = [item for item in tools if item["function"]["name"] in EDIT_WORKSPACE_TOOL_NAMES]
        elif access != "full":
            raise WorkspaceError("无效的工作区访问模式")
        return deepcopy(tools)

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

    @staticmethod
    def _revision(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def read_snapshot(self, path: Any, start_line: Any = None) -> dict[str, Any]:
        content, relative = self._read_text(path)
        return {"path": relative, "revision": self._revision(content), **numbered_window(content, start_line)}

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

    def edit_file(self, path: Any, edits: Any) -> dict[str, Any]:
        content, relative = self._read_text(path)
        updated, regions, details = apply_text_edits(content, edits)
        result = self.write_file(relative, updated)
        result.update(
            {
                "edits": len(details),
                "replacements": sum(item["replacements"] for item in details),
                "revision": self._revision(updated),
                "line_count": len(updated.splitlines()),
                **edited_excerpt(updated, regions),
            }
        )
        recovered = sorted({item["match"] for item in details} - {"exact"})
        if recovered:
            result["match"] = recovered
        return result

    def run_command(self, command: Any, timeout_seconds: Any = None) -> dict[str, Any]:
        from .code_runner import run_command

        self.root.mkdir(parents=True, exist_ok=True)
        return run_command(self.root, command, timeout_seconds)

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
            result = self.read_snapshot(arguments.get("path"), arguments.get("start_line"))
        elif name == "write_file":
            result = self.write_file(arguments.get("path"), arguments.get("content"))
        elif name == "edit_file" or name in LEGACY_PATCH_TOOL_NAMES:
            result = self.edit_file(arguments.get("path"), edit_list(name, arguments))
        elif name == "search_files":
            result = self.search_files(arguments.get("query"), arguments.get("path", ""))
        elif name == "delete_file":
            result = self.delete_file(arguments.get("path"))
        elif name == "run_command":
            result = self.run_command(arguments.get("command"), arguments.get("timeout_seconds"))
        elif name == "run_python":
            result = self.run_python(arguments.get("path"), arguments.get("arguments", []))
        elif name == "check_web_syntax":
            result = self.check_web_syntax(arguments.get("path"))
        else:
            raise WorkspaceError(f"不支持的工作区工具：{name}")
        return json.dumps(result, ensure_ascii=False)


class AgentSharedWorkspace:
    """Authenticated UI view of the host-level Agent workspace.

    Agent mode deliberately does not use ``ConversationWorkspace``.  Its host
    tools operate directly on the shared ``/home/share`` directory, while this
    view provides the authenticated UI with safe listing, download, and
    explicit single-file deletion for that same directory.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or AGENT_WORKSPACE_ROOT)

    @staticmethod
    def _clean_path(value: Any, *, allow_root: bool = False) -> PurePosixPath:
        raw = str(value or "").strip()
        # Models often name the workspace root as "/" when searching it.
        if allow_root and raw in {"", ".", "/", "./"}:
            return PurePosixPath(".")
        if not raw or "\x00" in raw or "\\" in raw:
            raise WorkspaceError("文件路径无效")
        path = PurePosixPath(raw)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise WorkspaceError("只能使用 Agent 工作区内的相对路径")
        if len(path.as_posix()) > 500:
            raise WorkspaceError("文件路径过长")
        return path

    def resolve(self, value: Any, *, allow_root: bool = False) -> tuple[Path, str]:
        relative = self._clean_path(value, allow_root=allow_root)
        candidate = self.root if relative == PurePosixPath(".") else self.root.joinpath(*relative.parts)
        root_resolved = self.root.resolve(strict=False)
        resolved = candidate.resolve(strict=False)
        if resolved != root_resolved and root_resolved not in resolved.parents:
            raise WorkspaceError("文件路径越过 Agent 工作区边界")
        return candidate, "" if relative == PurePosixPath(".") else relative.as_posix()

    def list_files(self) -> list[dict[str, Any]]:
        if not self.root.is_dir() or self.root.is_symlink():
            return []
        result: list[dict[str, Any]] = []
        for current, directories, files in os.walk(self.root, followlinks=False):
            current_path = Path(current)
            directories[:] = [
                directory for directory in directories
                if not (current_path / directory).is_symlink()
            ]
            for filename in files:
                path = current_path / filename
                if path.is_symlink() or not path.is_file():
                    continue
                try:
                    size = path.stat().st_size
                    relative = path.relative_to(self.root).as_posix()
                except (OSError, ValueError):
                    continue
                result.append({"path": relative, "size": size, "backend": "agent"})
                if len(result) >= AGENT_MAX_FILES:
                    return sorted(result, key=lambda item: item["path"].casefold())
        return sorted(result, key=lambda item: item["path"].casefold())

    def resolve_file(self, path: Any) -> tuple[Path, str]:
        target, relative = self.resolve(path)
        if not target.is_file() or target.is_symlink():
            raise WorkspaceError("Agent 工作区文件不存在")
        return target, relative

    def delete_file(self, path: Any) -> dict[str, Any]:
        target, relative = self.resolve_file(path)
        try:
            target.unlink()
        except FileNotFoundError as exc:
            raise WorkspaceError("Agent 工作区文件不存在") from exc
        except OSError as exc:
            raise WorkspaceError(f"删除 Agent 工作区文件失败：{exc}") from exc
        return {"ok": True, "path": relative}


def delete_conversation_workspace(user_id: int, conversation_id: str) -> None:
    workspace = ConversationWorkspace(user_id, conversation_id)
    if workspace.root.is_dir():
        shutil.rmtree(workspace.root)


def delete_user_workspaces(user_id: int) -> None:
    root = WORKSPACES_DIR / str(user_id)
    if root.is_dir():
        shutil.rmtree(root)

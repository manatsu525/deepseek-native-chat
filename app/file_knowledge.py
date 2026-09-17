"""Track which file contents the model can currently see during one answer.

A read is only worth its tokens when it tells the model something new. This
tracker records, per path, the revision and line ranges whose content is
present in the model's context: from a read, from a file the model wrote
itself, or from a fully read file that the model then changed with its own
edits (each edit result shows the edited regions). A later read of content
that is still visible returns a short "unchanged" note instead of the file.

Visibility must match the real request. Entries whose content is dropped by a
context checkpoint are forgotten, and a checkpoint carries the current
content of the entries it keeps. If a model asks for the same unchanged
content twice in a row it evidently lost track, so the full content is
returned the second time rather than looping.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

REPEATED_READ_WARNING_COUNT = 3


class FileKnowledge:
    def __init__(self, validate: Callable[[str], str | None] | None = None) -> None:
        # path -> {"revision", "line_count", "ranges": [[from, through, content]],
        #          "edited": bool, "duplicates": int, "reads": int}
        self._entries: dict[str, dict[str, Any]] = {}
        self._validate = validate
        self.version = 0

    def _touch(self, path: str, entry: dict[str, Any]) -> None:
        self._entries.pop(path, None)
        self._entries[path] = entry
        self.version += 1

    def forget(self, path: str) -> None:
        prefix = path.rstrip("/") + "/"
        for key in [key for key in self._entries if key == path or key.startswith(prefix)]:
            del self._entries[key]
            self.version += 1

    def known(self, path: str) -> bool:
        return path in self._entries

    @staticmethod
    def _complete(entry: dict[str, Any]) -> bool:
        return any(first <= 1 and through >= entry["line_count"] for first, through, _ in entry["ranges"])

    def record_read(self, data: dict[str, Any]) -> str | None:
        """Register a read result; return replacement text if it adds nothing new."""
        path = str(data.get("path") or "")
        revision = str(data.get("revision") or "")
        if not path or not revision or "content" not in data:
            return None
        first = int(data.get("from_line") or 0)
        through = int(data.get("through_line") or 0)
        line_count = int(data.get("line_count") or 0)
        entry = self._entries.get(path)
        if entry is None or entry["revision"] != revision:
            entry = {"revision": revision, "line_count": line_count, "ranges": [], "edited": False,
                     "duplicates": 0, "reads": 0}
        entry["reads"] += 1
        covered = any(
            have_first <= first and have_through >= through for have_first, have_through, _ in entry["ranges"]
        )
        if covered and entry["duplicates"] == 0:
            entry["duplicates"] = 1
            self._entries[path] = entry
            return json.dumps(self._unchanged(path, entry), ensure_ascii=False)
        # New content, or a second consecutive duplicate: return the real text.
        entry["duplicates"] = 0
        entry["ranges"] = [
            item for item in entry["ranges"] if not (first <= item[0] and through >= item[1])
        ] + [[first, through, str(data.get("content") or "")]]
        self._touch(path, entry)
        return None

    def record_own_change(self, data: dict[str, Any] | None, *, visible: bool, created: bool = False) -> None:
        """Update after the model's own write/edit, from a fresh read of the file.

        ``created`` is a full write: the model authored the whole content. An
        edit keeps the file visible only if it was fully visible beforehand.
        """
        if not data:
            return
        path = str(data.get("path") or "")
        previous = self._entries.get(path)
        complete_now = not data.get("truncated")
        if not visible or not complete_now or (not created and (previous is None or not self._complete(previous))):
            self.forget(path)
            return
        self._touch(path, {
            "revision": str(data.get("revision") or ""),
            "line_count": int(data.get("line_count") or 0),
            "ranges": [[int(data.get("from_line") or 0), int(data.get("through_line") or 0), str(data.get("content") or "")]],
            "edited": not created,
            "duplicates": 0,
            "reads": 0,
        })

    def _unchanged(self, path: str, entry: dict[str, Any]) -> dict[str, Any]:
        if entry["edited"]:
            message = (
                "文件内容你已经看过：上次完整读取后只有你自己的修改，修改后的片段见各次编辑结果的 updated_excerpt，"
                "其余部分与之前读到的一致。本次不再重复返回全文，请直接继续修改或回答。"
            )
        else:
            message = "文件自上次读取后未改变，内容已在上下文中（或 CONTEXT CHECKPOINT 的 file_snapshots 里），本次不再重复返回全文。请直接继续修改或回答。"
        result: dict[str, Any] = {
            "ok": True,
            "path": path,
            "revision": entry["revision"],
            "unchanged": True,
            "line_count": entry["line_count"],
            "message": message,
        }
        if entry["reads"] >= REPEATED_READ_WARNING_COUNT:
            result["warning"] = (
                f"这是第 {entry['reads']} 次读取同一版本的文件，重复读取不会带来新信息；"
                "请立即修改文件，或者结束工具调用并回答用户。"
            )
        return result

    def snapshots(self, budget: int) -> list[dict[str, Any]]:
        """Current content for a checkpoint, newest first within ``budget``.

        Entries that do not fit, or whose file changed behind our back, are
        forgotten because their content is no longer in the request.
        """
        kept: list[dict[str, Any]] = []
        used = 0
        for path, entry in reversed(list(self._entries.items())):
            if self._validate is not None and self._validate(path) != entry["revision"]:
                del self._entries[path]
                self.version += 1
                continue
            for first, through, content in sorted(entry["ranges"]):
                snapshot = {"path": path, "revision": entry["revision"], "line_count": entry["line_count"],
                            "from_line": first, "through_line": through, "content": content}
                size = len(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")))
                if used + size > budget:
                    entry["ranges"] = [item for item in entry["ranges"] if item[0] != first or item[1] != through]
                    continue
                kept.append(snapshot)
                used += size
            if not entry["ranges"]:
                del self._entries[path]
                self.version += 1
        kept.reverse()
        return kept

"""Durable Agent execution events, separate from streamed UI and plan state.

Requests and unsuccessful attempts are diagnostic events. Only committed
messages/results and explicit context projections contribute to model history.
No credentials/HTTP headers are recorded. Existing ordinary chat is unchanged.
"""
from __future__ import annotations

import copy
import json
import time
from typing import Any

from .db import Database


LEGACY_WEB_EVIDENCE_PREFIX = (
    "\n\n---\n[Context supplied by the application, not written by the user]\n"
    "WEB EVIDENCE FROM THIS CONVERSATION:\n"
)


def remove_legacy_web_evidence(messages: list[dict[str, Any]], originals: set[str]) -> None:
    """Clean generated suffixes in Agent replay, preserving original user text."""
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content not in originals:
            original, marker, _ = content.partition(LEGACY_WEB_EVIDENCE_PREFIX)
            if marker and original in originals:
                message["content"] = original
        elif isinstance(content, list):
            if not any(isinstance(p, dict) and p.get("type") == "text" and p.get("text") in originals for p in content):
                continue
            label = LEGACY_WEB_EVIDENCE_PREFIX.lstrip("\n")
            message["content"] = [p for p in content if not (
                isinstance(p, dict) and p.get("type") == "text"
                and str(p.get("text") or "").startswith(label) and p.get("text") not in originals)]


def project_history(events: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    messages = None
    for event in events:
        payload = event["payload"]
        if event["kind"] in {"history/start", "context/checkpoint"}:
            messages = copy.deepcopy(payload["messages"])
        elif event["kind"] in {"assistant/message", "tool/result"} and messages is not None:
            messages.append(copy.deepcopy(payload["message"]))
    if messages is None:
        return None
    # A started side effect without a durable result has an UNKNOWN outcome,
    # not a failure. Balance the protocol without pretending it is safe to repeat.
    repaired = []
    started = {event["payload"].get("call_id") for event in events if event["kind"] == "tool/start"}
    index = 0
    while index < len(messages):
        message = messages[index]
        repaired.append(message)
        index += 1
        calls = message.get("tool_calls") or []
        if not calls:
            continue
        results = {}
        while index < len(messages) and messages[index].get("role") == "tool":
            result = messages[index]
            results[result.get("tool_call_id")] = result
            index += 1
        for call in calls:
            call_id = call["id"]
            repaired.append(results.get(call_id) or {
                "role": "tool", "tool_call_id": call_id,
                "content": ("执行中断，未持久化此调用结果，结果未知；操作可能已经生效。先检查当前文件或外部状态，再决定是否需要重做。"
                            if call_id in started else "执行中断，此调用尚未开始，未执行。"),
            })
    return repaired


class AgentJournal:
    def __init__(self, db: Database, job_id: str, conversation_id: str):
        self.db, self.job_id, self.conversation_id = db, job_id, conversation_id
        self.pending_preview: dict[str, Any] | None = None

    def append(self, kind: str, payload: dict[str, Any]) -> None:
        if kind == "model/preview":
            # Keep the latest draft in memory, not one SQLite row per token.
            self.pending_preview = copy.deepcopy(payload)
            return
        if kind == "model/error":
            self.flush_preview()
        if kind in {"model/output", "model/attempt_failed"}:
            self.pending_preview = None
        # Serialize immediately: later payload mutation (fallback, compaction,
        # argument normalization) must not rewrite the recorded request.
        self.db.run(
            "INSERT INTO agent_events(job_id,conversation_id,kind,payload_json,created_at) VALUES(?,?,?,?,?)",
            (self.job_id, self.conversation_id, kind, json.dumps(payload, ensure_ascii=False), int(time.time())),
        )

    def flush_preview(self) -> None:
        if self.pending_preview is not None:
            self.append("model/attempt_failed", self.pending_preview)

    def events(self) -> list[dict[str, Any]]:
        return [{"id": row["id"], "kind": row["kind"], "payload": json.loads(row["payload_json"]),
                 "created_at": row["created_at"]}
                for row in self.db.all("SELECT * FROM agent_events WHERE job_id=? AND conversation_id=? ORDER BY id",
                                       (self.job_id, self.conversation_id))]

    def history(self, *, scope: str = "") -> list[dict[str, Any]] | None:
        checkpoint = self.db.one(
            "SELECT MAX(id) AS id FROM agent_events WHERE job_id=? AND conversation_id=? "
            "AND kind IN ('history/start','context/checkpoint')", (self.job_id, self.conversation_id))
        if not checkpoint or checkpoint["id"] is None:
            return None
        rows = self.db.all(
            "SELECT kind,payload_json FROM agent_events WHERE job_id=? AND conversation_id=? AND id>=? "
            "AND kind IN ('history/start','context/checkpoint','assistant/message','tool/result','tool/start') ORDER BY id",
            (self.job_id, self.conversation_id, checkpoint["id"]))
        events = [{"kind": row["kind"], "payload": json.loads(row["payload_json"])} for row in rows]
        history = project_history(events)
        if history is None:
            return None
        if any("WEB EVIDENCE FROM THIS CONVERSATION:" in str(m.get("content") or "")
               for m in history if m.get("role") == "user"):
            originals = {row["content"] for row in self.db.all(
                "SELECT content FROM messages WHERE conversation_id=? AND role='user'", (self.conversation_id,))}
            remove_legacy_web_evidence(history, originals)
        source = self.db.one(
            "SELECT payload_json FROM agent_events WHERE job_id=? AND conversation_id=? AND kind='turn/start' "
            "AND id<=? ORDER BY id DESC LIMIT 1", (self.job_id, self.conversation_id, checkpoint["id"]))
        previous_scope = json.loads(source["payload_json"]).get("scope") if source else None
        if previous_scope != scope:
            # Provider-native reasoning signatures/items cannot be transferred
            # to another route/model. The text and tool transcript still can.
            for message in history:
                for key in ("reasoning_content", "anthropic_thinking_blocks", "responses_output_items"):
                    message.pop(key, None)
        return history

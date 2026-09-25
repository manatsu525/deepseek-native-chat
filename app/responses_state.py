"""Responses continuation with a local history retained for compatibility."""
from __future__ import annotations

import hashlib
import json
from contextlib import asynccontextmanager
from typing import Any


def state_scope(provider: dict, job: dict, settings: dict) -> str:
    # Never persist credentials themselves. Changing any routing/configuration
    # input starts a new chain, as does switching user, conversation or mode.
    value = [provider.get("id"), provider.get("base_url"), provider.get("api_key"),
             job.get("user_id"), job.get("conversation_id"), job.get("model"),
             job.get("chat_mode"), job.get("effort"), job.get("timezone"), settings]
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def resume_state(rows: list[dict], scope: str) -> dict:
    # Rows are newest first. A retry deletes its answer and all descendants,
    # naturally leaving the correct parent. Never jump across failed answers.
    if len(rows) < 2 or rows[0]["role"] != "user" or rows[1]["role"] != "assistant":
        return {}
    meta = json.loads(rows[1].get("meta_json") or "{}")
    state = meta.get("responses_state") or {}
    if meta.get("failed") or meta.get("stopped") or state.get("scope") != scope:
        return {}
    return state


class ResponsesState:
    def __init__(self, saved: dict | None = None):
        saved = saved or {}
        self.disabled = bool(saved.get("disabled"))
        self.previous_id = str(saved.get("response_id") or "") if not self.disabled else ""
        self.pending: list[dict[str, Any]] | None = None
        self.candidate = ""
        self.candidate_stored = True
        self.reason = str(saved.get("fallback_reason") or "")
        self.chained_requests = 0

    def prepare(self, payload: dict, full_input: list[dict]) -> None:
        if "previous_response_id" in payload or "conversation" in payload:
            raise ValueError("Responses 会话 ID 由服务端自动管理，请移除高级 JSON 中的 previous_response_id/conversation")
        self.candidate = ""
        self.candidate_stored = True
        if payload.get("store") is False:
            self.disable("store_false")
        if self.disabled:
            payload["input"] = full_input
            return
        payload["store"] = True
        if self.previous_id and self.pending is not None:
            payload["previous_response_id"] = self.previous_id
            payload["input"] = self.pending
            self.chained_requests += 1
        else:
            payload["input"] = full_input

    def observe(self, event: dict) -> None:
        # A created ID alone does not establish a completed, reusable response.
        if event.get("type") == "response.completed":
            response = event.get("response") or {}
            self.candidate = str(response.get("id") or "")
            self.candidate_stored = response.get("store") is not False

    def accept(self) -> None:
        if self.disabled:
            return
        if not self.candidate or not self.candidate_stored:
            self.disable("upstream_missing_stored_response_id")
        else:
            self.previous_id = self.candidate

    def reset(self) -> None:
        self.previous_id = ""
        self.pending = None
        self.candidate = ""

    def disable(self, reason: str) -> None:
        self.reset()
        self.disabled = True
        self.reason = reason

    def export(self) -> dict:
        return {"response_id": self.previous_id, "disabled": self.disabled,
                "fallback_reason": self.reason, "chained_requests": self.chained_requests}

    @asynccontextmanager
    async def stream(
        self,
        client,
        method,
        url,
        *,
        headers,
        json,
        full_input,
        retry_status_codes: set[int] | None = None,
    ):
        # Retry once only for explicit state-field rejection, before consuming
        # any generated output. Configured HTTP statuses are yielded to the
        # shared retry controller so Responses does not swallow them first.
        retry_status_codes = set(retry_status_codes or ())
        for attempt in range(2):
            async with client.stream(method, url, headers=headers, json=json) as response:
                # Let the shared retry controller observe every configured
                # status. Other HTTP errors remain handled here because
                # Responses state fallback needs to inspect their body.
                if response.status_code in retry_status_codes:
                    yield response
                    return
                if response.status_code >= 400:
                    body = (await response.aread()).decode(errors="replace")[:2000]
                    lowered = body.lower()
                    field_error = any(word in lowered for word in (
                        "previous_response_id", "previous response", "store", "storage"))
                    rejected = any(word in lowered for word in (
                        "not found", "not supported", "unsupported", "unknown", "unrecognized",
                        "invalid", "expired", "not allowed", "not permitted", "does not exist"))
                    if (attempt == 0 and not self.disabled and response.status_code in (400, 404, 422)
                            and field_error and rejected):
                        self.disable("upstream_rejected_response_state")
                        json.pop("previous_response_id", None)
                        json.pop("store", None)
                        json["input"] = full_input
                        continue
                    raise RuntimeError(f"Custom API {response.status_code}: {body}")
                yield response
                return

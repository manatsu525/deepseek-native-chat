from __future__ import annotations

import asyncio
import io
import json
import os
import re
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
import uvicorn
from fastapi import Cookie, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import attachments
from .config import settings
from .db import Database
from .deepseek import list_models as deepseek_list_models
from .deepseek import stream_response as deepseek_stream_response
from .mimo import DEFAULT_SETTINGS as CUSTOM_DEFAULT_SETTINGS
from .mimo import MIMO_MAX_COMPLETION_TOKENS, custom_auth_headers, is_mimo_model, list_models as custom_list_models
from .mimo_local import stream_response as custom_stream_response
from .multi_agent import run_collaboration
from .reasoning_effort import DEFAULT as DEFAULT_REASONING_EFFORT
from .reasoning_effort import LEVELS as REASONING_EFFORT_LEVELS
from .security import load_secret, make_token, password_hash, password_ok, read_token
from .workspace import ConversationWorkspace, WorkspaceError, delete_conversation_workspace, delete_user_workspaces


db = Database(settings.db_path)
secret = b""
tasks: dict[str, asyncio.Task[Any]] = {}
attachment_cleanup_task: Optional[asyncio.Task[Any]] = None
attachment_upload_locks: dict[int, asyncio.Lock] = {}
attachment_processing_lock = asyncio.Lock()
attachment_job_lock = asyncio.Lock()
SUPPORTED_MODELS = {
    "deepseek": {"deepseek-v4-flash", "deepseek-v4-pro"},
    # Custom providers advertise their own model IDs through /models or a
    # manually entered model name, so there is no static allow-list here.
    "custom": set(),
}
DEFAULT_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "custom": "https://api.openai.com/v1",
}


class LoginBody(BaseModel):
    username: str
    password: str


class UserBody(BaseModel):
    username: str = Field(min_length=2, max_length=32)
    password: str = Field(min_length=8, max_length=128)
    is_admin: bool = False


class PasswordBody(BaseModel):
    password: str = Field(min_length=8, max_length=128)


class ProviderBody(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    api_key: str = Field(min_length=8, max_length=300)
    provider_type: Literal["deepseek", "custom"] = "deepseek"
    base_url: str = ""
    model: str = ""
    selected_models: list[str] = Field(default_factory=list, max_length=500)
    manual_models: Optional[list[str]] = Field(default=None, max_length=500)
    custom_settings: Optional[dict[str, Any]] = None


class ProviderModelsBody(BaseModel):
    model: str = ""
    selected_models: list[str] = Field(default_factory=list, max_length=500)
    manual_models: Optional[list[str]] = Field(default=None, max_length=500)


class CustomSettingsBody(BaseModel):
    thinking: Literal["enabled", "disabled"] = "enabled"
    reasoning_effort_enabled: bool = True
    dsml_fallback_enabled: bool = False
    max_completion_tokens: int = Field(default=65536, ge=256, le=MIMO_MAX_COMPLETION_TOKENS)
    temperature: float = Field(default=1.0, ge=0, le=1.5)
    top_p: float = Field(default=0.95, ge=0.01, le=1)
    web_tool_backend: Literal["parallel", "keenable", "tavily", "firecrawl", "you", "legacy"] = "parallel"


class CustomModelSettingsBody(CustomSettingsBody):
    model: str = Field(min_length=1, max_length=300)


class ChatBody(BaseModel):
    conversation_id: Optional[str] = None
    content: str = Field(default="", max_length=100_000)
    attachment_ids: list[str] = Field(default_factory=list, max_length=attachments.MAX_ATTACHMENTS)
    provider_id: int
    model: str = ""
    effort: str = DEFAULT_REASONING_EFFORT
    timezone: str = Field(default="UTC", min_length=1, max_length=64)
    chat_mode: Literal["standard", "multi_agent"] = "standard"


class RetryBody(BaseModel):
    prompt_message_id: int = Field(gt=0)
    provider_id: int
    model: str = ""
    effort: str = DEFAULT_REASONING_EFFORT
    timezone: str = Field(default="UTC", min_length=1, max_length=64)
    chat_mode: Literal["standard", "multi_agent"] = "standard"


class PinBody(BaseModel):
    pinned: bool


def normalize_custom_settings(value: Any = None) -> dict[str, Any]:
    data = dict(CUSTOM_DEFAULT_SETTINGS)
    if isinstance(value, CustomSettingsBody):
        data.update(value.model_dump())
    elif isinstance(value, dict):
        data.update(value)
    return CustomSettingsBody(**data).model_dump()


def _decoded_provider_settings(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("settings_json", {})
    decoded = value if isinstance(value, dict) else db.decode(str(value or "{}"), {})
    return decoded if isinstance(decoded, dict) else {}


def custom_settings_by_model(row: dict[str, Any], models: Optional[list[str]] = None) -> dict[str, dict[str, Any]]:
    """Return one independent, normalized Custom configuration per model.

    Older providers stored one configuration at the top level. Until they are
    persisted in the new format, treat that legacy configuration as the seed
    for each already-enabled model.
    """
    saved = _decoded_provider_settings(row)
    selected = models if models is not None else _models_from_settings(row.get("model"), saved)
    raw_by_model = saved.get("model_settings")
    has_model_map = isinstance(raw_by_model, dict)
    legacy = {key: saved[key] for key in CUSTOM_DEFAULT_SETTINGS if key in saved}
    result: dict[str, dict[str, Any]] = {}
    for model in selected:
        raw = raw_by_model.get(model) if has_model_map else legacy
        result[model] = normalize_custom_settings(raw if isinstance(raw, dict) else None)
    return result


def custom_settings_document(row: dict[str, Any], models: Optional[list[str]] = None) -> dict[str, Any]:
    selected = models if models is not None else provider_models(row)
    return {"models": selected, "model_settings": custom_settings_by_model(row, selected)}


def custom_settings_for_model(row: dict[str, Any], model: str) -> dict[str, Any]:
    return custom_settings_by_model(row, [model])[model]


def migrate_custom_provider_settings() -> None:
    """Persist the former API-wide settings as independent per-model values."""
    for provider in db.all("SELECT * FROM providers WHERE provider_type='custom'"):
        before = _decoded_provider_settings(provider)
        after = custom_settings_document(provider)
        if before != after:
            db.run(
                "UPDATE providers SET settings_json=? WHERE id=?",
                (json.dumps(after, ensure_ascii=False), provider["id"]),
            )


# Kept as a source-compatible alias for older integrations importing the
# original class/function names.
MimoSettingsBody = CustomSettingsBody
normalize_mimo_settings = normalize_custom_settings


def provider_type(row: dict[str, Any]) -> str:
    kind = row.get("provider_type") or "deepseek"
    return "custom" if kind == "mimo" else kind


def _clean_model_ids(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple, set)):
        return []
    models = list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))
    if any(len(item) > 300 for item in models):
        raise HTTPException(400, "模型名称过长")
    if len(models) > 500:
        raise HTTPException(400, "一次最多保存 500 个模型")
    return models


def _models_from_settings(model: Any, settings_json: Any) -> list[str]:
    settings_value = settings_json if isinstance(settings_json, dict) else db.decode(str(settings_json or "{}"), {})
    selected = _clean_model_ids(settings_value.get("models", [])) if isinstance(settings_value, dict) else []
    fallback = str(model or "").strip()
    if fallback and fallback not in selected:
        selected.insert(0, fallback)
    return selected


def provider_models(row: dict[str, Any]) -> list[str]:
    return _models_from_settings(row.get("model"), row.get("settings_json", "{}"))


def validate_provider_selection(kind: str, model: str, provider: Optional[dict[str, Any]] = None) -> None:
    if kind not in SUPPORTED_MODELS:
        raise HTTPException(400, "不支持的服务商类型")
    if kind == "custom":
        if not model.strip():
            raise HTTPException(400, "请选择或填写一个 custom 模型")
        if provider is not None and model not in provider_models(provider):
            raise HTTPException(400, "该模型未在此 custom API 配置中启用")
        return
    if model not in SUPPORTED_MODELS[kind]:
        available = "、".join(sorted(SUPPORTED_MODELS[kind]))
        raise HTTPException(400, f"{kind} 当前支持的模型为：{available}")


def now() -> int:
    return int(time.time())


def clean_base_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if not value.startswith(("https://", "http://")):
        raise HTTPException(400, "API 地址必须使用 http:// 或 https://")
    return value


def clean_timezone(value: str) -> str:
    name = str(value or "UTC").strip()
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise HTTPException(400, "无效的浏览器时区") from exc
    return name


def validate_effort(value: str) -> str:
    if value not in REASONING_EFFORT_LEVELS:
        raise HTTPException(400, "无效的思考深度")
    return value


def public_conversation(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    item["pinned"] = item.get("pinned_at") is not None
    return item


def trim_old_conversations(user_id: int) -> None:
    """Keep the newest 100 regular chats without ever pruning pinned chats."""
    with db.lock, db.connect() as connection:
        rows = connection.execute(
            """SELECT id FROM conversations
               WHERE user_id=? AND pinned_at IS NULL
               ORDER BY updated_at DESC LIMIT -1 OFFSET 100""",
            (user_id,),
        ).fetchall()
        if rows:
            connection.executemany("DELETE FROM conversations WHERE id=? AND user_id=?", [(row["id"], user_id) for row in rows])
    for row in rows:
        delete_conversation_workspace(user_id, row["id"])


def current_user(session: Optional[str] = Cookie(default=None)) -> dict[str, Any]:
    if not session:
        raise HTTPException(401, "请登录")
    user_id = read_token(session, secret)
    user = db.one("SELECT id, username, is_admin, created_at FROM users WHERE id=?", (user_id,)) if user_id else None
    if not user:
        raise HTTPException(401, "登录已失效")
    user["is_admin"] = bool(user["is_admin"])
    return user


def admin_user(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    if not user["is_admin"]:
        raise HTTPException(403, "仅管理员可用")
    return user


def public_provider(row: dict[str, Any]) -> dict[str, Any]:
    saved_settings = _decoded_provider_settings(row)
    saved_models = _models_from_settings(row.get("model"), saved_settings)
    key = row.pop("api_key", "")
    row["provider_type"] = provider_type(row)
    if row["provider_type"] == "custom":
        row["model_settings"] = custom_settings_by_model(row, saved_models)
        # Keep the old field useful for clients that have not yet learned the
        # per-model response shape. It represents only the provider's primary
        # model and is no longer used by this web client.
        row["settings"] = row["model_settings"].get(row.get("model"), normalize_custom_settings())
    else:
        row["model_settings"] = {}
        row["settings"] = saved_settings
    row["models"] = saved_models
    row.pop("settings_json", None)
    row["api_key_masked"] = (key[:3] + "••••" + key[-4:]) if len(key) > 8 else "••••••••"
    return row


def public_job(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    for name, fallback in (("searches_json", []), ("sources_json", []), ("usage_json", {}), ("agents_json", [])):
        row[name.removesuffix("_json")] = db.decode(row.pop(name, ""), fallback)
    row["stop_requested"] = bool(row["stop_requested"])
    if row.get("provider_type") == "mimo":
        row["provider_type"] = "custom"
    if row.get("status") in {"completed", "failed", "stopped"} and row.get("user_id") and row.get("conversation_id"):
        row["workspace_files"] = ConversationWorkspace(row["user_id"], row["conversation_id"]).list_files()
    return row


def public_attachment(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["original_name"],
        "kind": row["kind"],
        "media_type": row["media_type"],
        "size": int(row["original_size"]),
        "processed_size": int(row["processed_size"]),
    }


async def periodic_attachment_cleanup() -> None:
    while True:
        try:
            expired = await asyncio.to_thread(
                db.cleanup_expired_attachments,
                now() - attachments.ATTACHMENT_TTL_SECONDS,
            )
            await asyncio.to_thread(attachments.delete_files, expired)
            active_paths = await asyncio.to_thread(db.all_attachment_paths)
            await asyncio.to_thread(attachments.cleanup_orphan_files, active_paths)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Cleanup is best-effort and must never take the chat service down.
            pass
        await asyncio.sleep(6 * 60 * 60)


def title_for(text: str) -> str:
    compact = " ".join(text.split())
    return compact[:36] + ("…" if len(compact) > 36 else "")


async def run_job(job_id: str) -> None:
    job = db.one("SELECT * FROM jobs WHERE id=?", (job_id,))
    if not job:
        return
    attachment_records = db.attachments_for_job(job["user_id"], job_id)
    provider = db.one("SELECT * FROM providers WHERE id=? AND user_id=?", (job["provider_id"], job["user_id"]))
    if not provider:
        db.update_job(job_id, status="failed", error="API 配置不存在")
        if attachment_records:
            deleted = db.delete_attachments(job["user_id"], [item["id"] for item in attachment_records])
            await asyncio.to_thread(attachments.delete_files, deleted)
        return
    kind = provider_type(provider)
    job_workspace = ConversationWorkspace(job["user_id"], job["conversation_id"])
    history_rows = db.all(
        "SELECT role, content, meta_json FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT 20",
        (job["conversation_id"],),
    )
    history: list[dict[str, Any]] = []
    for row in reversed(history_rows):
        meta = db.decode(row.get("meta_json", "{}"), {})
        # Failed answers are kept for the user to inspect, but an incomplete
        # status sentence must not pollute the next model request's context.
        if row["role"] == "assistant" and meta.get("failed"):
            continue
        message: dict[str, Any] = {"role": row["role"], "content": row["content"]}
        # Custom Chat Completions gateways may accept historical reasoning_content
        # in later turns. Client-side
        # tool messages are intentionally not replayed here: the final assistant
        # message is persisted, while replaying an assistant tool_call without
        # its matching tool result can make compatible gateways reject history.
        if kind == "custom" and is_mimo_model(job.get("model")) and row["role"] == "assistant":
            if meta.get("reasoning") and not meta.get("invalid_answer"):
                message["reasoning_content"] = meta.get("reasoning", "")
        history.append(message)
    db.update_job(job_id, status="running", error="", stop_requested=0)
    last_write = 0.0
    attachment_lock_acquired = False

    def stopped() -> bool:
        state = db.one("SELECT stop_requested FROM jobs WHERE id=?", (job_id,))
        return not state or bool(state["stop_requested"])

    async def update(state: dict[str, Any]) -> None:
        nonlocal last_write
        stamp = time.monotonic()
        if stamp - last_write < 0.35 and not state.get("usage"):
            return
        last_write = stamp
        db.update_job(
            job_id,
            answer=state["answer"],
            reasoning=state["reasoning"],
            searches_json=json.dumps(state["searches"], ensure_ascii=False),
            sources_json=json.dumps(state["sources"], ensure_ascii=False),
            usage_json=json.dumps(state["usage"], ensure_ascii=False),
            agents_json=json.dumps(state.get("agents", []), ensure_ascii=False),
        )

    try:
        if attachment_records:
            await attachment_job_lock.acquire()
            attachment_lock_acquired = True
            history = await asyncio.to_thread(
                attachments.build_model_messages,
                history,
                attachment_records,
                kind == "custom",
            )
        if kind == "custom" and job.get("chat_mode") == "multi_agent":
            provider_settings = custom_settings_for_model(provider, job["model"])
            result = await run_collaboration(
                base_url=provider["base_url"],
                api_key=provider["api_key"],
                model=job["model"],
                messages=history,
                timeout=settings.request_timeout,
                stopped=stopped,
                update=update,
                settings=provider_settings,
                conversation_id=job["conversation_id"],
                user_timezone=job.get("timezone") or "UTC",
                effort=job["effort"],
                workspace=job_workspace,
            )
        elif kind == "custom":
            provider_settings = custom_settings_for_model(provider, job["model"])
            result = await custom_stream_response(
                base_url=provider["base_url"],
                api_key=provider["api_key"],
                model=job["model"],
                messages=history,
                timeout=settings.request_timeout,
                stopped=stopped,
                update=update,
                settings=provider_settings,
                conversation_id=job["conversation_id"],
                user_timezone=job.get("timezone") or "UTC",
                effort=job["effort"],
                workspace=job_workspace,
            )
        else:
            result = await deepseek_stream_response(
                base_url=provider["base_url"],
                api_key=provider["api_key"],
                model=job["model"],
                messages=history,
                effort=job["effort"],
                timeout=settings.request_timeout,
                stopped=stopped,
                update=update,
            )
        meta = {"job_id": job_id, "conversation_id": job["conversation_id"], "provider_id": job["provider_id"], "provider_type": kind, "model": job["model"], "chat_mode": job.get("chat_mode") or "standard", "reasoning": result["reasoning"], "searches": result["searches"], "sources": result["sources"], "usage": result["usage"], "agents": result.get("agents", []), "workspace_files": job_workspace.list_files()}
        if result.get("tool_trace"):
            meta["tool_trace"] = result["tool_trace"]
        db.run(
            "INSERT INTO messages(conversation_id, role, content, meta_json, created_at) VALUES(?,?,?,?,?)",
            (job["conversation_id"], "assistant", result["answer"], json.dumps(meta, ensure_ascii=False), now()),
        )
        db.run("UPDATE conversations SET updated_at=? WHERE id=?", (now(), job["conversation_id"]))
        db.update_job(
            job_id,
            status="completed",
            answer=result["answer"],
            reasoning=result["reasoning"],
            searches_json=json.dumps(result["searches"], ensure_ascii=False),
            sources_json=json.dumps(result["sources"], ensure_ascii=False),
            usage_json=json.dumps(result["usage"], ensure_ascii=False),
            agents_json=json.dumps(result.get("agents", []), ensure_ascii=False),
        )
    except asyncio.CancelledError:
        partial = db.one("SELECT answer, reasoning, searches_json, sources_json, usage_json, agents_json FROM jobs WHERE id=?", (job_id,)) or {}
        partial_agents = db.decode(partial.get("agents_json", "[]"), [])
        if partial.get("answer") or partial_agents:
            meta = {
                "job_id": job_id,
                "stopped": True,
                "provider_id": job["provider_id"],
                "provider_type": provider_type(provider),
                "model": job["model"],
                "chat_mode": job.get("chat_mode") or "standard",
                "reasoning": partial.get("reasoning", ""),
                "searches": db.decode(partial.get("searches_json", "[]"), []),
                "sources": db.decode(partial.get("sources_json", "[]"), []),
                "usage": db.decode(partial.get("usage_json", "{}"), {}),
                "agents": partial_agents,
            }
            db.run(
                "INSERT INTO messages(conversation_id, role, content, meta_json, created_at) VALUES(?,?,?,?,?)",
                (job["conversation_id"], "assistant", str(partial.get("answer") or "") + "\n\n_已停止生成_", json.dumps(meta, ensure_ascii=False), now()),
            )
        db.update_job(job_id, status="stopped", error="")
    except Exception as exc:
        error = str(exc)[:3000]
        partial = db.one(
            "SELECT answer,reasoning,searches_json,sources_json,usage_json,agents_json FROM jobs WHERE id=?",
            (job_id,),
        ) or {}
        meta = {
            "job_id": job_id,
            "conversation_id": job["conversation_id"],
            "failed": True,
            "invalid_answer": True,
            "error": error,
            "provider_id": job["provider_id"],
            "provider_type": kind,
            "model": job["model"],
            "chat_mode": job.get("chat_mode") or "standard",
            "reasoning": partial.get("reasoning", ""),
            "searches": db.decode(partial.get("searches_json", "[]"), []),
            "sources": db.decode(partial.get("sources_json", "[]"), []),
            "usage": db.decode(partial.get("usage_json", "{}"), {}),
            "agents": db.decode(partial.get("agents_json", "[]"), []),
            "workspace_files": job_workspace.list_files(),
        }
        failed_at = now()
        with db.lock, db.connect() as connection:
            connection.execute(
                "INSERT INTO messages(conversation_id,role,content,meta_json,created_at) VALUES(?,?,?,?,?)",
                (
                    job["conversation_id"],
                    "assistant",
                    partial.get("answer", ""),
                    json.dumps(meta, ensure_ascii=False),
                    failed_at,
                ),
            )
            connection.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?",
                (failed_at, job["conversation_id"]),
            )
            connection.execute(
                "UPDATE jobs SET status='failed',error=?,updated_at=? WHERE id=?",
                (error, failed_at, job_id),
            )
    finally:
        if attachment_lock_acquired:
            attachment_job_lock.release()
        if attachment_records:
            deleted = db.delete_attachments(job["user_id"], [item["id"] for item in attachment_records])
            await asyncio.to_thread(attachments.delete_files, deleted)
        tasks.pop(job_id, None)


def launch(job_id: str) -> None:
    if job_id not in tasks or tasks[job_id].done():
        tasks[job_id] = asyncio.create_task(run_job(job_id))


@asynccontextmanager
async def lifespan(_: FastAPI):
    global secret, attachment_cleanup_task
    db.init()
    migrate_custom_provider_settings()
    secret = load_secret(settings.secret_path)
    if not db.one("SELECT id FROM users LIMIT 1"):
        username = os.getenv("ADMIN_USERNAME", "admin").strip() or "admin"
        password = os.getenv("ADMIN_PASSWORD", "")
        if len(password) < 8:
            raise RuntimeError("首次启动必须设置至少 8 位 ADMIN_PASSWORD")
        db.run("INSERT INTO users(username,password_hash,is_admin,created_at) VALUES(?,?,1,?)", (username, password_hash(password), now()))
    stale = db.all("SELECT id FROM jobs WHERE status IN ('queued','running')")
    for item in stale:
        db.update_job(item["id"], status="queued", stop_requested=0)
        launch(item["id"])
    attachment_cleanup_task = asyncio.create_task(periodic_attachment_cleanup())
    yield
    if attachment_cleanup_task:
        attachment_cleanup_task.cancel()
        attachment_cleanup_task = None
    for task in list(tasks.values()):
        task.cancel()


app = FastAPI(title="DeepSeek Native Chat", lifespan=lifespan)
static_dir = Path(__file__).resolve().parent.parent / "static"
app.mount("/assets", StaticFiles(directory=static_dir), name="assets")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/login")
def login(body: LoginBody, response: Response) -> dict[str, Any]:
    user = db.one("SELECT * FROM users WHERE username=? COLLATE NOCASE", (body.username.strip(),))
    if not user or not password_ok(body.password, user["password_hash"]):
        raise HTTPException(401, "用户名或密码错误")
    token = make_token(user["id"], secret, settings.session_days)
    response.set_cookie("session", token, max_age=settings.session_days * 86400, httponly=True, secure=bool(settings.tls_cert_file), samesite="lax", path="/")
    return {"id": user["id"], "username": user["username"], "is_admin": bool(user["is_admin"])}


@app.post("/api/logout")
def logout(response: Response) -> dict[str, bool]:
    response.delete_cookie("session", path="/")
    return {"ok": True}


@app.get("/api/me")
def me(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    return user


@app.get("/api/users")
def users(_: dict[str, Any] = Depends(admin_user)) -> list[dict[str, Any]]:
    rows = db.all("SELECT id,username,is_admin,created_at FROM users ORDER BY id")
    for row in rows:
        row["is_admin"] = bool(row["is_admin"])
    return rows


@app.post("/api/users")
def add_user(body: UserBody, _: dict[str, Any] = Depends(admin_user)) -> dict[str, Any]:
    if db.one("SELECT COUNT(*) AS n FROM users")["n"] >= 3:
        raise HTTPException(400, "账号上限为 3 个")
    try:
        user_id = db.run("INSERT INTO users(username,password_hash,is_admin,created_at) VALUES(?,?,?,?)", (body.username.strip(), password_hash(body.password), int(body.is_admin), now()))
    except Exception as exc:
        if "UNIQUE" in str(exc):
            raise HTTPException(409, "用户名已存在") from exc
        raise
    return {"id": user_id, "username": body.username.strip(), "is_admin": body.is_admin}


@app.delete("/api/users/{user_id}")
def delete_user(user_id: int, admin: dict[str, Any] = Depends(admin_user)) -> dict[str, bool]:
    if user_id == admin["id"]:
        raise HTTPException(400, "不能删除当前登录账号")
    target = db.one("SELECT is_admin FROM users WHERE id=?", (user_id,))
    if not target:
        raise HTTPException(404, "账号不存在")
    if target["is_admin"] and db.one("SELECT COUNT(*) AS n FROM users WHERE is_admin=1")["n"] <= 1:
        raise HTTPException(400, "必须保留一个管理员")
    if db.one("SELECT id FROM jobs WHERE user_id=? AND status IN ('queued','running')", (user_id,)):
        raise HTTPException(409, "该账号正在生成回答，请先停止后再删除")
    attachment_records = db.get_attachments(user_id)
    db.run("DELETE FROM users WHERE id=?", (user_id,))
    attachments.delete_files(attachment_records)
    return {"ok": True}


@app.put("/api/users/{user_id}/password")
def change_password(user_id: int, body: PasswordBody, _: dict[str, Any] = Depends(admin_user)) -> dict[str, bool]:
    if not db.one("SELECT id FROM users WHERE id=?", (user_id,)):
        raise HTTPException(404, "账号不存在")
    db.run("UPDATE users SET password_hash=? WHERE id=?", (password_hash(body.password), user_id))
    return {"ok": True}


@app.get("/api/attachments")
def list_pending_attachments(draft_id: str, user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    if not re.fullmatch(r"[a-f0-9]{32}", draft_id):
        raise HTTPException(400, "无效的附件草稿标识")
    records = db.pending_attachments(user["id"], draft_id)
    return {
        "data": [public_attachment(record) for record in records],
        "max_files": attachments.MAX_ATTACHMENTS,
        "max_total_bytes": attachments.MAX_UPLOAD_BYTES,
    }


@app.post("/api/attachments")
async def upload_attachment(
    request: Request,
    draft_id: str,
    filename: str = "attachment",
    user: dict[str, Any] = Depends(current_user),
) -> dict[str, Any]:
    if not re.fullmatch(r"[a-f0-9]{32}", draft_id):
        raise HTTPException(400, "无效的附件草稿标识")
    name = attachments.safe_filename(filename)
    try:
        declared_size = int(request.headers.get("content-length") or 0)
    except ValueError:
        declared_size = 0
    if declared_size > attachments.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "一次消息的附件总量不能超过 50MB")

    lock = attachment_upload_locks.setdefault(user["id"], asyncio.Lock())
    async with lock:
        usage = db.attachment_usage(user["id"], draft_id)
        if usage["count"] >= attachments.MAX_ATTACHMENTS:
            raise HTTPException(400, "一次消息最多上传 10 个附件")
        remaining = attachments.MAX_UPLOAD_BYTES - usage["bytes"]
        if remaining <= 0 or (declared_size and declared_size > remaining):
            raise HTTPException(413, "一次消息的附件总量不能超过 50MB")

        attachment_id = uuid.uuid4().hex
        incoming = attachments.attachment_path(user["id"], f".incoming-{attachment_id}", ".upload")
        written = 0
        processed_record: Optional[dict[str, Any]] = None
        try:
            with incoming.open("wb") as output:
                async for chunk in request.stream():
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > remaining or written > attachments.MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "一次消息的附件总量不能超过 50MB")
                    output.write(chunk)
            if written <= 0:
                raise HTTPException(400, "附件内容为空")
            os.chmod(incoming, 0o600)
            async with attachment_processing_lock:
                result = await asyncio.to_thread(
                    attachments.process_upload,
                    incoming,
                    user["id"],
                    attachment_id,
                    name,
                    request.headers.get("content-type", "application/octet-stream"),
                )
            processed_record = {"stored_path": str(result["stored_path"])}
            record = db.create_attachment(
                attachment_id,
                user["id"],
                draft_id,
                name,
                str(result["kind"]),
                str(result["media_type"]),
                str(result["stored_path"]),
                written,
                int(result["processed_size"]),
                now(),
            )
            return public_attachment(record)
        except attachments.AttachmentError as exc:
            raise HTTPException(415, str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            if processed_record:
                attachments.delete_files([processed_record])
            raise HTTPException(500, "附件保存失败") from exc
        finally:
            incoming.unlink(missing_ok=True)


@app.delete("/api/attachments/{attachment_id}")
def delete_attachment(attachment_id: str, user: dict[str, Any] = Depends(current_user)) -> dict[str, bool]:
    if not re.fullmatch(r"[a-f0-9]{32}", attachment_id):
        raise HTTPException(400, "无效的附件标识")
    record = db.one("SELECT * FROM attachments WHERE id=? AND user_id=?", (attachment_id, user["id"]))
    if not record:
        return {"ok": True}
    if record.get("job_id"):
        job = db.one("SELECT status FROM jobs WHERE id=? AND user_id=?", (record["job_id"], user["id"]))
        if job and job["status"] in {"queued", "running"}:
            raise HTTPException(409, "附件正在用于生成回答，暂时不能删除")
    deleted = db.delete_attachments(user["id"], [attachment_id])
    attachments.delete_files(deleted)
    return {"ok": True}


@app.get("/api/providers")
def providers(user: dict[str, Any] = Depends(current_user)) -> list[dict[str, Any]]:
    return [public_provider(row) for row in db.all("SELECT * FROM providers WHERE user_id=? ORDER BY id", (user["id"],))]


async def test_custom_model(base_url: str, api_key: str, model: str) -> None:
    """Validate a manually entered model with a one-token chat request."""
    mimo_model = is_mimo_model(model)
    token_field = "max_completion_tokens" if mimo_model else "max_tokens"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply OK"}],
        token_field: 1,
        "stream": False,
    }
    if mimo_model:
        # Keep a connection test cheap and deterministic. MiMo accepts the
        # thinking switch, while ordinary Custom providers never receive it.
        payload["thinking"] = {"type": "disabled"}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30, connect=5), follow_redirects=True) as client:
            response = await client.post(
                base_url.rstrip("/") + "/chat/completions",
                headers=custom_auth_headers(api_key, base_url=base_url),
                json=payload,
            )
    except Exception as exc:
        raise HTTPException(502, f"手填模型 {model} 连接失败：{type(exc).__name__}") from exc
    if response.status_code >= 400:
        message = ""
        try:
            data = response.json()
            error = data.get("error") or {}
            message = error.get("message") if isinstance(error, dict) else data.get("detail")
        except Exception:
            pass
        detail = f"手填模型 {model} 测试失败（HTTP {response.status_code}）"
        if message:
            detail += f"：{str(message)[:300]}"
        raise HTTPException(400, detail)


async def test_provider_credentials(kind: str, base: str, api_key: str, manual_values: Any) -> dict[str, Any]:
    # The current UI sends checkbox selections separately from manually typed
    # names. Fall back to selected_models for older clients.
    manual_models = _clean_model_ids(manual_values)
    manual_tested: list[str] = []
    try:
        models = await (custom_list_models(base, api_key) if kind == "custom" else deepseek_list_models(base, api_key))
    except Exception as exc:
        if kind != "custom" or not manual_models:
            raise HTTPException(400, f"API 测试失败：{exc}") from exc
        models = []
        models_warning = str(exc)
    else:
        models_warning = ""
    if kind == "custom":
        if len(manual_models) > 20:
            raise HTTPException(400, "一次最多测试 20 个手填模型")
        advertised = set(models)
        for model_id in manual_models:
            if model_id not in advertised:
                await test_custom_model(base, api_key, model_id)
                manual_tested.append(model_id)
        models = list(dict.fromkeys([*models, *manual_models]))
        supported = models
    else:
        supported = sorted(SUPPORTED_MODELS[kind])
    return {
        "ok": True,
        "provider_type": kind,
        "models": models,
        "supported_models": supported,
        "native_search_models": [m for m in models if m in SUPPORTED_MODELS[kind]],
        "manual_tested": manual_tested,
        "models_warning": models_warning,
    }


@app.post("/api/providers/test")
async def test_provider(body: ProviderBody, _: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    kind = body.provider_type
    base = clean_base_url(body.base_url or DEFAULT_BASE_URLS[kind])
    manual_values = body.manual_models if body.manual_models is not None else body.selected_models
    return await test_provider_credentials(kind, base, body.api_key, manual_values)


@app.post("/api/providers")
def add_provider(body: ProviderBody, user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    kind = body.provider_type
    base = clean_base_url(body.base_url or DEFAULT_BASE_URLS[kind])
    selected_models = _clean_model_ids(body.selected_models)
    if kind == "custom":
        model = body.model.strip() or (selected_models[0] if selected_models else "")
        if model and model not in selected_models:
            selected_models.insert(0, model)
        if not selected_models:
            raise HTTPException(400, "请至少选择或填写一个 custom 模型")
    else:
        model = body.model.strip() or "deepseek-v4-flash"
        selected_models = [model]
    validate_provider_selection(kind, model)
    if kind == "custom":
        initial_settings = normalize_custom_settings(body.custom_settings)
        settings_value = {
            "models": selected_models,
            "model_settings": {item: dict(initial_settings) for item in selected_models},
        }
    else:
        settings_value = {"models": selected_models}
    settings_json = json.dumps(settings_value, ensure_ascii=False)
    provider_id = db.run(
        "INSERT INTO providers(user_id,name,api_key,base_url,model,provider_type,settings_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
        (user["id"], body.name.strip(), body.api_key.strip(), base, model, kind, settings_json, now()),
    )
    return public_provider(db.one("SELECT * FROM providers WHERE id=?", (provider_id,)))


@app.post("/api/providers/{provider_id}/test")
async def test_saved_provider(provider_id: int, body: ProviderModelsBody, user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    provider = db.one("SELECT * FROM providers WHERE id=? AND user_id=?", (provider_id, user["id"]))
    if not provider:
        raise HTTPException(404, "API 配置不存在")
    kind = provider_type(provider)
    manual_values = body.manual_models if body.manual_models is not None else body.selected_models
    return await test_provider_credentials(kind, provider["base_url"], provider["api_key"], manual_values)


@app.put("/api/providers/{provider_id}/models")
def update_provider_models(provider_id: int, body: ProviderModelsBody, user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    provider = db.one("SELECT * FROM providers WHERE id=? AND user_id=?", (provider_id, user["id"]))
    if not provider:
        raise HTTPException(404, "API 配置不存在")
    kind = provider_type(provider)
    selected_models = _clean_model_ids(body.selected_models)
    if kind == "custom":
        model = body.model.strip() or (selected_models[0] if selected_models else "")
        if model and model not in selected_models:
            selected_models.insert(0, model)
        if not selected_models:
            raise HTTPException(400, "请至少选择或填写一个 custom 模型")
    else:
        model = body.model.strip() or "deepseek-v4-flash"
        selected_models = [model]
    validate_provider_selection(kind, model)
    settings_value = (
        custom_settings_document(provider, selected_models)
        if kind == "custom"
        else {"models": selected_models}
    )
    db.run(
        "UPDATE providers SET model=?,settings_json=? WHERE id=? AND user_id=?",
        (model, json.dumps(settings_value, ensure_ascii=False), provider_id, user["id"]),
    )
    return public_provider(db.one("SELECT * FROM providers WHERE id=? AND user_id=?", (provider_id, user["id"])))


@app.delete("/api/providers/{provider_id}")
def delete_provider(provider_id: int, user: dict[str, Any] = Depends(current_user)) -> dict[str, bool]:
    if not db.one("SELECT id FROM providers WHERE id=? AND user_id=?", (provider_id, user["id"])):
        raise HTTPException(404, "API 配置不存在")
    if db.one("SELECT id FROM jobs WHERE provider_id=? AND status IN ('queued','running')", (provider_id,)):
        raise HTTPException(409, "该 API 正在生成回答，暂时不能删除")
    db.run("DELETE FROM providers WHERE id=? AND user_id=?", (provider_id, user["id"]))
    return {"ok": True}


@app.put("/api/providers/{provider_id}/settings")
def update_provider_settings(provider_id: int, body: CustomModelSettingsBody, user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    provider = db.one("SELECT * FROM providers WHERE id=? AND user_id=?", (provider_id, user["id"]))
    if not provider:
        raise HTTPException(404, "API 配置不存在")
    if provider_type(provider) != "custom":
        raise HTTPException(400, "只有 custom API 支持这组参数")
    model = body.model.strip()
    validate_provider_selection("custom", model, provider)
    settings_value = custom_settings_document(provider)
    settings_value["model_settings"][model] = normalize_custom_settings(body.model_dump(exclude={"model"}))
    db.run("UPDATE providers SET settings_json=? WHERE id=? AND user_id=?", (json.dumps(settings_value, ensure_ascii=False), provider_id, user["id"]))
    return public_provider(db.one("SELECT * FROM providers WHERE id=? AND user_id=?", (provider_id, user["id"])))


@app.get("/api/conversations")
def conversations(page: int = 1, user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    page = max(1, page)
    total = db.one("SELECT COUNT(*) AS n FROM conversations WHERE user_id=?", (user["id"],))["n"]
    rows = db.all(
        """SELECT * FROM conversations WHERE user_id=?
           ORDER BY pinned_at IS NULL, pinned_at DESC, updated_at DESC
           LIMIT 10 OFFSET ?""",
        (user["id"], (page - 1) * 10),
    )
    return {
        "items": [public_conversation(row) for row in rows],
        "page": page,
        "pages": max(1, (total + 9) // 10),
        "total": total,
    }


@app.get("/api/conversations/{conversation_id}")
def conversation(conversation_id: str, user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    conv = db.one("SELECT * FROM conversations WHERE id=? AND user_id=?", (conversation_id, user["id"]))
    if not conv:
        raise HTTPException(404, "对话不存在")
    rows = db.all("SELECT id,role,content,meta_json,created_at FROM messages WHERE conversation_id=? ORDER BY id", (conversation_id,))
    for row in rows:
        row["meta"] = db.decode(row.pop("meta_json"), {})
    active = db.one("SELECT * FROM jobs WHERE conversation_id=? AND status IN ('queued','running') ORDER BY created_at DESC LIMIT 1", (conversation_id,))
    workspace_files = ConversationWorkspace(user["id"], conversation_id).list_files()
    if workspace_files:
        for row in reversed(rows):
            if row["role"] == "assistant":
                row["meta"]["conversation_id"] = conversation_id
                row["meta"]["workspace_files"] = workspace_files
                break
    return {"conversation": public_conversation(conv), "messages": rows, "active_job": public_job(active) if active else None, "workspace_files": workspace_files}


def owned_workspace(conversation_id: str, user: dict[str, Any]) -> ConversationWorkspace:
    if not db.one("SELECT id FROM conversations WHERE id=? AND user_id=?", (conversation_id, user["id"])):
        raise HTTPException(404, "对话不存在")
    return ConversationWorkspace(user["id"], conversation_id)


@app.get("/api/conversations/{conversation_id}/workspace")
def conversation_workspace(conversation_id: str, user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    workspace = owned_workspace(conversation_id, user)
    files = workspace.list_files()
    return {"files": files, "total_size": sum(int(item["size"]) for item in files)}


@app.get("/api/conversations/{conversation_id}/workspace/files/{file_path:path}")
def download_workspace_file(conversation_id: str, file_path: str, user: dict[str, Any] = Depends(current_user)) -> FileResponse:
    workspace = owned_workspace(conversation_id, user)
    try:
        target, relative = workspace.resolve(file_path)
    except WorkspaceError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not target.is_file() or target.is_symlink():
        raise HTTPException(404, "工作区文件不存在")
    return FileResponse(target, filename=Path(relative).name, media_type="application/octet-stream")


@app.get("/api/conversations/{conversation_id}/workspace.zip")
def download_workspace_zip(conversation_id: str, user: dict[str, Any] = Depends(current_user)) -> StreamingResponse:
    workspace = owned_workspace(conversation_id, user)
    files = workspace.list_files()
    if not files:
        raise HTTPException(404, "当前对话还没有工作区文件")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for item in files:
            target, relative = workspace.resolve(item["path"])
            output.write(target, relative)
    archive.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="workspace-{conversation_id[:8]}.zip"'}
    return StreamingResponse(archive, media_type="application/zip", headers=headers)


@app.post("/api/conversations/{conversation_id}/pin")
def pin_conversation(
    conversation_id: str,
    body: PinBody,
    user: dict[str, Any] = Depends(current_user),
) -> dict[str, Any]:
    conv = db.one("SELECT * FROM conversations WHERE id=? AND user_id=?", (conversation_id, user["id"]))
    if not conv:
        raise HTTPException(404, "对话不存在")
    if body.pinned and conv.get("pinned_at") is None:
        db.run(
            "UPDATE conversations SET pinned_at=? WHERE id=? AND user_id=?",
            (time.time_ns() // 1_000_000, conversation_id, user["id"]),
        )
    elif not body.pinned and conv.get("pinned_at") is not None:
        db.run(
            "UPDATE conversations SET pinned_at=NULL WHERE id=? AND user_id=?",
            (conversation_id, user["id"]),
        )
    updated = db.one("SELECT * FROM conversations WHERE id=? AND user_id=?", (conversation_id, user["id"]))
    return public_conversation(updated or conv)


@app.post("/api/conversations/{conversation_id}/retry")
async def retry_answer(
    conversation_id: str,
    body: RetryBody,
    user: dict[str, Any] = Depends(current_user),
) -> dict[str, str]:
    """Discard messages after one prompt and regenerate its answer in place."""
    provider = db.one("SELECT * FROM providers WHERE id=? AND user_id=?", (body.provider_id, user["id"]))
    if not provider:
        raise HTTPException(404, "请选择有效的 API 配置")
    kind = provider_type(provider)
    model = body.model.strip() or provider["model"]
    if kind == "deepseek" and model != provider["model"]:
        raise HTTPException(400, "当前回答使用的模型与 API 配置不一致，请重新选择模型配置")
    validate_provider_selection(kind, model, provider)
    if body.chat_mode == "multi_agent" and kind != "custom":
        raise HTTPException(400, "多智能体协作模式仅支持 Custom 模型")
    validate_effort(body.effort)
    timezone_name = clean_timezone(body.timezone)
    job_id = uuid.uuid4().hex
    created_at = now()

    with db.lock, db.connect() as connection:
        conversation_row = connection.execute(
            "SELECT id FROM conversations WHERE id=? AND user_id=?",
            (conversation_id, user["id"]),
        ).fetchone()
        if not conversation_row:
            raise HTTPException(404, "对话不存在")
        active = connection.execute(
            "SELECT id FROM jobs WHERE conversation_id=? AND status IN ('queued','running') LIMIT 1",
            (conversation_id,),
        ).fetchone()
        if active:
            raise HTTPException(409, "当前对话仍在生成回答")
        prompt = connection.execute(
            "SELECT id,role,meta_json FROM messages WHERE id=? AND conversation_id=?",
            (body.prompt_message_id, conversation_id),
        ).fetchone()
        if not prompt or prompt["role"] != "user":
            raise HTTPException(409, "找不到要重新回答的问题")
        prompt_meta = db.decode(prompt["meta_json"], {})
        if isinstance(prompt_meta, dict) and prompt_meta.get("attachments"):
            raise HTTPException(409, "原问题包含已清理的附件，请重新上传后再提问")

        connection.execute(
            "DELETE FROM messages WHERE conversation_id=? AND id>?",
            (conversation_id, prompt["id"]),
        )
        connection.execute(
            """INSERT INTO jobs(
                   id,user_id,conversation_id,provider_id,provider_type,model,
                   effort,timezone,chat_mode,status,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                job_id, user["id"], conversation_id, body.provider_id, kind,
                model, body.effort, timezone_name, body.chat_mode, "queued", created_at, created_at,
            ),
        )
        connection.execute("UPDATE conversations SET updated_at=? WHERE id=?", (created_at, conversation_id))

    launch(job_id)
    return {"job_id": job_id, "conversation_id": conversation_id}


@app.delete("/api/conversations")
def delete_all_conversations(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    attachment_records = db.get_attachments(user["id"])
    deleted, compacted = db.clear_user_chats(user["id"])
    if not compacted:
        raise HTTPException(409, "请先等待或停止所有正在生成的回答")
    attachments.delete_files(attachment_records)
    delete_user_workspaces(user["id"])
    return {"ok": True, "deleted": deleted, "compacted": True}


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, user: dict[str, Any] = Depends(current_user)) -> dict[str, bool]:
    active = db.one("SELECT id FROM jobs WHERE conversation_id=? AND user_id=? AND status IN ('queued','running')", (conversation_id, user["id"]))
    if active:
        raise HTTPException(409, "请先停止正在生成的回答")
    attachment_records = db.all(
        "SELECT * FROM attachments WHERE user_id=? AND (conversation_id=? OR (job_id IS NULL AND draft_id=?))",
        (user["id"], conversation_id, conversation_id),
    )
    if attachment_records:
        db.delete_attachments(user["id"], [item["id"] for item in attachment_records])
    db.run("DELETE FROM conversations WHERE id=? AND user_id=?", (conversation_id, user["id"]))
    attachments.delete_files(attachment_records)
    delete_conversation_workspace(user["id"], conversation_id)
    return {"ok": True}


@app.post("/api/chat")
async def chat(body: ChatBody, user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    provider = db.one("SELECT * FROM providers WHERE id=? AND user_id=?", (body.provider_id, user["id"]))
    if not provider:
        raise HTTPException(404, "请选择有效的 API 配置")
    kind = provider_type(provider)
    model = body.model.strip() or provider["model"]
    if kind == "deepseek" and model != provider["model"]:
        raise HTTPException(400, "当前回答使用的模型与 API 配置不一致，请重新选择模型配置")
    validate_provider_selection(kind, model, provider)
    if body.chat_mode == "multi_agent" and kind != "custom":
        raise HTTPException(400, "多智能体协作模式仅支持 Custom 模型")
    validate_effort(body.effort)
    content = body.content.strip()
    attachment_ids = list(dict.fromkeys(body.attachment_ids))
    if any(not re.fullmatch(r"[a-f0-9]{32}", item) for item in attachment_ids):
        raise HTTPException(400, "无效的附件标识")
    if not content and not attachment_ids:
        raise HTTPException(400, "消息和附件不能同时为空")
    attachment_records = db.get_attachments(user["id"], attachment_ids)
    if len(attachment_records) != len(attachment_ids) or any(item.get("job_id") for item in attachment_records):
        raise HTTPException(400, "部分附件不存在、已过期或已经发送")
    if kind != "custom" and any(item["kind"] == "image" for item in attachment_records):
        raise HTTPException(400, "当前 DeepSeek Responses 路由不接受图片，请改用支持视觉输入的 Custom 模型")
    timezone_name = clean_timezone(body.timezone)
    conversation_id = body.conversation_id
    created_conversation = False
    if conversation_id:
        if not db.one("SELECT id FROM conversations WHERE id=? AND user_id=?", (conversation_id, user["id"])):
            raise HTTPException(404, "对话不存在")
        if db.one("SELECT id FROM jobs WHERE conversation_id=? AND status IN ('queued','running')", (conversation_id,)):
            raise HTTPException(409, "当前对话仍在生成回答")
    else:
        conversation_id = uuid.uuid4().hex
        created_conversation = True
        title_source = content or "、".join(item["original_name"] for item in attachment_records) or "附件对话"
        db.run("INSERT INTO conversations(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)", (conversation_id, user["id"], title_for(title_source), now(), now()))
        trim_old_conversations(user["id"])
    attachment_meta = [public_attachment(item) for item in attachment_records]
    message_content = content or "请分析这些附件。"
    message_id = db.run(
        "INSERT INTO messages(conversation_id,role,content,meta_json,created_at) VALUES(?,?,?,?,?)",
        (conversation_id, "user", message_content, json.dumps({"attachments": attachment_meta}, ensure_ascii=False), now()),
    )
    db.run("UPDATE conversations SET updated_at=? WHERE id=?", (now(), conversation_id))
    job_id = uuid.uuid4().hex
    db.run(
        "INSERT INTO jobs(id,user_id,conversation_id,provider_id,provider_type,model,effort,timezone,chat_mode,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (job_id, user["id"], conversation_id, body.provider_id, kind, model, body.effort, timezone_name, body.chat_mode, "queued", now(), now()),
    )
    if attachment_ids and not db.claim_attachments(user["id"], attachment_ids, conversation_id, job_id):
        db.run("DELETE FROM jobs WHERE id=? AND user_id=?", (job_id, user["id"]))
        db.run("DELETE FROM messages WHERE id=? AND conversation_id=?", (message_id, conversation_id))
        if created_conversation:
            db.run("DELETE FROM conversations WHERE id=? AND user_id=?", (conversation_id, user["id"]))
        raise HTTPException(409, "附件状态已变化，请重新选择后发送")
    launch(job_id)
    return {"job_id": job_id, "conversation_id": conversation_id, "message_id": message_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    job = db.one("SELECT * FROM jobs WHERE id=? AND user_id=?", (job_id, user["id"]))
    if not job:
        raise HTTPException(404, "任务不存在")
    return public_job(job)


@app.post("/api/jobs/{job_id}/stop")
async def stop_job(job_id: str, user: dict[str, Any] = Depends(current_user)) -> dict[str, bool]:
    job = db.one("SELECT id,status FROM jobs WHERE id=? AND user_id=?", (job_id, user["id"]))
    if not job:
        raise HTTPException(404, "任务不存在")
    if job["status"] in {"queued", "running"}:
        db.update_job(job_id, status="stopped", stop_requested=1)
        task = tasks.get(job_id)
        if task:
            task.cancel()
    return {"ok": True}


@app.get("/{path:path}")
def frontend(path: str, request: Request) -> FileResponse:
    candidate = static_dir / path
    if path and candidate.is_file() and static_dir in candidate.resolve().parents:
        return FileResponse(candidate)
    return FileResponse(static_dir / "index.html")


if __name__ == "__main__":
    kwargs: dict[str, Any] = {"host": settings.host, "port": settings.port, "log_level": "info"}
    if settings.tls_cert_file and settings.tls_key_file:
        kwargs.update(ssl_certfile=settings.tls_cert_file, ssl_keyfile=settings.tls_key_file)
    uvicorn.run("app.main:app", **kwargs)

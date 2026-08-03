from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Optional

import uvicorn
from fastapi import Cookie, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings
from .db import Database
from .deepseek import list_models as deepseek_list_models
from .deepseek import stream_response as deepseek_stream_response
from .mimo import DEFAULT_SETTINGS as MIMO_DEFAULT_SETTINGS
from .mimo import MIMO_MAX_COMPLETION_TOKENS, MIMO_MODELS, list_models as mimo_list_models
from .mimo_local import stream_response as mimo_stream_response
from .security import load_secret, make_token, password_hash, password_ok, read_token


db = Database(settings.db_path)
secret = b""
tasks: dict[str, asyncio.Task[Any]] = {}
SUPPORTED_MODELS = {
    "deepseek": {"deepseek-v4-flash"},
    "mimo": MIMO_MODELS,
}
DEFAULT_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "mimo": "https://api.xiaomimimo.com/v1",
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
    provider_type: Literal["deepseek", "mimo"] = "deepseek"
    base_url: str = ""
    model: str = ""
    mimo_settings: Optional[dict[str, Any]] = None


class MimoSettingsBody(BaseModel):
    thinking: Literal["enabled", "disabled"] = "enabled"
    max_completion_tokens: int = Field(default=8192, ge=256, le=MIMO_MAX_COMPLETION_TOKENS)
    temperature: float = Field(default=1.0, ge=0, le=1.5)
    top_p: float = Field(default=0.95, ge=0.01, le=1)


class ChatBody(BaseModel):
    conversation_id: Optional[str] = None
    content: str = Field(min_length=1, max_length=100_000)
    provider_id: int
    model: str = ""
    effort: str = "high"


def normalize_mimo_settings(value: Any = None) -> dict[str, Any]:
    data = dict(MIMO_DEFAULT_SETTINGS)
    if isinstance(value, MimoSettingsBody):
        data.update(value.model_dump())
    elif isinstance(value, dict):
        data.update(value)
    return MimoSettingsBody(**data).model_dump()


def provider_type(row: dict[str, Any]) -> str:
    return row.get("provider_type") or "deepseek"


def validate_provider_selection(kind: str, model: str) -> None:
    if kind not in SUPPORTED_MODELS:
        raise HTTPException(400, "不支持的服务商类型")
    if model not in SUPPORTED_MODELS[kind]:
        available = "、".join(sorted(SUPPORTED_MODELS[kind]))
        raise HTTPException(400, f"{kind} 当前支持的模型为：{available}")


def now() -> int:
    return int(time.time())


def clean_base_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if not value.startswith("https://"):
        raise HTTPException(400, "API 地址必须使用 HTTPS")
    return value


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
    key = row.pop("api_key", "")
    row["provider_type"] = row.get("provider_type") or "deepseek"
    row["settings"] = db.decode(row.pop("settings_json", "{}"), {})
    row["api_key_masked"] = (key[:3] + "••••" + key[-4:]) if len(key) > 8 else "••••••••"
    return row


def public_job(row: dict[str, Any]) -> dict[str, Any]:
    for name, fallback in (("searches_json", []), ("sources_json", []), ("usage_json", {})):
        row[name.removesuffix("_json")] = db.decode(row.pop(name), fallback)
    row["stop_requested"] = bool(row["stop_requested"])
    return row


def title_for(text: str) -> str:
    compact = " ".join(text.split())
    return compact[:36] + ("…" if len(compact) > 36 else "")


async def run_job(job_id: str) -> None:
    job = db.one("SELECT * FROM jobs WHERE id=?", (job_id,))
    if not job:
        return
    provider = db.one("SELECT * FROM providers WHERE id=? AND user_id=?", (job["provider_id"], job["user_id"]))
    if not provider:
        db.update_job(job_id, status="failed", error="API 配置不存在")
        return
    kind = provider_type(provider)
    history_rows = db.all(
        "SELECT role, content, meta_json FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT 20",
        (job["conversation_id"],),
    )
    history: list[dict[str, Any]] = []
    for row in reversed(history_rows):
        message: dict[str, Any] = {"role": row["role"], "content": row["content"]}
        # MiMo accepts historical reasoning_content in later turns. Client-side
        # tool messages are intentionally not replayed here: the final assistant
        # message is persisted, while replaying an assistant tool_call without
        # its matching tool result can make compatible gateways reject history.
        if kind == "mimo" and row["role"] == "assistant":
            meta = db.decode(row.get("meta_json", "{}"), {})
            if meta.get("reasoning"):
                message["reasoning_content"] = meta.get("reasoning", "")
        history.append(message)
    db.update_job(job_id, status="running", error="", stop_requested=0)
    last_write = 0.0

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
        )

    try:
        provider_settings = db.decode(provider.get("settings_json", "{}"), {})
        if kind == "mimo":
            result = await mimo_stream_response(
                base_url=provider["base_url"],
                api_key=provider["api_key"],
                model=job["model"],
                messages=history,
                timeout=settings.request_timeout,
                stopped=stopped,
                update=update,
                settings=provider_settings,
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
        meta = {"job_id": job_id, "provider_id": job["provider_id"], "provider_type": kind, "model": job["model"], "reasoning": result["reasoning"], "searches": result["searches"], "sources": result["sources"], "usage": result["usage"]}
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
        )
    except asyncio.CancelledError:
        partial = db.one("SELECT answer, reasoning, searches_json, sources_json, usage_json FROM jobs WHERE id=?", (job_id,)) or {}
        if partial.get("answer"):
            meta = {
                "job_id": job_id,
                "stopped": True,
                "provider_id": job["provider_id"],
                "provider_type": provider_type(provider),
                "model": job["model"],
                "reasoning": partial.get("reasoning", ""),
                "searches": db.decode(partial.get("searches_json", "[]"), []),
                "sources": db.decode(partial.get("sources_json", "[]"), []),
                "usage": db.decode(partial.get("usage_json", "{}"), {}),
            }
            db.run(
                "INSERT INTO messages(conversation_id, role, content, meta_json, created_at) VALUES(?,?,?,?,?)",
                (job["conversation_id"], "assistant", partial["answer"] + "\n\n_已停止生成_", json.dumps(meta, ensure_ascii=False), now()),
            )
        db.update_job(job_id, status="stopped", error="")
    except Exception as exc:
        db.update_job(job_id, status="failed", error=str(exc)[:3000])
    finally:
        tasks.pop(job_id, None)


def launch(job_id: str) -> None:
    if job_id not in tasks or tasks[job_id].done():
        tasks[job_id] = asyncio.create_task(run_job(job_id))


@asynccontextmanager
async def lifespan(_: FastAPI):
    global secret
    db.init()
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
    yield
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
    db.run("DELETE FROM users WHERE id=?", (user_id,))
    return {"ok": True}


@app.put("/api/users/{user_id}/password")
def change_password(user_id: int, body: PasswordBody, _: dict[str, Any] = Depends(admin_user)) -> dict[str, bool]:
    if not db.one("SELECT id FROM users WHERE id=?", (user_id,)):
        raise HTTPException(404, "账号不存在")
    db.run("UPDATE users SET password_hash=? WHERE id=?", (password_hash(body.password), user_id))
    return {"ok": True}


@app.get("/api/providers")
def providers(user: dict[str, Any] = Depends(current_user)) -> list[dict[str, Any]]:
    return [public_provider(row) for row in db.all("SELECT * FROM providers WHERE user_id=? ORDER BY id", (user["id"],))]


@app.post("/api/providers/test")
async def test_provider(body: ProviderBody, _: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    kind = body.provider_type
    base = clean_base_url(body.base_url or DEFAULT_BASE_URLS[kind])
    try:
        models = await (mimo_list_models(base, body.api_key) if kind == "mimo" else deepseek_list_models(base, body.api_key))
    except Exception as exc:
        raise HTTPException(400, f"API 测试失败：{exc}") from exc
    supported = sorted(SUPPORTED_MODELS[kind])
    return {"ok": True, "provider_type": kind, "models": models, "supported_models": supported, "native_search_models": [m for m in models if m in SUPPORTED_MODELS[kind]]}


@app.post("/api/providers")
def add_provider(body: ProviderBody, user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    kind = body.provider_type
    base = clean_base_url(body.base_url or DEFAULT_BASE_URLS[kind])
    model = body.model.strip() or ("mimo-v2.5-pro" if kind == "mimo" else "deepseek-v4-flash")
    validate_provider_selection(kind, model)
    settings_json = json.dumps(normalize_mimo_settings(body.mimo_settings if kind == "mimo" else None), ensure_ascii=False)
    provider_id = db.run(
        "INSERT INTO providers(user_id,name,api_key,base_url,model,provider_type,settings_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
        (user["id"], body.name.strip(), body.api_key.strip(), base, model, kind, settings_json, now()),
    )
    return public_provider(db.one("SELECT * FROM providers WHERE id=?", (provider_id,)))


@app.delete("/api/providers/{provider_id}")
def delete_provider(provider_id: int, user: dict[str, Any] = Depends(current_user)) -> dict[str, bool]:
    if not db.one("SELECT id FROM providers WHERE id=? AND user_id=?", (provider_id, user["id"])):
        raise HTTPException(404, "API 配置不存在")
    if db.one("SELECT id FROM jobs WHERE provider_id=? AND status IN ('queued','running')", (provider_id,)):
        raise HTTPException(409, "该 API 正在生成回答，暂时不能删除")
    db.run("DELETE FROM providers WHERE id=? AND user_id=?", (provider_id, user["id"]))
    return {"ok": True}


@app.put("/api/providers/{provider_id}/settings")
def update_provider_settings(provider_id: int, body: MimoSettingsBody, user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    provider = db.one("SELECT provider_type FROM providers WHERE id=? AND user_id=?", (provider_id, user["id"]))
    if not provider:
        raise HTTPException(404, "API 配置不存在")
    if provider_type(provider) != "mimo":
        raise HTTPException(400, "只有 MiMo API 支持这组联网参数")
    db.run("UPDATE providers SET settings_json=? WHERE id=? AND user_id=?", (json.dumps(body.model_dump(), ensure_ascii=False), provider_id, user["id"]))
    return public_provider(db.one("SELECT * FROM providers WHERE id=? AND user_id=?", (provider_id, user["id"])))


@app.get("/api/conversations")
def conversations(page: int = 1, user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    page = max(1, page)
    total = db.one("SELECT COUNT(*) AS n FROM conversations WHERE user_id=?", (user["id"],))["n"]
    rows = db.all("SELECT * FROM conversations WHERE user_id=? ORDER BY updated_at DESC LIMIT 10 OFFSET ?", (user["id"], (page - 1) * 10))
    return {"items": rows, "page": page, "pages": max(1, (total + 9) // 10), "total": total}


@app.get("/api/conversations/{conversation_id}")
def conversation(conversation_id: str, user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    conv = db.one("SELECT * FROM conversations WHERE id=? AND user_id=?", (conversation_id, user["id"]))
    if not conv:
        raise HTTPException(404, "对话不存在")
    rows = db.all("SELECT id,role,content,meta_json,created_at FROM messages WHERE conversation_id=? ORDER BY id", (conversation_id,))
    for row in rows:
        row["meta"] = db.decode(row.pop("meta_json"), {})
    active = db.one("SELECT * FROM jobs WHERE conversation_id=? AND status IN ('queued','running') ORDER BY created_at DESC LIMIT 1", (conversation_id,))
    return {"conversation": conv, "messages": rows, "active_job": public_job(active) if active else None}


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, user: dict[str, Any] = Depends(current_user)) -> dict[str, bool]:
    active = db.one("SELECT id FROM jobs WHERE conversation_id=? AND user_id=? AND status IN ('queued','running')", (conversation_id, user["id"]))
    if active:
        raise HTTPException(409, "请先停止正在生成的回答")
    db.run("DELETE FROM conversations WHERE id=? AND user_id=?", (conversation_id, user["id"]))
    return {"ok": True}


@app.post("/api/chat")
async def chat(body: ChatBody, user: dict[str, Any] = Depends(current_user)) -> dict[str, str]:
    provider = db.one("SELECT * FROM providers WHERE id=? AND user_id=?", (body.provider_id, user["id"]))
    if not provider:
        raise HTTPException(404, "请选择有效的 API 配置")
    kind = provider_type(provider)
    model = body.model.strip() or provider["model"]
    if model != provider["model"]:
        raise HTTPException(400, "当前回答使用的模型与 API 配置不一致，请重新选择模型配置")
    validate_provider_selection(kind, model)
    if body.effort not in {"high", "max"}:
        raise HTTPException(400, "无效的思考深度")
    conversation_id = body.conversation_id
    if conversation_id:
        if not db.one("SELECT id FROM conversations WHERE id=? AND user_id=?", (conversation_id, user["id"])):
            raise HTTPException(404, "对话不存在")
        if db.one("SELECT id FROM jobs WHERE conversation_id=? AND status IN ('queued','running')", (conversation_id,)):
            raise HTTPException(409, "当前对话仍在生成回答")
    else:
        conversation_id = uuid.uuid4().hex
        db.run("INSERT INTO conversations(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)", (conversation_id, user["id"], title_for(body.content), now(), now()))
        surplus = db.all("SELECT id FROM conversations WHERE user_id=? ORDER BY updated_at DESC LIMIT -1 OFFSET 100", (user["id"],))
        for item in surplus:
            db.run("DELETE FROM conversations WHERE id=?", (item["id"],))
    db.run("INSERT INTO messages(conversation_id,role,content,created_at) VALUES(?,?,?,?)", (conversation_id, "user", body.content.strip(), now()))
    db.run("UPDATE conversations SET updated_at=? WHERE id=?", (now(), conversation_id))
    job_id = uuid.uuid4().hex
    db.run(
        "INSERT INTO jobs(id,user_id,conversation_id,provider_id,provider_type,model,effort,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (job_id, user["id"], conversation_id, body.provider_id, kind, model, body.effort, "queued", now(), now()),
    )
    launch(job_id)
    return {"job_id": job_id, "conversation_id": conversation_id}


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

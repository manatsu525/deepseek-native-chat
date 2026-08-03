from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional


class Database:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = threading.RLock()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=20)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=20000")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init(self) -> None:
        with self.lock, self.connect() as db:
            db.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    is_admin INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS providers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    api_key TEXT NOT NULL,
                    base_url TEXT NOT NULL DEFAULT 'https://api.deepseek.com',
                    model TEXT NOT NULL DEFAULT 'deepseek-v4-flash',
                    provider_type TEXT NOT NULL DEFAULT 'deepseek',
                    settings_json TEXT NOT NULL DEFAULT '{}',
                    created_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_conversations_user_updated ON conversations(user_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    meta_json TEXT NOT NULL DEFAULT '{}',
                    created_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    provider_id INTEGER NOT NULL REFERENCES providers(id) ON DELETE CASCADE,
                    provider_type TEXT NOT NULL DEFAULT 'deepseek',
                    model TEXT NOT NULL,
                    effort TEXT NOT NULL,
                    status TEXT NOT NULL,
                    answer TEXT NOT NULL DEFAULT '',
                    reasoning TEXT NOT NULL DEFAULT '',
                    searches_json TEXT NOT NULL DEFAULT '[]',
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    usage_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT NOT NULL DEFAULT '',
                    stop_requested INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user_id, created_at DESC);
                """
            )
            provider_columns = {row["name"] for row in db.execute("PRAGMA table_info(providers)").fetchall()}
            if "provider_type" not in provider_columns:
                db.execute("ALTER TABLE providers ADD COLUMN provider_type TEXT NOT NULL DEFAULT 'deepseek'")
            if "settings_json" not in provider_columns:
                db.execute("ALTER TABLE providers ADD COLUMN settings_json TEXT NOT NULL DEFAULT '{}'")
            job_columns = {row["name"] for row in db.execute("PRAGMA table_info(jobs)").fetchall()}
            if "provider_type" not in job_columns:
                db.execute("ALTER TABLE jobs ADD COLUMN provider_type TEXT NOT NULL DEFAULT 'deepseek'")
            # Older releases called the OpenAI-compatible provider "mimo".
            # Keep existing API configurations and job history, but expose the
            # new generic name everywhere after the next startup.
            db.execute("UPDATE providers SET provider_type='custom' WHERE provider_type='mimo'")
            db.execute("UPDATE jobs SET provider_type='custom' WHERE provider_type='mimo'")
        self.path.chmod(0o600)

    def one(self, sql: str, args: tuple[Any, ...] = ()) -> Optional[dict[str, Any]]:
        with self.lock, self.connect() as db:
            row = db.execute(sql, args).fetchone()
            return dict(row) if row else None

    def all(self, sql: str, args: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self.lock, self.connect() as db:
            return [dict(row) for row in db.execute(sql, args).fetchall()]

    def run(self, sql: str, args: tuple[Any, ...] = ()) -> int:
        with self.lock, self.connect() as db:
            cur = db.execute(sql, args)
            return int(cur.lastrowid or 0)

    def update_job(self, job_id: str, **values: Any) -> None:
        if not values:
            return
        values["updated_at"] = int(time.time())
        fields = ", ".join(f"{key}=?" for key in values)
        self.run(f"UPDATE jobs SET {fields} WHERE id=?", (*values.values(), job_id))

    @staticmethod
    def decode(value: str, fallback: Any) -> Any:
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return fallback

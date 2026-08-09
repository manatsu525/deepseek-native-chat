from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional


class RetryBranchError(RuntimeError):
    """Raised when an owned chat record cannot be used to create a retry branch."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


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
                    timezone TEXT NOT NULL DEFAULT 'UTC',
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
                CREATE TABLE IF NOT EXISTS attachments (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
                    job_id TEXT REFERENCES jobs(id) ON DELETE CASCADE,
                    draft_id TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    original_size INTEGER NOT NULL,
                    processed_size INTEGER NOT NULL,
                    created_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_attachments_user_pending ON attachments(user_id, draft_id, job_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_attachments_job ON attachments(job_id);
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
            if "timezone" not in job_columns:
                db.execute("ALTER TABLE jobs ADD COLUMN timezone TEXT NOT NULL DEFAULT 'UTC'")
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

    def create_attachment(
        self,
        attachment_id: str,
        user_id: int,
        draft_id: str,
        original_name: str,
        kind: str,
        media_type: str,
        stored_path: str,
        original_size: int,
        processed_size: int,
        created_at: int,
    ) -> dict[str, Any]:
        self.run(
            """INSERT INTO attachments(
                   id,user_id,draft_id,original_name,kind,media_type,stored_path,
                   original_size,processed_size,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                attachment_id, user_id, draft_id, original_name, kind, media_type,
                stored_path, original_size, processed_size, created_at,
            ),
        )
        return self.one("SELECT * FROM attachments WHERE id=? AND user_id=?", (attachment_id, user_id)) or {}

    def attachment_usage(self, user_id: int, draft_id: str) -> dict[str, int]:
        row = self.one(
            "SELECT COUNT(*) AS count,COALESCE(SUM(original_size),0) AS bytes FROM attachments WHERE user_id=? AND draft_id=? AND job_id IS NULL",
            (user_id, draft_id),
        ) or {}
        return {"count": int(row.get("count", 0)), "bytes": int(row.get("bytes", 0))}

    def pending_attachments(self, user_id: int, draft_id: str) -> list[dict[str, Any]]:
        return self.all(
            "SELECT * FROM attachments WHERE user_id=? AND draft_id=? AND job_id IS NULL ORDER BY created_at,id",
            (user_id, draft_id),
        )

    def get_attachments(self, user_id: int, attachment_ids: Optional[list[str]] = None) -> list[dict[str, Any]]:
        if attachment_ids is None:
            return self.all("SELECT * FROM attachments WHERE user_id=? ORDER BY created_at,id", (user_id,))
        unique_ids = list(dict.fromkeys(str(item) for item in attachment_ids))
        if not unique_ids:
            return []
        placeholders = ",".join("?" for _ in unique_ids)
        rows = self.all(
            f"SELECT * FROM attachments WHERE user_id=? AND id IN ({placeholders}) ORDER BY created_at,id",
            (user_id, *unique_ids),
        )
        by_id = {row["id"]: row for row in rows}
        return [by_id[item] for item in unique_ids if item in by_id]

    def claim_attachments(self, user_id: int, attachment_ids: list[str], conversation_id: str, job_id: str) -> bool:
        unique_ids = list(dict.fromkeys(attachment_ids))
        if not unique_ids:
            return True
        placeholders = ",".join("?" for _ in unique_ids)
        with self.lock, self.connect() as connection:
            rows = connection.execute(
                f"SELECT id FROM attachments WHERE user_id=? AND job_id IS NULL AND id IN ({placeholders})",
                (user_id, *unique_ids),
            ).fetchall()
            if len(rows) != len(unique_ids):
                return False
            connection.execute(
                f"UPDATE attachments SET conversation_id=?,job_id=? WHERE user_id=? AND job_id IS NULL AND id IN ({placeholders})",
                (conversation_id, job_id, user_id, *unique_ids),
            )
            return True

    def attachments_for_job(self, user_id: int, job_id: str) -> list[dict[str, Any]]:
        return self.all(
            "SELECT * FROM attachments WHERE user_id=? AND job_id=? ORDER BY created_at,id",
            (user_id, job_id),
        )

    def delete_attachments(self, user_id: int, attachment_ids: Optional[list[str]] = None) -> list[dict[str, Any]]:
        with self.lock, self.connect() as connection:
            if attachment_ids is None:
                rows = connection.execute("SELECT * FROM attachments WHERE user_id=?", (user_id,)).fetchall()
                connection.execute("DELETE FROM attachments WHERE user_id=?", (user_id,))
            else:
                unique_ids = list(dict.fromkeys(str(item) for item in attachment_ids))
                if not unique_ids:
                    return []
                placeholders = ",".join("?" for _ in unique_ids)
                rows = connection.execute(
                    f"SELECT * FROM attachments WHERE user_id=? AND id IN ({placeholders})",
                    (user_id, *unique_ids),
                ).fetchall()
                connection.execute(
                    f"DELETE FROM attachments WHERE user_id=? AND id IN ({placeholders})",
                    (user_id, *unique_ids),
                )
            return [dict(row) for row in rows]

    def cleanup_expired_attachments(self, cutoff: int) -> list[dict[str, Any]]:
        with self.lock, self.connect() as connection:
            rows = connection.execute(
                """SELECT a.* FROM attachments a
                   LEFT JOIN jobs j ON j.id=a.job_id
                   WHERE a.created_at<? AND (a.job_id IS NULL OR j.status NOT IN ('queued','running'))""",
                (cutoff,),
            ).fetchall()
            if rows:
                connection.executemany("DELETE FROM attachments WHERE id=?", [(row["id"],) for row in rows])
            return [dict(row) for row in rows]

    def all_attachment_paths(self) -> set[str]:
        return {row["stored_path"] for row in self.all("SELECT stored_path FROM attachments")}

    def trim_conversations(
        self,
        user_id: int,
        *,
        protected_conversation_ids: Optional[list[str]] = None,
        conversation_limit: int = 100,
    ) -> list[str]:
        """Enforce a soft conversation limit without evicting active work."""
        with self.lock, self.connect() as connection:
            return self._trim_conversations(
                connection,
                user_id=user_id,
                protected_conversation_ids=protected_conversation_ids,
                conversation_limit=conversation_limit,
            )

    def _trim_conversations(
        self,
        connection: sqlite3.Connection,
        *,
        user_id: int,
        protected_conversation_ids: Optional[list[str]],
        conversation_limit: int,
    ) -> list[str]:
        """Delete the oldest inactive conversations on an existing transaction."""
        total = int(
            connection.execute("SELECT COUNT(*) FROM conversations WHERE user_id=?", (user_id,)).fetchone()[0]
        )
        excess = max(0, total - conversation_limit)
        if not excess:
            return []

        protected = list(dict.fromkeys(str(item) for item in protected_conversation_ids or [] if item))
        conditions = [
            "c.user_id=?",
            """NOT EXISTS (
                   SELECT 1 FROM jobs j
                   WHERE j.conversation_id=c.id AND j.status IN ('queued','running')
               )""",
        ]
        arguments: list[Any] = [user_id]
        if protected:
            placeholders = ",".join("?" for _ in protected)
            conditions.append(f"c.id NOT IN ({placeholders})")
            arguments.extend(protected)
        where = " AND ".join(conditions)
        surplus = connection.execute(
            f"""SELECT c.id FROM conversations c
                WHERE {where}
                ORDER BY c.updated_at ASC,c.id ASC LIMIT ?""",
            (*arguments, excess),
        ).fetchall()

        deleted: list[str] = []
        for row in surplus:
            conversation_id = str(row["id"])
            # Recheck activity in the DELETE itself.  This keeps the invariant
            # intact even if a second process begins a job after our candidate
            # selection but before this transaction obtains its write lock.
            cursor = connection.execute(
                """DELETE FROM conversations
                   WHERE id=? AND user_id=?
                     AND NOT EXISTS (
                         SELECT 1 FROM jobs
                         WHERE conversation_id=? AND status IN ('queued','running')
                     )""",
                (conversation_id, user_id, conversation_id),
            )
            if cursor.rowcount:
                deleted.append(conversation_id)
        return deleted

    def create_retry_branch(
        self,
        *,
        user_id: int,
        source_conversation_id: str,
        assistant_message_id: int,
        provider_id: int,
        provider_type: str,
        model: str,
        effort: str,
        timezone: str,
        conversation_id: str,
        job_id: str,
        created_at: int,
        conversation_limit: int = 100,
    ) -> dict[str, Any]:
        """Clone a conversation prefix and queue a fresh answer without a duplicate prompt.

        The source records, provider ownership, branch messages, job, and retention
        cleanup all live in one SQLite transaction.  In particular, retention is not
        allowed to evict either the source conversation or the new branch.
        """
        with self.lock, self.connect() as connection:
            source = connection.execute(
                "SELECT * FROM conversations WHERE id=? AND user_id=?",
                (source_conversation_id, user_id),
            ).fetchone()
            if not source:
                raise RetryBranchError("conversation_not_found")

            provider = connection.execute(
                "SELECT id FROM providers WHERE id=? AND user_id=?",
                (provider_id, user_id),
            ).fetchone()
            if not provider:
                raise RetryBranchError("provider_not_found")

            assistant_message = connection.execute(
                "SELECT * FROM messages WHERE id=? AND conversation_id=?",
                (assistant_message_id, source_conversation_id),
            ).fetchone()
            if not assistant_message:
                raise RetryBranchError("message_not_found")
            if assistant_message["role"] != "assistant":
                raise RetryBranchError("message_not_assistant")

            source_user_message = connection.execute(
                """SELECT * FROM messages
                   WHERE conversation_id=? AND role='user' AND id<?
                   ORDER BY id DESC LIMIT 1""",
                (source_conversation_id, assistant_message_id),
            ).fetchone()
            if not source_user_message:
                raise RetryBranchError("source_user_not_found")

            compact_title = " ".join(str(source_user_message["content"] or "").split())
            if len(compact_title) > 36:
                compact_title = compact_title[:36] + "…"
            branch_title = f"{compact_title or source['title'] or '重新回答'} · 重新回答"
            connection.execute(
                "INSERT INTO conversations(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)",
                (conversation_id, user_id, branch_title, created_at, created_at),
            )

            prefix = connection.execute(
                """SELECT role,content,meta_json,created_at FROM messages
                   WHERE conversation_id=? AND id<=? ORDER BY id""",
                (source_conversation_id, source_user_message["id"]),
            ).fetchall()
            cloned_messages: list[dict[str, Any]] = []
            attachments_omitted = False
            for message in prefix:
                meta = self.decode(message["meta_json"], {})
                if not isinstance(meta, dict):
                    meta = {}
                # Attachments belong to the source job/conversation and are never
                # silently replayed on retry.  Copying their display metadata would
                # make the branch imply that those files are still available.
                if meta.pop("attachments", None):
                    attachments_omitted = True
                meta_json = json.dumps(meta, ensure_ascii=False)
                cursor = connection.execute(
                    "INSERT INTO messages(conversation_id,role,content,meta_json,created_at) VALUES(?,?,?,?,?)",
                    (conversation_id, message["role"], message["content"], meta_json, message["created_at"]),
                )
                cloned_messages.append(
                    {
                        "id": int(cursor.lastrowid),
                        "role": message["role"],
                        "content": message["content"],
                        "meta_json": meta_json,
                        "created_at": message["created_at"],
                    }
                )

            connection.execute(
                """INSERT INTO jobs(
                       id,user_id,conversation_id,provider_id,provider_type,model,
                       effort,timezone,status,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    job_id,
                    user_id,
                    conversation_id,
                    provider_id,
                    provider_type,
                    model,
                    effort,
                    timezone,
                    "queued",
                    created_at,
                    created_at,
                ),
            )

            self._trim_conversations(
                connection,
                user_id=user_id,
                protected_conversation_ids=[source_conversation_id, conversation_id],
                conversation_limit=conversation_limit,
            )

            branch = connection.execute(
                "SELECT * FROM conversations WHERE id=? AND user_id=?",
                (conversation_id, user_id),
            ).fetchone()
            job = connection.execute("SELECT * FROM jobs WHERE id=? AND user_id=?", (job_id, user_id)).fetchone()
            return {
                "conversation": dict(branch),
                "messages": cloned_messages,
                "job": dict(job),
                "attachments_omitted": attachments_omitted,
            }

    def clear_user_chats(self, user_id: int) -> tuple[int, bool]:
        """Delete one user's chat graph and compact SQLite when it is safe."""
        with self.lock:
            with self.connect() as db:
                active = db.execute("SELECT 1 FROM jobs WHERE status IN ('queued','running') LIMIT 1").fetchone()
                if active:
                    return 0, False
                count = int(db.execute("SELECT COUNT(*) FROM conversations WHERE user_id=?", (user_id,)).fetchone()[0])
                db.execute("DELETE FROM attachments WHERE user_id=?", (user_id,))
                db.execute("DELETE FROM conversations WHERE user_id=?", (user_id,))
            # DELETE alone does not shrink a SQLite file. Keep application
            # writes paused while reclaiming both WAL and database space.
            with self.connect() as db:
                db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                db.execute("VACUUM")
            return count, True

    @staticmethod
    def decode(value: str, fallback: Any) -> Any:
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return fallback

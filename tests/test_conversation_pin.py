"""Offline regression tests for pinning conversations in history."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import attachments, main
from app.config import Settings
from app.db import Database


class ConversationPinTests(unittest.TestCase):
    admin_password = "pin-test-password"

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name) / "data"
        self.original_db = main.db
        self.original_settings = main.settings
        self.original_secret = main.secret
        self.original_tasks = main.tasks
        self.original_cleanup_task = main.attachment_cleanup_task
        self.original_attachments_dir = attachments.ATTACHMENTS_DIR
        self.environment = patch.dict(
            os.environ,
            {"ADMIN_USERNAME": "pin-test-admin", "ADMIN_PASSWORD": self.admin_password},
            clear=False,
        )
        self.environment.start()

        main.settings = Settings(data_dir=self.data_dir)
        main.db = Database(main.settings.db_path)
        main.secret = b""
        main.tasks = {}
        main.attachment_cleanup_task = None
        attachments.ATTACHMENTS_DIR = self.data_dir / "attachments"
        self.client = TestClient(main.app)
        self.client.__enter__()
        login = self.client.post(
            "/api/login",
            json={"username": "pin-test-admin", "password": self.admin_password},
        )
        self.assertEqual(login.status_code, 200, login.text)
        self.user_id = int(login.json()["id"])

    def tearDown(self) -> None:
        try:
            self.client.__exit__(None, None, None)
        finally:
            main.db = self.original_db
            main.settings = self.original_settings
            main.secret = self.original_secret
            main.tasks = self.original_tasks
            main.attachment_cleanup_task = self.original_cleanup_task
            attachments.ATTACHMENTS_DIR = self.original_attachments_dir
            self.environment.stop()
            self.temp_dir.cleanup()

    def add_conversation(self, conversation_id: str, title: str, updated_at: int, pinned_at: int | None = None) -> None:
        main.db.run(
            "INSERT INTO conversations(id,user_id,title,created_at,updated_at,pinned_at) VALUES(?,?,?,?,?,?)",
            (conversation_id, self.user_id, title, updated_at, updated_at, pinned_at),
        )

    def history_ids(self) -> list[str]:
        return [item["id"] for item in self.client.get("/api/conversations?page=1").json()["items"]]

    def test_pinned_chats_stay_above_newer_unpinned_chats(self) -> None:
        self.add_conversation("old-pinned", "old pinned", 100, pinned_at=150)
        self.add_conversation("newest", "newest", 300)
        self.add_conversation("middle", "middle", 200)

        ids = self.history_ids()
        self.assertEqual(ids, ["old-pinned", "newest", "middle"])
        first = self.client.get("/api/conversations?page=1").json()["items"][0]
        self.assertTrue(first["pinned"])
        self.assertEqual(first["pinned_at"], 150)

    def test_multiple_pins_keep_recent_pin_first(self) -> None:
        self.add_conversation("a", "A", 100)
        self.add_conversation("b", "B", 200)
        self.add_conversation("c", "C", 300)
        first = self.client.post("/api/conversations/a/pin", json={"pinned": True})
        second = self.client.post("/api/conversations/c/pin", json={"pinned": True})
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 200, second.text)
        self.assertTrue(first.json()["pinned"])
        self.assertTrue(second.json()["pinned"])
        self.assertEqual(self.history_ids()[:2], ["c", "a"])
        self.assertEqual(self.history_ids()[2], "b")

    def test_unpin_moves_chat_to_the_front_of_regular_history(self) -> None:
        self.add_conversation("pinned", "pinned", 100, pinned_at=400)
        self.add_conversation("fresh", "fresh", 300)
        self.add_conversation("older", "older", 200)
        response = self.client.post("/api/conversations/pinned/pin", json={"pinned": False})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()["pinned"])
        self.assertIsNone(response.json()["pinned_at"])
        self.assertEqual(self.history_ids()[0], "pinned")
        self.assertGreater(response.json()["updated_at"], 300)

    def test_init_adds_pinned_at_to_legacy_conversation_tables(self) -> None:
        legacy = Database(self.data_dir / "legacy.db")
        with legacy.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE conversations (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                INSERT INTO conversations(id,user_id,title,created_at,updated_at)
                VALUES('legacy',1,'legacy',1,1);
                """
            )
        legacy.init()
        columns = {row["name"] for row in legacy.all("PRAGMA table_info(conversations)")}
        self.assertIn("pinned_at", columns)
        self.assertEqual(legacy.one("SELECT COUNT(*) AS n FROM conversations")["n"], 1)

    def test_trim_never_deletes_pinned_conversations(self) -> None:
        self.add_conversation("keep-pinned", "keep", 1, pinned_at=50)
        for index in range(100):
            self.add_conversation(f"regular-{index:03d}", f"regular {index}", 100 + index)
        self.add_conversation("overflow", "overflow", 1000)
        main.trim_old_conversations(self.user_id)
        remaining = {row["id"] for row in main.db.all("SELECT id FROM conversations WHERE user_id=?", (self.user_id,))}
        self.assertIn("keep-pinned", remaining)
        self.assertIn("overflow", remaining)
        self.assertNotIn("regular-000", remaining)
        self.assertEqual(len(remaining), 101)


if __name__ == "__main__":
    unittest.main()

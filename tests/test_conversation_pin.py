"""Offline regression tests for conversation pinning."""

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
    password = "pin-test-password"

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name) / "data"
        self.originals = (
            main.db,
            main.settings,
            main.secret,
            main.tasks,
            main.attachment_cleanup_task,
            attachments.ATTACHMENTS_DIR,
        )
        self.environment = patch.dict(
            os.environ,
            {"ADMIN_USERNAME": "pin-test-admin", "ADMIN_PASSWORD": self.password},
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
        response = self.client.post(
            "/api/login",
            json={"username": "pin-test-admin", "password": self.password},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.user_id = int(response.json()["id"])

    def tearDown(self) -> None:
        try:
            self.client.__exit__(None, None, None)
        finally:
            (
                main.db,
                main.settings,
                main.secret,
                main.tasks,
                main.attachment_cleanup_task,
                attachments.ATTACHMENTS_DIR,
            ) = self.originals
            self.environment.stop()
            self.temp_dir.cleanup()

    def add_conversation(self, conversation_id: str, updated_at: int, pinned_at: int | None = None) -> None:
        main.db.run(
            "INSERT INTO conversations(id,user_id,title,created_at,updated_at,pinned_at) VALUES(?,?,?,?,?,?)",
            (conversation_id, self.user_id, conversation_id, updated_at, updated_at, pinned_at),
        )

    def history(self) -> list[dict]:
        response = self.client.get("/api/conversations?page=1")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["items"]

    def test_pinned_chats_are_first_and_recent_pins_win(self) -> None:
        self.add_conversation("old-pin", 100, 300)
        self.add_conversation("new-pin", 200, 400)
        self.add_conversation("latest-chat", 500)

        items = self.history()

        self.assertEqual([item["id"] for item in items], ["new-pin", "old-pin", "latest-chat"])
        self.assertEqual([item["pinned"] for item in items], [True, True, False])

    def test_unpin_does_not_fabricate_recent_activity(self) -> None:
        self.add_conversation("old", 100, 400)
        self.add_conversation("new", 300)

        response = self.client.post("/api/conversations/old/pin", json={"pinned": False})

        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()["pinned"])
        self.assertEqual(response.json()["updated_at"], 100)
        self.assertEqual([item["id"] for item in self.history()], ["new", "old"])

    def test_pin_is_idempotent(self) -> None:
        self.add_conversation("chat", 100)
        first = self.client.post("/api/conversations/chat/pin", json={"pinned": True})
        second = self.client.post("/api/conversations/chat/pin", json={"pinned": True})

        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(first.json()["pinned_at"], second.json()["pinned_at"])

    def test_pruning_keeps_pins_and_the_newest_100_regular_chats(self) -> None:
        self.add_conversation("pinned", 1, 10)
        for index in range(101):
            self.add_conversation(f"regular-{index:03d}", 100 + index)

        main.trim_old_conversations(self.user_id)

        remaining = {
            row["id"] for row in main.db.all("SELECT id FROM conversations WHERE user_id=?", (self.user_id,))
        }
        self.assertIn("pinned", remaining)
        self.assertNotIn("regular-000", remaining)
        self.assertEqual(len(remaining), 101)

    def test_init_migrates_an_existing_conversation_table(self) -> None:
        legacy = Database(self.data_dir / "legacy.db")
        with legacy.connect() as connection:
            connection.executescript(
                """CREATE TABLE conversations (
                       id TEXT PRIMARY KEY,
                       user_id INTEGER NOT NULL,
                       title TEXT NOT NULL,
                       created_at INTEGER NOT NULL,
                       updated_at INTEGER NOT NULL
                   );"""
            )

        legacy.init()

        columns = {row["name"] for row in legacy.all("PRAGMA table_info(conversations)")}
        self.assertIn("pinned_at", columns)


if __name__ == "__main__":
    unittest.main()

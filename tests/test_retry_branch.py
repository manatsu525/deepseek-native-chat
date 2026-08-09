"""Offline regression coverage for retry branches.

These tests deliberately replace both provider streaming entry points and
guard ``httpx.AsyncClient``.  They exercise the actual FastAPI route against a
temporary SQLite database, so running them never spends a provider API credit
or touches the deployment's data directory.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import attachments, main
from app.config import Settings
from app.db import Database


class RetryBranchTests(unittest.TestCase):
    """Exercise retry through HTTP while keeping every external call local."""

    admin_password = "retry-test-password"

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
            {"ADMIN_USERNAME": "retry-test-admin", "ADMIN_PASSWORD": self.admin_password},
            clear=False,
        )
        self.environment.start()

        # ``lifespan`` reads these globals.  Point both chat data and the
        # periodic attachment cleanup at the temporary directory before the
        # TestClient starts the application.
        main.settings = Settings(data_dir=self.data_dir)
        main.db = Database(main.settings.db_path)
        main.secret = b""
        main.tasks = {}
        main.attachment_cleanup_task = None
        attachments.ATTACHMENTS_DIR = self.data_dir / "attachments"

        self.client = TestClient(main.app)
        self.client.__enter__()
        self.user_id = self._login()

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

    def _login(self) -> int:
        response = self.client.post(
            "/api/login",
            json={"username": "retry-test-admin", "password": self.admin_password},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return int(response.json()["id"])

    def _add_provider(self, user_id: int, name: str, model: str) -> int:
        return main.db.run(
            """INSERT INTO providers(
                   user_id,name,api_key,base_url,model,provider_type,settings_json,created_at
               ) VALUES(?,?,?,?,?,?,?,?)""",
            (
                user_id,
                name,
                f"{name}-token",
                "https://provider.invalid/v1",
                model,
                "custom",
                json.dumps({"models": [model]}),
                100,
            ),
        )

    def _seed_source_thread(self, user_id: int) -> tuple[str, int]:
        source_id = "source-conversation"
        main.db.run(
            "INSERT INTO conversations(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)",
            (source_id, user_id, "Original thread", 100, 100),
        )
        self._message(source_id, "user", "first question", 101)
        self._message(source_id, "assistant", "first answer", 102, {"model": "old-model"})
        self._message(source_id, "user", "second question", 103)
        target_id = self._message(source_id, "assistant", "old second answer", 104, {"model": "old-model"})
        return source_id, target_id

    @staticmethod
    def _message(
        conversation_id: str,
        role: str,
        content: str,
        created_at: int,
        meta: dict[str, object] | None = None,
    ) -> int:
        return main.db.run(
            "INSERT INTO messages(conversation_id,role,content,meta_json,created_at) VALUES(?,?,?,?,?)",
            (conversation_id, role, content, json.dumps(meta or {}), created_at),
        )

    @staticmethod
    def _messages(conversation_id: str) -> list[dict[str, object]]:
        return main.db.all(
            "SELECT id,role,content,meta_json,created_at FROM messages WHERE conversation_id=? ORDER BY id",
            (conversation_id,),
        )

    def _retry(
        self,
        conversation_id: str,
        message_id: int,
        provider_id: int,
        model: str,
    ):
        return self.client.post(
            f"/api/conversations/{conversation_id}/retry",
            json={
                "message_id": message_id,
                "provider_id": provider_id,
                "model": model,
                "effort": "max",
                "timezone": "UTC",
            },
        )

    def _wait_for_job(self, job_id: str) -> dict[str, object]:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            job = main.db.one("SELECT * FROM jobs WHERE id=?", (job_id,))
            if job and job["status"] not in {"queued", "running"}:
                return job
            time.sleep(0.01)
        self.fail(f"mocked retry job {job_id} did not finish")

    def test_retry_uses_prefix_once_preserves_source_and_uses_current_provider(self) -> None:
        self._add_provider(self.user_id, "old-provider", "old-model")
        current_provider_id = self._add_provider(self.user_id, "current-provider", "current-model")
        source_id, target_id = self._seed_source_thread(self.user_id)
        source_before = self._messages(source_id)
        conversation_before = main.db.one("SELECT * FROM conversations WHERE id=?", (source_id,))
        custom_calls: list[dict[str, object]] = []
        deepseek_calls: list[dict[str, object]] = []

        async def fake_custom_stream_response(**kwargs: object) -> dict[str, object]:
            custom_calls.append(copy.deepcopy(kwargs))
            update = kwargs["update"]
            await update({"answer": "mock retry answer", "reasoning": "", "searches": [], "sources": [], "usage": {}})
            return {"answer": "mock retry answer", "reasoning": "", "searches": [], "sources": [], "usage": {}}

        async def fake_deepseek_stream_response(**kwargs: object) -> dict[str, object]:
            deepseek_calls.append(copy.deepcopy(kwargs))
            raise AssertionError("retry selected the wrong provider stream")

        # No real model gateway can be reached: both entry points are fake and
        # the only HTTP client used by this module raises if it is instantiated.
        with patch.object(main, "custom_stream_response", new=fake_custom_stream_response), patch.object(
            main, "deepseek_stream_response", new=fake_deepseek_stream_response
        ), patch.object(main.httpx, "AsyncClient", side_effect=AssertionError("external HTTP is forbidden in retry tests")):
            response = self._retry(source_id, target_id, current_provider_id, "current-model")
            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            branch_id = payload["conversation"]["id"]
            self.assertNotEqual(branch_id, source_id)
            self.assertFalse(payload["attachments_omitted"])
            job_rows = main.db.all("SELECT * FROM jobs WHERE conversation_id=?", (branch_id,))
            self.assertEqual(len(job_rows), 1)
            job = self._wait_for_job(str(job_rows[0]["id"]))

        # The source thread is untouched.  The branch is exactly the prefix
        # ending at the original user prompt, followed by the mocked answer.
        self.assertEqual(self._messages(source_id), source_before)
        self.assertEqual(main.db.one("SELECT * FROM conversations WHERE id=?", (source_id,)), conversation_before)
        branch_messages = self._messages(str(branch_id))
        self.assertEqual(
            [(row["role"], row["content"]) for row in branch_messages],
            [
                ("user", "first question"),
                ("assistant", "first answer"),
                ("user", "second question"),
                ("assistant", "mock retry answer"),
            ],
        )

        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["provider_id"], current_provider_id)
        self.assertEqual(job["model"], "current-model")
        self.assertEqual(len(custom_calls), 1)
        self.assertEqual(deepseek_calls, [])
        request = custom_calls[0]
        self.assertEqual(request["api_key"], "current-provider-token")
        self.assertEqual(request["base_url"], "https://provider.invalid/v1")
        self.assertEqual(request["model"], "current-model")
        self.assertEqual(request["effort"], "max")
        self.assertEqual(request["user_timezone"], "UTC")
        history = request["messages"]
        self.assertEqual(
            [(item["role"], item["content"]) for item in history],
            [("user", "first question"), ("assistant", "first answer"), ("user", "second question")],
        )
        self.assertEqual(sum(item["content"] == "second question" for item in history), 1)

    def test_retry_requires_auth_and_rejects_missing_target_without_creating_a_job(self) -> None:
        provider_id = self._add_provider(self.user_id, "current-provider", "current-model")
        source_id, _ = self._seed_source_thread(self.user_id)

        self.client.cookies.clear()
        anonymous = self._retry(source_id, 999999, provider_id, "current-model")
        self.assertEqual(anonymous.status_code, 401, anonymous.text)

        self._login()
        invalid = self._retry(source_id, 999999, provider_id, "current-model")
        self.assertEqual(invalid.status_code, 404, invalid.text)
        self.assertEqual(main.db.one("SELECT COUNT(*) AS n FROM jobs")["n"], 0)
        self.assertEqual(main.db.one("SELECT COUNT(*) AS n FROM conversations")["n"], 1)

    def test_retrying_an_earlier_answer_excludes_later_turns_without_duplicate_prompt(self) -> None:
        provider_id = self._add_provider(self.user_id, "current-provider", "current-model")
        source_id, earlier_answer_id = self._seed_source_thread(self.user_id)
        self._message(source_id, "user", "third question", 105)
        self._message(source_id, "assistant", "third answer", 106, {"model": "old-model"})

        # Call the SQLite transaction directly: this verifies the exact branch
        # boundary without starting a provider stream or making an HTTP request.
        branch = main.db.create_retry_branch(
            user_id=self.user_id,
            source_conversation_id=source_id,
            assistant_message_id=earlier_answer_id,
            provider_id=provider_id,
            provider_type="custom",
            model="current-model",
            effort="high",
            timezone="UTC",
            conversation_id="earlier-answer-branch",
            job_id="earlier-answer-job",
            created_at=200,
        )

        branch_messages = self._messages(str(branch["conversation"]["id"]))
        self.assertEqual(
            [(row["role"], row["content"]) for row in branch_messages],
            [("user", "first question"), ("assistant", "first answer"), ("user", "second question")],
        )
        contents = [str(row["content"]) for row in branch_messages]
        self.assertEqual(contents.count("second question"), 1)
        self.assertNotIn("old second answer", contents)
        self.assertNotIn("third question", contents)
        self.assertNotIn("third answer", contents)
        self.assertEqual(main.db.one("SELECT status FROM jobs WHERE id='earlier-answer-job'")["status"], "queued")

    def test_retry_retention_never_evicts_an_active_conversation(self) -> None:
        provider_id = self._add_provider(self.user_id, "current-provider", "current-model")
        source_id, target_id = self._seed_source_thread(self.user_id)
        main.db.run(
            "INSERT INTO conversations(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)",
            ("active-old", self.user_id, "Active", 1, 1),
        )
        main.db.run(
            """INSERT INTO jobs(
                   id,user_id,conversation_id,provider_id,provider_type,model,
                   effort,timezone,status,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            ("active-job", self.user_id, "active-old", provider_id, "custom", "current-model", "high", "UTC", "running", 1, 1),
        )
        # source + active + 98 inactive threads = 100.  The retry makes 101,
        # so the oldest *inactive* thread must be pruned instead of active-old.
        for index in range(98):
            main.db.run(
                "INSERT INTO conversations(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)",
                (f"inactive-{index}", self.user_id, "Inactive", index + 2, index + 2),
            )

        branch = main.db.create_retry_branch(
            user_id=self.user_id,
            source_conversation_id=source_id,
            assistant_message_id=target_id,
            provider_id=provider_id,
            provider_type="custom",
            model="current-model",
            effort="high",
            timezone="UTC",
            conversation_id="retry-branch",
            job_id="retry-job",
            created_at=1000,
        )

        self.assertEqual(main.db.one("SELECT COUNT(*) AS n FROM conversations")["n"], 100)
        self.assertIsNotNone(main.db.one("SELECT id FROM conversations WHERE id='active-old'"))
        self.assertIsNotNone(main.db.one("SELECT id FROM jobs WHERE id='active-job'"))
        self.assertIsNotNone(main.db.one("SELECT id FROM conversations WHERE id=?", (source_id,)))
        self.assertIsNotNone(main.db.one("SELECT id FROM conversations WHERE id=?", (branch["conversation"]["id"],)))
        self.assertIsNone(main.db.one("SELECT id FROM conversations WHERE id='inactive-0'"))

    def test_new_chat_retention_never_evicts_an_active_conversation(self) -> None:
        provider_id = self._add_provider(self.user_id, "current-provider", "current-model")
        main.db.run(
            "INSERT INTO conversations(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)",
            ("active-old", self.user_id, "Active", 1, 1),
        )
        main.db.run(
            """INSERT INTO jobs(
                   id,user_id,conversation_id,provider_id,provider_type,model,
                   effort,timezone,status,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            ("active-job", self.user_id, "active-old", provider_id, "custom", "current-model", "high", "UTC", "queued", 1, 1),
        )
        for index in range(99):
            main.db.run(
                "INSERT INTO conversations(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)",
                (f"inactive-{index}", self.user_id, "Inactive", index + 2, index + 2),
            )

        async def fake_custom_stream_response(**_: object) -> dict[str, object]:
            return {"answer": "mock new-chat answer", "reasoning": "", "searches": [], "sources": [], "usage": {}}

        with patch.object(main, "custom_stream_response", new=fake_custom_stream_response), patch.object(
            main.httpx, "AsyncClient", side_effect=AssertionError("external HTTP is forbidden in retry tests")
        ):
            response = self.client.post(
                "/api/chat",
                json={
                    "content": "new question",
                    "attachment_ids": [],
                    "provider_id": provider_id,
                    "model": "current-model",
                    "effort": "high",
                    "timezone": "UTC",
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            self._wait_for_job(str(response.json()["job_id"]))

        self.assertEqual(main.db.one("SELECT COUNT(*) AS n FROM conversations")["n"], 100)
        self.assertIsNotNone(main.db.one("SELECT id FROM conversations WHERE id='active-old'"))
        self.assertIsNotNone(main.db.one("SELECT id FROM jobs WHERE id='active-job'"))
        self.assertIsNone(main.db.one("SELECT id FROM conversations WHERE id='inactive-0'"))


if __name__ == "__main__":
    unittest.main()

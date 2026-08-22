"""Offline regression tests for regenerating any answer in place."""

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


class RetryInPlaceTests(unittest.TestCase):
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
            json={"username": "retry-test-admin", "password": self.admin_password},
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

    def add_provider(self, name: str, model: str) -> int:
        return main.db.run(
            """INSERT INTO providers(
                   user_id,name,api_key,base_url,model,provider_type,settings_json,created_at
               ) VALUES(?,?,?,?,?,?,?,?)""",
            (
                self.user_id,
                name,
                f"{name}-token",
                "https://provider.invalid/v1",
                model,
                "custom",
                json.dumps({"models": [model]}),
                100,
            ),
        )

    def add_message(self, conversation_id: str, role: str, content: str, created_at: int, meta: dict | None = None) -> int:
        return main.db.run(
            "INSERT INTO messages(conversation_id,role,content,meta_json,created_at) VALUES(?,?,?,?,?)",
            (conversation_id, role, content, json.dumps(meta or {}), created_at),
        )

    def seed_conversation(self, *, attachment: bool = False) -> str:
        conversation_id = "same-conversation"
        main.db.run(
            "INSERT INTO conversations(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)",
            (conversation_id, self.user_id, "Original", 100, 104),
        )
        self.add_message(conversation_id, "user", "first question", 101)
        self.add_message(conversation_id, "assistant", "first answer", 102, {"model": "old-model"})
        prompt_meta = {"attachments": [{"id": "gone", "name": "old.pdf"}]} if attachment else {}
        self.add_message(conversation_id, "user", "retry this question", 103, prompt_meta)
        self.add_message(conversation_id, "assistant", "answer to remove", 104, {"model": "old-model"})
        return conversation_id

    def retry(self, conversation_id: str, prompt_message_id: int, provider_id: int, model: str):
        return self.client.post(
            f"/api/conversations/{conversation_id}/retry",
            json={
                "prompt_message_id": prompt_message_id,
                "provider_id": provider_id,
                "model": model,
                "effort": "max",
                "timezone": "UTC",
            },
        )

    def wait_for_job(self, job_id: str) -> dict:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            job = main.db.one("SELECT * FROM jobs WHERE id=?", (job_id,))
            if job and job["status"] not in {"queued", "running"}:
                return job
            time.sleep(0.01)
        self.fail(f"mock job {job_id} did not finish")

    def test_retry_replaces_latest_answer_without_duplicating_the_question(self) -> None:
        self.add_provider("old-provider", "old-model")
        provider_id = self.add_provider("current-provider", "current-model")
        conversation_id = self.seed_conversation()
        calls: list[dict] = []

        async def fake_custom_stream_response(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            return {
                "answer": "replacement answer",
                "reasoning": "",
                "searches": [],
                "sources": [],
                "usage": {},
            }

        with patch.object(main, "custom_stream_response", new=fake_custom_stream_response), patch.object(
            main.httpx,
            "AsyncClient",
            side_effect=AssertionError("external HTTP is forbidden in retry tests"),
        ):
            prompt = main.db.one(
                "SELECT id FROM messages WHERE conversation_id=? AND content=?",
                (conversation_id, "retry this question"),
            )
            response = self.retry(conversation_id, prompt["id"], provider_id, "current-model")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["conversation_id"], conversation_id)
            job = self.wait_for_job(response.json()["job_id"])

        messages = main.db.all(
            "SELECT role,content FROM messages WHERE conversation_id=? ORDER BY id",
            (conversation_id,),
        )
        self.assertEqual(
            [(item["role"], item["content"]) for item in messages],
            [
                ("user", "first question"),
                ("assistant", "first answer"),
                ("user", "retry this question"),
                ("assistant", "replacement answer"),
            ],
        )
        self.assertEqual(sum(item["content"] == "retry this question" for item in messages), 1)
        self.assertNotIn("answer to remove", [item["content"] for item in messages])
        self.assertEqual(main.db.one("SELECT COUNT(*) AS n FROM conversations")["n"], 1)
        self.assertEqual(job["provider_id"], provider_id)
        self.assertEqual(job["model"], "current-model")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["api_key"], "current-provider-token")
        self.assertEqual(calls[0]["base_url"], "https://provider.invalid/v1")
        self.assertEqual(calls[0]["effort"], "max")
        self.assertEqual(
            [(item["role"], item["content"]) for item in calls[0]["messages"]],
            [
                ("user", "first question"),
                ("assistant", "first answer"),
                ("user", "retry this question"),
            ],
        )

    def test_retry_with_expired_historical_attachment_keeps_the_old_answer(self) -> None:
        provider_id = self.add_provider("current-provider", "current-model")
        conversation_id = self.seed_conversation(attachment=True)

        prompt = main.db.one(
            "SELECT id FROM messages WHERE conversation_id=? AND content=?",
            (conversation_id, "retry this question"),
        )
        response = self.retry(conversation_id, prompt["id"], provider_id, "current-model")

        self.assertEqual(response.status_code, 409, response.text)
        self.assertIn("附件", response.json()["detail"])
        latest = main.db.one(
            "SELECT role,content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT 1",
            (conversation_id,),
        )
        self.assertEqual((latest["role"], latest["content"]), ("assistant", "answer to remove"))
        self.assertEqual(main.db.one("SELECT COUNT(*) AS n FROM jobs")["n"], 0)

    def test_retry_reuses_a_retained_attachment(self) -> None:
        provider_id = self.add_provider("current-provider", "current-model")
        conversation_id = "attachment-conversation"
        main.db.run(
            "INSERT INTO conversations(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)",
            (conversation_id, self.user_id, "Attachment", 100, 103),
        )
        main.db.run(
            """INSERT INTO jobs(
                   id,user_id,conversation_id,provider_id,provider_type,model,
                   effort,timezone,status,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            ("original-job", self.user_id, conversation_id, provider_id, "custom", "current-model", "high", "UTC", "completed", 100, 100),
        )
        stored = attachments.ATTACHMENTS_DIR / str(self.user_id) / "kept.txt"
        stored.parent.mkdir(parents=True, exist_ok=True)
        stored.write_text("retained attachment body", encoding="utf-8")
        main.db.create_attachment(
            "a" * 32,
            self.user_id,
            "old-draft",
            "notes.txt",
            "document",
            "text/plain",
            str(stored),
            24,
            24,
            100,
        )
        main.db.run(
            "UPDATE attachments SET conversation_id=?,job_id=? WHERE id=?",
            (conversation_id, "original-job", "a" * 32),
        )
        prompt_id = self.add_message(
            conversation_id,
            "user",
            "answer from this file",
            101,
            {"attachments": [{"id": "a" * 32, "name": "notes.txt", "kind": "document"}]},
        )
        self.add_message(conversation_id, "assistant", "old answer", 102)
        histories: list[list[dict]] = []

        async def fake_custom_stream_response(**kwargs):
            histories.append(copy.deepcopy(kwargs["messages"]))
            return {"answer": "new answer", "reasoning": "", "searches": [], "sources": [], "usage": {}}

        with patch.object(main, "custom_stream_response", new=fake_custom_stream_response):
            response = self.retry(conversation_id, prompt_id, provider_id, "current-model")
            self.assertEqual(response.status_code, 200, response.text)
            self.wait_for_job(response.json()["job_id"])

        self.assertIn("retained attachment body", histories[0][-1]["content"])
        retained = main.db.one("SELECT job_id FROM attachments WHERE id=?", ("a" * 32,))
        self.assertEqual(retained["job_id"], response.json()["job_id"])
        self.assertTrue(stored.is_file())

    def test_retry_earlier_prompt_discards_every_message_below_it(self) -> None:
        provider_id = self.add_provider("current-provider", "current-model")
        conversation_id = self.seed_conversation()
        first_prompt = main.db.one(
            "SELECT id FROM messages WHERE conversation_id=? AND content=?",
            (conversation_id, "first question"),
        )
        histories: list[list[dict]] = []

        async def fake_custom_stream_response(**kwargs):
            histories.append(copy.deepcopy(kwargs["messages"]))
            return {"answer": "new first answer", "reasoning": "", "searches": [], "sources": [], "usage": {}}

        with patch.object(main, "custom_stream_response", new=fake_custom_stream_response), patch.object(
            main.httpx,
            "AsyncClient",
            side_effect=AssertionError("external HTTP is forbidden in retry tests"),
        ):
            response = self.retry(conversation_id, first_prompt["id"], provider_id, "current-model")
            self.assertEqual(response.status_code, 200, response.text)
            self.wait_for_job(response.json()["job_id"])

        messages = main.db.all(
            "SELECT role,content FROM messages WHERE conversation_id=? ORDER BY id",
            (conversation_id,),
        )
        self.assertEqual(
            [(item["role"], item["content"]) for item in messages],
            [("user", "first question"), ("assistant", "new first answer")],
        )
        self.assertEqual(
            [(item["role"], item["content"]) for item in histories[0]],
            [("user", "first question")],
        )

    def test_failed_prompt_without_assistant_can_be_retried(self) -> None:
        provider_id = self.add_provider("current-provider", "current-model")
        conversation_id = "failed-conversation"
        main.db.run(
            "INSERT INTO conversations(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)",
            (conversation_id, self.user_id, "Failed", 100, 101),
        )
        prompt_id = self.add_message(conversation_id, "user", "failed question", 101)
        main.db.run(
            """INSERT INTO jobs(
                   id,user_id,conversation_id,provider_id,provider_type,model,
                   effort,timezone,status,error,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "old-failed-job", self.user_id, conversation_id, provider_id, "custom",
                "current-model", "high", "UTC", "failed", "upstream failed", 102, 102,
            ),
        )

        async def fake_custom_stream_response(**kwargs):
            return {"answer": "recovered answer", "reasoning": "", "searches": [], "sources": [], "usage": {}}

        with patch.object(main, "custom_stream_response", new=fake_custom_stream_response), patch.object(
            main.httpx,
            "AsyncClient",
            side_effect=AssertionError("external HTTP is forbidden in retry tests"),
        ):
            response = self.retry(conversation_id, prompt_id, provider_id, "current-model")
            self.assertEqual(response.status_code, 200, response.text)
            self.wait_for_job(response.json()["job_id"])

        messages = main.db.all(
            "SELECT role,content FROM messages WHERE conversation_id=? ORDER BY id",
            (conversation_id,),
        )
        self.assertEqual(
            [(item["role"], item["content"]) for item in messages],
            [("user", "failed question"), ("assistant", "recovered answer")],
        )

    def test_failed_answer_is_persisted_but_excluded_from_later_model_context(self) -> None:
        provider_id = self.add_provider("current-provider", "current-model")
        conversation_id = "persisted-failure"
        main.db.run(
            "INSERT INTO conversations(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)",
            (conversation_id, self.user_id, "Failure", 100, 100),
        )
        prompt_id = self.add_message(conversation_id, "user", "question that fails", 101)

        async def failing_response(**kwargs):
            await kwargs["update"](
                {
                    "answer": "已经查到一部分资料。",
                    "reasoning": "partial reasoning",
                    "searches": [{"id": "search-1", "status": "completed", "action": "search"}],
                    "sources": [{"url": "https://example.com", "title": "Example"}],
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }
            )
            raise RuntimeError("upstream exploded")

        with patch.object(main, "custom_stream_response", new=failing_response):
            response = self.retry(conversation_id, prompt_id, provider_id, "current-model")
            self.assertEqual(response.status_code, 200, response.text)
            failed_job = self.wait_for_job(response.json()["job_id"])

        self.assertEqual(failed_job["status"], "failed")
        persisted = main.db.one(
            "SELECT content,meta_json FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT 1",
            (conversation_id,),
        )
        self.assertEqual(persisted["content"], "已经查到一部分资料。")
        meta = json.loads(persisted["meta_json"])
        self.assertTrue(meta["failed"])
        self.assertEqual(meta["error"], "upstream exploded")
        self.assertEqual(meta["reasoning"], "partial reasoning")

        loaded = self.client.get(f"/api/conversations/{conversation_id}")
        self.assertEqual(loaded.status_code, 200, loaded.text)
        self.assertIsNone(loaded.json()["active_job"])
        self.assertEqual(loaded.json()["messages"][-1]["meta"]["error"], "upstream exploded")

        later_histories: list[list[dict]] = []
        later_prompt = self.add_message(conversation_id, "user", "follow-up question", 102)

        async def successful_response(**kwargs):
            later_histories.append(copy.deepcopy(kwargs["messages"]))
            return {"answer": "follow-up answer", "reasoning": "", "searches": [], "sources": [], "usage": {}}

        with patch.object(main, "custom_stream_response", new=successful_response):
            response = self.retry(conversation_id, later_prompt, provider_id, "current-model")
            self.assertEqual(response.status_code, 200, response.text)
            self.wait_for_job(response.json()["job_id"])

        self.assertEqual(
            [(item["role"], item["content"]) for item in later_histories[0]],
            [("user", "question that fails"), ("user", "follow-up question")],
        )


if __name__ == "__main__":
    unittest.main()

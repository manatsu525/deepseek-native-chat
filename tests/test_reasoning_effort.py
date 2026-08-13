"""Offline checks for the five-level reasoning effort control."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import attachments, main
from app.config import Settings
from app.db import Database
from app.mimo import DEFAULT_REASONING_EFFORT, REASONING_EFFORTS
from app.mimo_local import _apply_thinking_options


class ReasoningEffortTests(unittest.TestCase):
    admin_password = "effort-test-password"

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
            {"ADMIN_USERNAME": "effort-test-admin", "ADMIN_PASSWORD": self.admin_password},
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
            json={"username": "effort-test-admin", "password": self.admin_password},
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

    def add_provider(self) -> int:
        return main.db.run(
            """INSERT INTO providers(
                   user_id,name,api_key,base_url,model,provider_type,settings_json,created_at
               ) VALUES(?,?,?,?,?,?,?,?)""",
            (
                self.user_id,
                "effort-provider",
                "effort-token-xx",
                "https://provider.invalid/v1",
                "ordinary-model",
                "custom",
                json.dumps({"models": ["ordinary-model"]}),
                100,
            ),
        )

    def test_five_levels_are_low_through_max_and_default_is_high(self) -> None:
        self.assertEqual(REASONING_EFFORTS, ("low", "medium", "high", "xhigh", "max"))
        self.assertEqual(DEFAULT_REASONING_EFFORT, "high")
        self.assertEqual(main.validate_effort("low"), "low")
        self.assertEqual(main.validate_effort("xhigh"), "xhigh")
        with self.assertRaises(Exception):
            main.validate_effort("ultra")

    def test_generic_custom_payload_sends_selected_effort(self) -> None:
        payload: dict = {}
        _apply_thinking_options(
            payload,
            "https://gateway.invalid/v1",
            "ordinary-model",
            "enabled",
            "xhigh",
            True,
            65536,
        )
        self.assertEqual(payload["reasoning_effort"], "xhigh")

    def test_unknown_internal_effort_falls_back_to_high(self) -> None:
        payload: dict = {}
        _apply_thinking_options(
            payload,
            "https://gateway.invalid/v1",
            "ordinary-model",
            "enabled",
            "not-a-level",
            True,
            65536,
        )
        self.assertEqual(payload["reasoning_effort"], "high")

    def test_nvidia_deepseek_sends_low_in_chat_template_kwargs(self) -> None:
        payload: dict = {}
        _apply_thinking_options(
            payload,
            "https://integrate.api.nvidia.com/v1",
            "deepseek-ai/deepseek-v4-flash",
            "enabled",
            "low",
            True,
            65536,
        )
        self.assertEqual(payload["chat_template_kwargs"]["reasoning_effort"], "low")

    def test_chat_rejects_unknown_effort_and_accepts_medium(self) -> None:
        provider_id = self.add_provider()
        rejected = self.client.post(
            "/api/chat",
            json={
                "content": "offline",
                "provider_id": provider_id,
                "model": "ordinary-model",
                "effort": "ultra",
                "timezone": "UTC",
            },
        )
        self.assertEqual(rejected.status_code, 400, rejected.text)

        calls: list[dict] = []

        async def fake_custom_stream_response(**kwargs):
            calls.append(kwargs)
            return {"answer": "ok", "reasoning": "", "searches": [], "sources": [], "usage": {}}

        with patch.object(main, "custom_stream_response", new=fake_custom_stream_response), patch.object(
            main.httpx,
            "AsyncClient",
            side_effect=AssertionError("external HTTP is forbidden in effort tests"),
        ):
            accepted = self.client.post(
                "/api/chat",
                json={
                    "content": "offline",
                    "provider_id": provider_id,
                    "model": "ordinary-model",
                    "effort": "medium",
                    "timezone": "UTC",
                },
            )
            self.assertEqual(accepted.status_code, 200, accepted.text)
            job_id = accepted.json()["job_id"]
            job = main.db.one("SELECT effort FROM jobs WHERE id=?", (job_id,))
            self.assertEqual(job["effort"], "medium")


if __name__ == "__main__":
    unittest.main()

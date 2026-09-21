"""Regression tests for account-scoped conversations and shared management resources."""

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


class SharedResourceAccessTests(unittest.TestCase):
    admin_password = "shared-resource-admin-password"
    user_password = "shared-resource-user-password"

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
            {"ADMIN_USERNAME": "shared-resource-admin", "ADMIN_PASSWORD": self.admin_password},
            clear=False,
        )
        self.environment.start()

        main.settings = Settings(data_dir=self.data_dir)
        main.db = Database(main.settings.db_path)
        main.secret = b""
        main.tasks = {}
        main.attachment_cleanup_task = None
        attachments.ATTACHMENTS_DIR = self.data_dir / "attachments"
        self.admin = TestClient(main.app)
        self.admin.__enter__()
        login = self.admin.post(
            "/api/login",
            json={"username": "shared-resource-admin", "password": self.admin_password},
        )
        self.assertEqual(login.status_code, 200, login.text)
        self.admin_id = int(login.json()["id"])
        created = self.admin.post(
            "/api/users",
            json={"username": "shared-resource-user", "password": self.user_password, "is_admin": False},
        )
        self.assertEqual(created.status_code, 200, created.text)
        self.user_id = int(created.json()["id"])

        provider = self.admin.post(
            "/api/providers",
            json={
                "name": "shared-custom",
                "api_key": "shared-resource-api-key",
                "provider_type": "custom",
                "base_url": "https://provider.invalid/v1",
                "model": "shared-model",
                "selected_models": ["shared-model"],
            },
        )
        self.assertEqual(provider.status_code, 200, provider.text)
        self.provider_id = int(provider.json()["id"])
        self.user = TestClient(main.app)
        self.user.__enter__()
        login = self.user.post(
            "/api/login",
            json={"username": "shared-resource-user", "password": self.user_password},
        )
        self.assertEqual(login.status_code, 200, login.text)

    def tearDown(self) -> None:
        try:
            self.user.__exit__(None, None, None)
            self.admin.__exit__(None, None, None)
        finally:
            main.db = self.original_db
            main.settings = self.original_settings
            main.secret = self.original_secret
            main.tasks = self.original_tasks
            main.attachment_cleanup_task = self.original_cleanup_task
            attachments.ATTACHMENTS_DIR = self.original_attachments_dir
            self.environment.stop()
            self.temp_dir.cleanup()

    @staticmethod
    def settings_body(model: str = "shared-model") -> dict:
        return {
            "model": model,
            "thinking": "enabled",
            "reasoning_effort": "high",
            "reasoning_effort_enabled": True,
            "lowest_price_aggregators": [],
            "max_completion_tokens": 4096,
            "temperature_enabled": True,
            "temperature": 1,
            "top_p_enabled": True,
            "top_p": 0.95,
            "web_tool_backend": "parallel",
            "request_overrides": {},
        }

    def test_api_models_and_skills_are_shared_but_mutations_are_admin_only(self) -> None:
        admin_providers = self.admin.get("/api/providers")
        user_providers = self.user.get("/api/providers")
        self.assertEqual(admin_providers.status_code, 200, admin_providers.text)
        self.assertEqual(user_providers.status_code, 200, user_providers.text)
        self.assertEqual(admin_providers.json(), user_providers.json())
        self.assertNotIn("user_id", user_providers.json()[0])

        self.assertEqual(self.user.get(f"/api/providers/{self.provider_id}/key").status_code, 403)
        self.assertEqual(
            self.user.post(
                "/api/providers",
                json={
                    "name": "not-allowed",
                    "api_key": "not-allowed-key",
                    "provider_type": "custom",
                    "base_url": "https://provider.invalid/v1",
                    "model": "other-model",
                    "selected_models": ["other-model"],
                },
            ).status_code,
            403,
        )
        self.assertEqual(self.user.post("/api/providers/test", json={"name": "x", "api_key": "valid-key-123"}).status_code, 403)
        self.assertEqual(
            self.user.put(
                f"/api/providers/{self.provider_id}",
                json={"name": "changed", "api_key": "", "model": "shared-model", "selected_models": ["shared-model"]},
            ).status_code,
            403,
        )
        self.assertEqual(
            self.user.put(
                f"/api/providers/{self.provider_id}/models",
                json={"model": "shared-model", "selected_models": ["shared-model"]},
            ).status_code,
            403,
        )
        self.assertEqual(
            self.user.put(f"/api/providers/{self.provider_id}/settings", json=self.settings_body()).status_code,
            403,
        )
        self.assertEqual(self.user.delete(f"/api/providers/{self.provider_id}").status_code, 403)

        admin_skills = self.admin.get("/api/skills")
        user_skills = self.user.get("/api/skills")
        self.assertEqual(admin_skills.status_code, 200, admin_skills.text)
        self.assertEqual(user_skills.status_code, 200, user_skills.text)
        self.assertEqual(admin_skills.json(), user_skills.json())
        self.assertEqual(
            self.user.post("/api/skills/install", json={"source": "/does/not/exist", "name": "x"}).status_code,
            403,
        )
        self.assertEqual(
            self.user.put("/api/skills/writing-plans/enabled", json={"enabled": False}).status_code,
            403,
        )
        self.assertEqual(self.user.delete("/api/skills/not-installed").status_code, 403)

    def test_conversations_remain_private_while_both_accounts_can_use_shared_provider(self) -> None:
        with patch.object(main, "launch", new=lambda _: None):
            admin_chat = self.admin.post(
                "/api/chat",
                json={"content": "admin private message", "provider_id": self.provider_id, "model": "shared-model"},
            )
            user_chat = self.user.post(
                "/api/chat",
                json={"content": "user private message", "provider_id": self.provider_id, "model": "shared-model"},
            )
        self.assertEqual(admin_chat.status_code, 200, admin_chat.text)
        self.assertEqual(user_chat.status_code, 200, user_chat.text)
        admin_conversation = admin_chat.json()["conversation_id"]
        user_conversation = user_chat.json()["conversation_id"]
        self.assertNotEqual(admin_conversation, user_conversation)
        self.assertEqual(self.admin.get(f"/api/conversations/{admin_conversation}").status_code, 200)
        self.assertEqual(self.user.get(f"/api/conversations/{user_conversation}").status_code, 200)
        self.assertEqual(self.admin.get(f"/api/conversations/{user_conversation}").status_code, 404)
        self.assertEqual(self.user.get(f"/api/conversations/{admin_conversation}").status_code, 404)


if __name__ == "__main__":
    unittest.main()

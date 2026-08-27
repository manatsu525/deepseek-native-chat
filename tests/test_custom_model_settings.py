"""Offline regression tests for independent Custom settings per model."""

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


class CustomModelSettingsTests(unittest.TestCase):
    admin_password = "custom-settings-test-password"

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
            {"ADMIN_USERNAME": "custom-settings-admin", "ADMIN_PASSWORD": self.admin_password},
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
            json={"username": "custom-settings-admin", "password": self.admin_password},
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

    def add_legacy_provider(self) -> int:
        settings = {
            "models": ["model-a", "model-b"],
            "thinking": "disabled",
            "thinking_budget_tokens": 4096,
            "reasoning_effort_enabled": False,
            "dsml_fallback_enabled": True,
            "max_completion_tokens": 4096,
            "temperature": 0.4,
            "top_p": 0.8,
            "web_tool_backend": "you",
        }
        return main.db.run(
            """INSERT INTO providers(
                   user_id,name,api_key,base_url,model,provider_type,settings_json,created_at
               ) VALUES(?,?,?,?,?,?,?,?)""",
            (
                self.user_id,
                "multi-model-provider",
                "custom-settings-token",
                "https://provider.invalid/v1",
                "model-a",
                "custom",
                json.dumps(settings),
                100,
            ),
        )

    def settings_body(self, model: str, *, temperature: float, backend: str, effort: str = "high", aggregators: tuple[str, ...] = ()) -> dict:
        return {
            "model": model,
            "thinking": "enabled",
            "reasoning_effort": effort,
            "reasoning_effort_enabled": True,
            "lowest_price_aggregators": list(aggregators),
            "dsml_fallback_enabled": False,
            "max_completion_tokens": 8192,
            "temperature": temperature,
            "top_p": 0.9,
            "web_tool_backend": backend,
        }

    def wait_for_job(self, job_id: str) -> dict:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            job = main.db.one("SELECT * FROM jobs WHERE id=?", (job_id,))
            if job and job["status"] not in {"queued", "running"}:
                return job
            time.sleep(0.01)
        self.fail(f"mock job {job_id} did not finish")

    def test_legacy_api_settings_migrate_then_models_change_independently(self) -> None:
        provider_id = self.add_legacy_provider()

        main.migrate_custom_provider_settings()
        stored = json.loads(main.db.one("SELECT settings_json FROM providers WHERE id=?", (provider_id,))["settings_json"])
        self.assertEqual(set(stored), {"models", "model_settings"})
        self.assertEqual(stored["model_settings"]["model-a"]["temperature"], 0.4)
        self.assertEqual(stored["model_settings"]["model-b"]["temperature"], 0.4)
        self.assertEqual(stored["model_settings"]["model-a"]["reasoning_effort"], "high")
        self.assertEqual(stored["model_settings"]["model-b"]["reasoning_effort"], "high")
        self.assertNotIn("thinking_budget_tokens", stored["model_settings"]["model-a"])
        self.assertEqual(stored["model_settings"]["model-a"]["lowest_price_aggregators"], [])
        self.assertEqual(stored["model_settings"]["model-b"]["lowest_price_aggregators"], [])

        response = self.client.put(
            f"/api/providers/{provider_id}/settings",
            json=self.settings_body("model-a", temperature=1.1, backend="parallel"),
        )
        self.assertEqual(response.status_code, 200, response.text)
        provider = response.json()
        self.assertEqual(provider["model_settings"]["model-a"]["temperature"], 1.1)
        self.assertEqual(provider["model_settings"]["model-a"]["web_tool_backend"], "parallel")
        self.assertEqual(provider["model_settings"]["model-b"]["temperature"], 0.4)
        self.assertEqual(provider["model_settings"]["model-b"]["web_tool_backend"], "you")

        response = self.client.put(
            f"/api/providers/{provider_id}/models",
            json={"model": "model-a", "selected_models": ["model-a", "model-b", "model-c"]},
        )
        self.assertEqual(response.status_code, 200, response.text)
        provider = response.json()
        self.assertEqual(provider["model_settings"]["model-a"]["temperature"], 1.1)
        self.assertEqual(provider["model_settings"]["model-b"]["temperature"], 0.4)
        self.assertEqual(provider["model_settings"]["model-c"]["temperature"], 1.0)
        self.assertFalse(provider["model_settings"]["model-c"]["dsml_fallback_enabled"])

    def test_job_receives_only_the_selected_models_settings(self) -> None:
        provider_id = self.add_legacy_provider()
        main.migrate_custom_provider_settings()
        response = self.client.put(
            f"/api/providers/{provider_id}/settings",
            json=self.settings_body("model-a", temperature=1.1, backend="parallel", effort="xhigh", aggregators=("openrouter",)),
        )
        self.assertEqual(response.status_code, 200, response.text)
        response = self.client.put(
            f"/api/providers/{provider_id}/settings",
            json=self.settings_body("model-b", temperature=0.7, backend="parallel", effort="low", aggregators=("vercel",)),
        )
        self.assertEqual(response.status_code, 200, response.text)
        provider = response.json()
        self.assertEqual(provider["model_settings"]["model-a"]["reasoning_effort"], "xhigh")
        self.assertEqual(provider["model_settings"]["model-b"]["reasoning_effort"], "low")
        self.assertEqual(provider["model_settings"]["model-a"]["lowest_price_aggregators"], ["openrouter"])
        self.assertEqual(provider["model_settings"]["model-b"]["lowest_price_aggregators"], ["vercel"])
        calls: list[dict] = []

        async def fake_custom_stream_response(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            return {"answer": "ok", "reasoning": "", "searches": [], "sources": [], "usage": {}}

        with patch.object(main, "custom_stream_response", new=fake_custom_stream_response), patch.object(
            main.httpx,
            "AsyncClient",
            side_effect=AssertionError("external HTTP is forbidden in Custom settings tests"),
        ):
            response = self.client.post(
                "/api/chat",
                json={
                    "content": "offline test",
                    "provider_id": provider_id,
                    "model": "model-b",
                    # The legacy request field is intentionally different;
                    # Custom execution must use model-b's saved setting.
                    "effort": "xhigh",
                    "timezone": "UTC",
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.wait_for_job(response.json()["job_id"])

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["model"], "model-b")
        self.assertEqual(calls[0]["settings"]["temperature"], 0.7)
        self.assertEqual(calls[0]["settings"]["web_tool_backend"], "parallel")
        self.assertEqual(calls[0]["settings"]["reasoning_effort"], "low")
        self.assertEqual(calls[0]["settings"]["lowest_price_aggregators"], ["vercel"])
        self.assertEqual(calls[0]["effort"], "low")
        self.assertNotEqual(calls[0]["settings"]["temperature"], 1.1)

    def test_settings_reject_a_model_not_enabled_for_the_api(self) -> None:
        provider_id = self.add_legacy_provider()
        response = self.client.put(
            f"/api/providers/{provider_id}/settings",
            json=self.settings_body("unknown-model", temperature=1.0, backend="parallel"),
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("未在此 custom API", response.json()["detail"])

    def test_connection_key_protocol_address_and_models_can_be_edited(self) -> None:
        provider_id = self.add_legacy_provider()
        main.migrate_custom_provider_settings()

        revealed = self.client.get(f"/api/providers/{provider_id}/key")
        self.assertEqual(revealed.status_code, 200, revealed.text)
        self.assertEqual(revealed.json()["api_key"], "custom-settings-token")
        self.assertEqual(revealed.headers["cache-control"], "no-store")

        response = self.client.put(
            f"/api/providers/{provider_id}",
            json={
                "name": "edited-provider",
                "api_key": "replacement-api-key",
                "provider_type": "custom_response",
                "base_url": "https://responses.invalid/v1/",
                "model": "model-b",
                "selected_models": ["model-b", "model-c"],
                "manual_models": ["model-c"],
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        public = response.json()
        self.assertEqual(public["name"], "edited-provider")
        self.assertEqual(public["provider_type"], "custom_response")
        self.assertEqual(public["base_url"], "https://responses.invalid/v1")
        self.assertEqual(public["models"], ["model-b", "model-c"])
        stored = main.db.one("SELECT * FROM providers WHERE id=?", (provider_id,))
        self.assertEqual(stored["api_key"], "replacement-api-key")
        self.assertEqual(public["model_settings"]["model-b"]["temperature"], 0.4)
        self.assertEqual(public["model_settings"]["model-c"]["temperature"], 1.0)

        response = self.client.put(
            f"/api/providers/{provider_id}",
            json={
                "name": "edited-again",
                "api_key": "",
                "provider_type": "custom_messages",
                "base_url": "https://messages.invalid/v1",
                "model": "model-b",
                "selected_models": ["model-b"],
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        stored = main.db.one("SELECT * FROM providers WHERE id=?", (provider_id,))
        self.assertEqual(stored["api_key"], "replacement-api-key")
        self.assertEqual(stored["provider_type"], "custom_messages")

    def test_manual_model_can_be_saved_without_a_successful_probe(self) -> None:
        response = self.client.post(
            "/api/providers",
            json={
                "name": "unverified-manual-model",
                "api_key": "manual-model-api-key",
                "provider_type": "custom_response",
                "base_url": "https://offline.invalid/v1",
                "model": "tencent/hy3",
                "selected_models": ["tencent/hy3"],
                "manual_models": ["tencent/hy3"],
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["models"], ["tencent/hy3"])


if __name__ == "__main__":
    unittest.main()

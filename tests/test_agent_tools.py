from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.agent import AgentRuntime
from app.db import Database
from app.skills import DEFAULT_SKILLS, SkillRegistry


class AgentToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = Database(self.root / "chat.db")
        self.db.init()
        self.user_id = self.db.run(
            "INSERT INTO users(username,password_hash,created_at) VALUES(?,?,?)",
            ("agent-user", "hash", 1),
        )
        self.conversation_id = "agent-test"
        self.db.run(
            "INSERT INTO conversations(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)",
            (self.conversation_id, self.user_id, "Agent", 1, 1),
        )
        self.runtime = AgentRuntime(self.db, self.user_id, self.conversation_id)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_host_file_and_command_tools_use_real_path(self) -> None:
        with patch("app.agent.AGENT_PROJECT_ROOT", self.root):
            written = self.runtime.execute("host_write_file", {"path": "src/app.py", "content": "print('ok')\n"})
            self.assertIn('"ok": true', written)
            read = self.runtime.execute("host_read_file", {"path": "src/app.py"})
            self.assertIn("1|print('ok')", read)
            command = self.runtime.execute("host_run_command", {"command": "printf hello"})
            self.assertIn('"stdout": "hello"', command)

    def test_conversation_management_is_scoped_to_current_user(self) -> None:
        created = self.runtime.execute("conversation_create", {"title": "新线程"})
        self.assertIn('"ok": true', created)
        listed = self.runtime.execute("conversation_list", {})
        self.assertIn("新线程", listed)
        self.assertIn("Agent", listed)

    def test_default_skills_are_present_and_exposed_as_tools(self) -> None:
        registry = SkillRegistry()
        self.assertTrue(set(DEFAULT_SKILLS).issubset({item.skill_id for item in registry.all()}))
        names = {item["function"]["name"] for item in self.runtime.tool_definitions}
        self.assertIn("skill_install", names)
        self.assertIn("conversation_create", names)
        self.assertIn("frontend_validate_page", names)


if __name__ == "__main__":
    unittest.main()

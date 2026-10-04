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
            "INSERT INTO users(username,password_hash,is_admin,created_at) VALUES(?,?,?,?)",
            ("agent-user", "hash", 1, 1),
        )
        self.conversation_id = "agent-test"
        self.db.run(
            "INSERT INTO conversations(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)",
            (self.conversation_id, self.user_id, "Agent", 1, 1),
        )
        self.runtime = AgentRuntime(self.db, self.user_id, self.conversation_id, is_admin=True)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_host_file_and_command_tools_use_real_path(self) -> None:
        with patch("app.agent.AGENT_PROJECT_ROOT", self.root):
            written = self.runtime.execute("write_file", {"path": "src/app.py", "content": "print('ok')\n"})
            self.assertIn('"ok": true', written)
            read = self.runtime.execute("read_file", {"path": "src/app.py"})
            self.assertIn("1|print('ok')", read)
            command = self.runtime.execute("run_command", {"command": "printf hello"})
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
        self.assertIn("check_web_syntax", names)
        self.assertNotIn("host_read_file", names)

    def test_non_admin_can_read_but_cannot_mutate_shared_skills(self) -> None:
        runtime = AgentRuntime(self.db, self.user_id, self.conversation_id, is_admin=False)
        names = {item["function"]["name"] for item in runtime.tool_definitions}
        self.assertIn("skill_list", names)
        self.assertIn("skill_read", names)
        self.assertNotIn("skill_install", names)
        self.assertNotIn("skill_enable", names)
        self.assertNotIn("skill_remove", names)
        denied = runtime.execute("skill_enable", {"skill_id": "writing-plans", "enabled": False})
        self.assertIn("仅管理员", denied)

    def test_large_listing_collapses_to_one_level_with_counts(self) -> None:
        import json

        with patch("app.agent.AGENT_PROJECT_ROOT", self.root):
            for index in range(40):
                (self.root / "big" / f"sub{index}").mkdir(parents=True)
                for name in range(10):
                    (self.root / "big" / f"sub{index}" / f"f{name}.txt").write_text("x")
            (self.root / "big" / ".git").mkdir()
            (self.root / "big" / ".git" / "HEAD").write_text("ref")
            (self.root / "big" / "README.md").write_text("hello")
            listed = json.loads(self.runtime.execute("list_files", {"path": "big"}))
            self.assertTrue(listed["truncated"])
            self.assertIn("只列出第一层", listed["note"])
            self.assertLessEqual(len(listed["entries"]), 42)
            by_path = {Path(item["path"]).name: item for item in listed["entries"]}
            self.assertEqual(by_path["sub3"]["files_below"], 10)
            self.assertEqual(by_path["README.md"]["size"], 5)
            self.assertNotIn(".git", by_path)
            small = json.loads(self.runtime.execute("list_files", {"path": "big/sub3"}))
            self.assertFalse(small["truncated"])
            self.assertEqual(len(small["entries"]), 10)

    def test_partial_file_views_in_commands_show_the_whole_file(self) -> None:
        import json

        with patch("app.agent.AGENT_PROJECT_ROOT", self.root):
            (self.root / "conf.json").write_text("\n".join(f"line {n}" for n in range(1, 41)) + "\n")
            result = json.loads(self.runtime.execute("run_command", {"command": "sed -n '10,12p' conf.json; echo ===; head -3 conf.json | wc -l"}))
            self.assertTrue(result["ok"])
            self.assertIn("     1\tline 1", result["stdout"])
            self.assertIn("    40\tline 40", result["stdout"])
            self.assertEqual(result["stdout"].count("line 40"), 1)  # the piped head is untouched
            self.assertEqual(len(result["notes"]), 1)
            self.assertIn("sed -n '10,12p' conf.json", result["notes"][0])
            self.assertIn("40 行", result["notes"][0])
            # Big or missing files are left to the command as written.
            (self.root / "huge.txt").write_text("y" * 40_000)
            result = json.loads(self.runtime.execute("run_command", {"command": "head -c 5 huge.txt; tail -n 1 missing.txt"}))
            self.assertNotIn("notes", result)
            self.assertEqual(result["stdout"], "yyyyy")


if __name__ == "__main__":
    unittest.main()

import unittest

from app.prompts import build_system_prompt, TOOL_CONTINUATION_RULES, files_group_rules


class SystemPromptTests(unittest.TestCase):
    def test_tool_continuation_is_shared_when_file_tools_deferred(self) -> None:
        for agent_mode in (False, True):
            for backend in ("parallel", "keenable", "legacy"):
                for file_tools_loaded in (False, True):
                    with self.subTest(agent=agent_mode, backend=backend, loaded=file_tools_loaded):
                        kwargs = dict(agent_mode=agent_mode, web_enabled=True, web_backend=backend,
                                      workspace_access=None if agent_mode else "full", user_timezone="UTC",
                                      file_tools_loaded=file_tools_loaded)
                        prompt = build_system_prompt(**kwargs)
                        self.assertEqual(prompt.count(TOOL_CONTINUATION_RULES), 1)
                        self.assertIn("same active user request", prompt)
                        self.assertIn("concrete new finding", prompt)
                        self.assertIn("specific information still missing", prompt)
                        self.assertIn("when the requested result is ready", prompt)
                        self.assertEqual(prompt, build_system_prompt(**kwargs))
                        if not agent_mode and not file_tools_loaded:
                            self.assertNotIn("Working with files and commands:", prompt)
                            self.assertIn("not loaded yet", prompt)
        self.assertNotIn("Work visibly", files_group_rules("full"))

    def test_research_rules_keep_user_terms_exact(self) -> None:
        prompt = build_system_prompt(agent_mode=False, web_enabled=True, web_backend="parallel",
                                     workspace_access="full", user_timezone="UTC")
        self.assertIn("your first search must contain them verbatim", prompt)
        self.assertIn("could not be verified", prompt)
        self.assertIn("Today is", prompt)
        self.assertIn("edit_file", prompt)
        self.assertIn("update_plan", prompt)
        self.assertIn("discarded", prompt)
        self.assertNotIn("/home/share", prompt)
        # Rules that came out of the VCMI benchmark: no paging through files
        # with shell tools, visible progress text, and no proof-hunting in
        # engine source when the documentation already answers the question.
        self.assertIn("never with cat, sed -n, head or tail", prompt)
        self.assertIn("Work visibly", prompt)
        self.assertIn("do not read an engine's or framework's source code", prompt)

    def test_agent_prompt_describes_host_and_skills(self) -> None:
        prompt = build_system_prompt(agent_mode=True, web_enabled=True, web_backend="parallel",
                                     workspace_access=None, user_timezone="Asia/Shanghai",
                                     skills_prompt="INSTALLED AGENT SKILLS:\n- a: b")
        self.assertIn("/home/share", prompt)
        self.assertIn("git clone --depth 1", prompt)
        self.assertIn("INSTALLED AGENT SKILLS", prompt)
        self.assertIn("Asia/Shanghai", prompt)
        self.assertNotIn("discarded", prompt)

    def test_prompt_is_stable_and_compact(self) -> None:
        kwargs = dict(agent_mode=False, web_enabled=False, web_backend="parallel", workspace_access="read_only", user_timezone="UTC")
        self.assertEqual(build_system_prompt(**kwargs), build_system_prompt(**kwargs))
        self.assertIn("must not create", build_system_prompt(**kwargs))
        self.assertIn("No web search", build_system_prompt(**kwargs))
        self.assertLess(len(build_system_prompt(agent_mode=True, web_enabled=True, web_backend="parallel",
                                                workspace_access=None, user_timezone="UTC")), 6_000)


if __name__ == "__main__":
    unittest.main()

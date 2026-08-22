import unittest

from app.mimo_local import ENTITY_FIDELITY_PROMPT, _dated_system_prompt


class ResearchPromptTests(unittest.TestCase):
    def test_exact_unknown_entities_must_not_be_silently_rewritten(self) -> None:
        prompt = _dated_system_prompt("base", "UTC")

        self.assertIn(ENTITY_FIDELITY_PROMPT, prompt)
        self.assertIn("first search must include the user's exact identifying terms", prompt)
        self.assertIn("Absence from one search", prompt)
        self.assertIn("explicitly say that it could not be verified", prompt)


if __name__ == "__main__":
    unittest.main()

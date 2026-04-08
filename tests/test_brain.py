import unittest

from core.brain import BrainOutputError, sanitize_llm_output


class BrainSanitizerTests(unittest.TestCase):
    def test_extracts_json_from_noisy_model_output(self) -> None:
        response = sanitize_llm_output('Sure. {"action":"scroll","value":"down"}')
        self.assertEqual("scroll", response.action)
        self.assertEqual("down", response.value)

    def test_rejects_missing_action(self) -> None:
        with self.assertRaises(BrainOutputError):
            sanitize_llm_output('{"value":"down"}')


if __name__ == "__main__":
    unittest.main()

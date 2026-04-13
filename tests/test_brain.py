import unittest
from pathlib import Path

from core.brain import Brain, BrainOutputError, sanitize_llm_output


class BrainSanitizerTests(unittest.TestCase):
    def test_extracts_json_from_noisy_model_output(self) -> None:
        response = sanitize_llm_output('Sure. {"action":"scroll","value":"down"}')
        self.assertEqual("scroll", response.action)
        self.assertEqual("down", response.value)

    def test_rejects_missing_action(self) -> None:
        with self.assertRaises(BrainOutputError):
            sanitize_llm_output('{"value":"down"}')


class BrainPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.brain = Brain(prompt_template_path="config/prompt_template.txt")

    def test_fallback_uses_ocr_history_for_address_bar(self) -> None:
        response = self.brain.infer_action(
            audio_samples=None,
            sample_rate=16000,
            screenshot_path=None,
            ocr_history="Browser address bar",
        )
        self.assertEqual("hotkey", response.action)
        self.assertEqual(["ctrl", "l"], response.value)

    def test_reports_missing_openvino_model_as_unloaded(self) -> None:
        brain = Brain(
            prompt_template_path="config/prompt_template.txt",
            model_path=Path("models") / "missing-gemma",
        )
        diagnostics = brain.diagnostics()
        self.assertFalse(diagnostics["model_path_exists"])

    def test_uses_custom_infer_when_present(self) -> None:
        brain = Brain(
            prompt_template_path="config/prompt_template.txt",
            infer_backend=lambda _prompt, _screenshot, _audio, _rate: '{"action":"click","value":null}',
            model_path=Path("models") / "missing-gemma",
        )
        response = brain.infer_action(audio_samples=None, sample_rate=16000, screenshot_path=None, ocr_history="")
        self.assertEqual("click", response.action)
        self.assertIsNone(response.value)


if __name__ == "__main__":
    unittest.main()

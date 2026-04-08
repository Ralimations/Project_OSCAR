import unittest

from actions.controller import ActionRequest
from core.router import DirectCommandRouter
from main import OSCAR, RuntimeConfig


class FakeWindowManager:
    def __init__(self, title: str | None) -> None:
        self._title = title

    def get_active_window_title(self) -> str | None:
        return self._title


class MacroRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RuntimeConfig(
            settings={
                "runtime": {
                    "log_level": "INFO",
                    "dormant_fps": 15,
                    "gesture_cooldown_ms": 200,
                },
                "gestures": {
                    "pinch_threshold": 0.045,
                    "scroll_deadzone": 0.03,
                    "scroll_scale": 1200,
                    "nod_threshold": 12,
                    "shake_threshold": 18,
                },
                "screenpipe": {"database_path": "database/screenpipe.db"},
                "voice": {"model_path": "", "piper_executable": "piper"},
                "models": {"wake_word_model_path": "", "whisper_model_path": ""},
            },
            gestures={},
            app_macros={
                "BandLab": {
                    "start_recording": {
                        "action": "press_key",
                        "value": "r",
                    }
                }
            },
        )
        self.oscar = OSCAR(self.config)
        self.oscar._window_manager = FakeWindowManager("BandLab Studio")

    def tearDown(self) -> None:
        self.oscar.stop()

    def test_resolves_macro_for_active_window(self) -> None:
        request = self.oscar.resolve_macro_request("start recording")
        self.assertIsInstance(request, ActionRequest)
        self.assertEqual("press_key", request.action)
        self.assertEqual("r", request.value)

    def test_returns_none_for_unknown_command(self) -> None:
        request = self.oscar.resolve_macro_request("stop recording")
        self.assertIsNone(request)


class DirectRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = DirectCommandRouter()

    def test_routes_simple_open_app_request(self) -> None:
        result = self.router.route("open notepad")
        self.assertIsNotNone(result)
        self.assertEqual("open_app", result.request.action)
        self.assertEqual("notepad", result.request.value)

    def test_routes_open_app_with_task_as_sequence(self) -> None:
        result = self.router.route("open brave and watch the trailer for dune 3")
        self.assertIsNotNone(result)
        self.assertEqual("sequence", result.request.action)
        steps = result.request.value
        self.assertEqual("open_app", steps[0]["action"])
        self.assertEqual("brave", steps[0]["value"])
        self.assertEqual("write_text", steps[3]["action"])
        self.assertIn("youtube", steps[3]["value"])


if __name__ == "__main__":
    unittest.main()

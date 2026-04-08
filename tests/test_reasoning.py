import os
import sqlite3
import tempfile
import unittest

from actions.controller import ActionRequest
from core.brain import Brain, sanitize_llm_output
from core.memory import Memory
from main import OSCAR, RuntimeConfig


class FakeWindowManager:
    def __init__(self, title: str | None) -> None:
        self._title = title

    def get_active_window_title(self) -> str | None:
        return self._title


class FakeController:
    def __init__(self) -> None:
        self.dispatched: list[ActionRequest] = []

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def dispatch(self, request: ActionRequest) -> None:
        self.dispatched.append(request)

    def register_handler(self, action: str, handler) -> None:
        return


class FakeEyes:
    def __init__(self) -> None:
        self.paused = False

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False


class OrchestratorReasoningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RuntimeConfig(
            settings={
                "runtime": {"log_level": "INFO", "dormant_fps": 15, "gesture_cooldown_ms": 200},
                "gestures": {"pinch_threshold": 0.045, "scroll_deadzone": 0.03, "scroll_scale": 1200, "nod_threshold": 12, "shake_threshold": 18},
                "screenpipe": {"database_path": "database/screenpipe.db"},
                "voice": {"model_path": "", "piper_executable": "piper"},
                "models": {"wake_word_model_path": "", "whisper_model_path": ""},
            },
            gestures={},
            app_macros={"BandLab": {"start_recording": {"action": "press_key", "value": "r"}}},
        )
        self.oscar = OSCAR(self.config)
        self.oscar._controller = FakeController()
        self.oscar._window_manager = FakeWindowManager("BandLab Studio")
        self.oscar._eyes = FakeEyes()
        self.oscar._brain = Brain(prompt_template_path="config/prompt_template.txt")

    def tearDown(self) -> None:
        self.oscar.stop()

    def test_memory_route_dispatches_speak_from_context(self) -> None:
        self.oscar._memory = type("MemoryStub", (), {"query_recent_text": lambda _self, _prompt: "john@example.com"})()
        self.oscar.handle_wake_word("What was that email address?")
        self.assertEqual("speak", self.oscar._controller.dispatched[-1].action)
        self.assertEqual("john@example.com", self.oscar._controller.dispatched[-1].value)
        self.assertFalse(self.oscar._eyes.paused)

    def test_macro_route_takes_priority_before_brain(self) -> None:
        self.oscar.handle_wake_word("start recording")
        self.assertEqual("press_key", self.oscar._controller.dispatched[-1].action)
        self.assertEqual("r", self.oscar._controller.dispatched[-1].value)


class BrainTests(unittest.TestCase):
    def test_route_decision_detects_visual_query(self) -> None:
        brain = Brain(prompt_template_path="config/prompt_template.txt")
        decision = brain.decide_route("Where is the submit button?")
        self.assertEqual("vision", decision.route)

    def test_sanitize_allows_sequence_actions(self) -> None:
        response = sanitize_llm_output('{"action":"sequence","value":[{"action":"press_key","value":"r"}]}')
        self.assertEqual("sequence", response.action)


class MemoryTests(unittest.TestCase):
    def test_memory_reads_matching_text_from_sqlite(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as handle:
            db_path = handle.name
        connection = sqlite3.connect(db_path)
        try:
            connection.execute("CREATE TABLE ocr_log (text TEXT, window_title TEXT)")
            connection.execute("INSERT INTO ocr_log (text, window_title) VALUES (?, ?)", ("john@example.com", "Mail"))
            connection.commit()
            connection.close()

            memory = Memory(database_path=db_path)
            result = memory.query_recent_text("john")
            self.assertIn("john@example.com", result)
        finally:
            if connection:
                try:
                    connection.close()
                except sqlite3.Error:
                    pass
            os.remove(db_path)


if __name__ == "__main__":
    unittest.main()

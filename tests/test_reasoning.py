import sqlite3
import tempfile
import unittest

from actions.controller import ActionRequest
from core.brain import Brain, sanitize_llm_output
from core.ears import AudioCaptureEvent
from core.memory import Memory
from main import OSCAR, OrchestratorState, RuntimeConfig


class FakeController:
    def __init__(self) -> None:
        self.dispatched: list[ActionRequest] = []

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def dispatch(self, request: ActionRequest) -> None:
        self.dispatched.append(request)


class FakeEyes:
    def __init__(self) -> None:
        self.paused = False

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def diagnostics(self) -> dict[str, bool]:
        return {
            "mediapipe_tasks_available": True,
            "hand_landmarker_model_present": True,
            "face_landmarker_model_present": True,
        }

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False


class FakeWindowManager:
    def __init__(self, title: str | None = None) -> None:
        self.title = title

    def get_active_window_title(self) -> str | None:
        return self.title


class OrchestratorReasoningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RuntimeConfig(
            settings={
                "runtime": {"log_level": "INFO", "dormant_fps": 15, "gesture_cooldown_ms": 200, "sandbox_actions_only": True},
                "hardware": {"cpu_threads": 4, "openvino_device": "AUTO"},
                "gestures": {"pinch_threshold": 0.045, "scroll_deadzone": 0.03, "scroll_scale": 1200, "nod_threshold": 12, "shake_threshold": 18},
                "audio": {"enabled": False, "sample_rate": 16000, "chunk_ms": 80, "capture_seconds": 3.0, "wake_threshold": 0.5},
                "screenpipe": {"database_path": "database/screenpipe.db", "history_limit": 8},
                "voice": {},
                "models": {
                    "inference_backend": "fallback",
                    "wake_word_model_path": "",
                    "gemma_model_path": "models/gemma-4-e4b-openvino",
                    "hand_landmarker_task_path": "",
                    "face_landmarker_task_path": "",
                },
            },
            gestures={},
            app_macros={},
        )
        self.oscar = OSCAR(self.config)
        self.oscar._controller = FakeController()
        self.oscar._eyes = FakeEyes()
        self.oscar._window_manager = FakeWindowManager()

    def tearDown(self) -> None:
        self.oscar.stop()

    def test_audio_trigger_dispatches_gemma_action(self) -> None:
        self.oscar._memory = type("MemoryStub", (), {"recent_history": lambda _self, limit=8: "address bar visible"})()
        self.oscar._vision = type("VisionStub", (), {"capture_snapshot": lambda _self: ""})()
        self.oscar._brain = Brain(
            prompt_template_path="config/prompt_template.txt",
            infer_backend=lambda _prompt, _screenshot, _audio, _rate: '{"action":"hotkey","value":["ctrl","l"]}',
        )

        self.oscar._process_audio_trigger(
            AudioCaptureEvent(audio_samples=[0.1], sample_rate=16000, source="audio_file")
        )

        self.assertEqual("hotkey", self.oscar._controller.dispatched[-1].action)
        self.assertEqual(["ctrl", "l"], self.oscar._controller.dispatched[-1].value)
        self.assertFalse(self.oscar._eyes.paused)
        self.assertEqual(OrchestratorState.DORMANT, self.oscar._state)

    def test_transcript_direct_route_bypasses_heavy_context(self) -> None:
        self.oscar._memory = type("MemoryStub", (), {"query_recent_text": lambda *_args, **_kwargs: self.fail("memory route should not run")})()
        self.oscar._vision = type("VisionStub", (), {"answer_visual_question": lambda *_args, **_kwargs: self.fail("vision route should not run")})()
        self.oscar._brain = Brain(
            prompt_template_path="config/prompt_template.txt",
            infer_backend=lambda *_args, **_kwargs: self.fail("brain route should not run"),
        )

        self.oscar._process_audio_trigger(
            AudioCaptureEvent(audio_samples=[], sample_rate=16000, source="transcript", transcript="open notepad")
        )

        self.assertEqual("open_app", self.oscar._controller.dispatched[-1].action)
        self.assertEqual("notepad", self.oscar._controller.dispatched[-1].value)

    def test_transcript_memory_route_speaks_matching_history(self) -> None:
        self.oscar._memory = type("MemoryStub", (), {"query_recent_text": lambda _self, prompt, limit=8: f"match for {prompt}"})()

        self.oscar._process_audio_trigger(
            AudioCaptureEvent(
                audio_samples=[],
                sample_rate=16000,
                source="transcript",
                transcript="what was that email address from earlier",
            )
        )

        self.assertEqual("speak", self.oscar._controller.dispatched[-1].action)
        self.assertIn("email address", self.oscar._controller.dispatched[-1].value)

    def test_transcript_macro_route_uses_active_window(self) -> None:
        self.oscar._window_manager = FakeWindowManager(title="BandLab Studio")
        self.oscar._config = RuntimeConfig(
            settings=self.config.settings,
            gestures=self.config.gestures,
            app_macros={
                "BandLab": {
                    "start recording": {
                        "action": "press_key",
                        "value": "r",
                    }
                }
            },
        )

        self.oscar._process_audio_trigger(
            AudioCaptureEvent(audio_samples=[], sample_rate=16000, source="transcript", transcript="start recording")
        )

        self.assertEqual("press_key", self.oscar._controller.dispatched[-1].action)
        self.assertEqual("r", self.oscar._controller.dispatched[-1].value)

    def test_transcript_planner_falls_back_to_brain(self) -> None:
        self.oscar._brain = Brain(
            prompt_template_path="config/prompt_template.txt",
            infer_backend=lambda _prompt, _screenshot, _audio, _rate: '{"action":"press_key","value":"enter"}',
        )

        self.oscar._process_audio_trigger(
            AudioCaptureEvent(audio_samples=[], sample_rate=16000, source="transcript", transcript="confirm that choice")
        )

        self.assertEqual("press_key", self.oscar._controller.dispatched[-1].action)
        self.assertEqual("enter", self.oscar._controller.dispatched[-1].value)

    def test_wake_phrase_transcript_dispatches_acknowledgement(self) -> None:
        self.oscar._process_audio_trigger(
            AudioCaptureEvent(audio_samples=[], sample_rate=16000, source="transcript", transcript="hey oscar")
        )

        self.assertEqual("speak", self.oscar._controller.dispatched[-1].action)
        self.assertEqual("Yes? How can I help you?", self.oscar._controller.dispatched[-1].value)

    def test_raw_wake_word_event_dispatches_acknowledgement(self) -> None:
        self.oscar._process_audio_trigger(
            AudioCaptureEvent(audio_samples=[0.1], sample_rate=16000, source="wake_word", transcript=None)
        )

        self.assertEqual("speak", self.oscar._controller.dispatched[-1].action)
        self.assertEqual("Yes? How can I help you?", self.oscar._controller.dispatched[-1].value)


class BrainTests(unittest.TestCase):
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
            memory = Memory(database_path=db_path)
            result = memory.query_recent_text("john")
            self.assertIn("john@example.com", result)
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()

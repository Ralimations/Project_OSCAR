import unittest

from core.router import DirectCommandRouter, classify_transcript_intent, TranscriptIntent


class RouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = DirectCommandRouter()

    def test_routes_zoom_in_to_hotkey(self) -> None:
        result = self.router.route("zoom in")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual("hotkey", result.request.action)
        self.assertEqual(["ctrl", "+"], result.request.value)

    def test_routes_add_new_tab_to_hotkey(self) -> None:
        result = self.router.route("add new tab")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual("hotkey", result.request.action)
        self.assertEqual(["ctrl", "t"], result.request.value)

    def test_routes_delete_tab_to_hotkey(self) -> None:
        result = self.router.route("delete tab")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual("hotkey", result.request.action)
        self.assertEqual(["ctrl", "w"], result.request.value)

    def test_routes_open_task_in_app_to_sequence(self) -> None:
        result = self.router.route("open bandlab in brave")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual("sequence", result.request.action)
        self.assertEqual("brave", result.request.value[0]["value"])
        self.assertEqual("bandlab", result.request.value[3]["value"])

    def test_routes_go_back_to_hotkey(self) -> None:
        result = self.router.route("go back")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(["alt", "left"], result.request.value)

    def test_classifies_history_requests_as_memory(self) -> None:
        self.assertEqual(
            TranscriptIntent.MEMORY,
            classify_transcript_intent("what was that email address from earlier"),
        )

    def test_classifies_visual_requests_as_vision(self) -> None:
        self.assertEqual(
            TranscriptIntent.VISION,
            classify_transcript_intent("where is the submit button"),
        )


if __name__ == "__main__":
    unittest.main()

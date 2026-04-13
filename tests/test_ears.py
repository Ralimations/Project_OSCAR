import queue
import unittest

from core.ears import AudioCaptureEvent, Ears


class EarsTests(unittest.TestCase):
    def test_injected_audio_reaches_callback(self) -> None:
        observed: queue.Queue[AudioCaptureEvent] = queue.Queue()
        ears = Ears(on_audio_captured=lambda event: observed.put(event), audio_settings={"enabled": False})
        ears.start()
        try:
            ears.inject_audio_capture([0.0, 0.1, -0.1])
            event = observed.get(timeout=1.0)
            self.assertEqual([0.0, 0.1, -0.1], event.audio_samples)
            self.assertEqual("wake_word", event.source)
        finally:
            ears.stop()

    def test_diagnostics_report_audio_flags(self) -> None:
        ears = Ears(on_audio_captured=None, audio_settings={"enabled": False})
        diagnostics = ears.diagnostics()
        self.assertFalse(diagnostics["audio_capture_enabled"])


if __name__ == "__main__":
    unittest.main()

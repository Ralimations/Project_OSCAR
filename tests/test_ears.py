import queue
import unittest

from core.ears import Ears, TranscriptEvent


class EarsTests(unittest.TestCase):
    def test_injected_transcript_reaches_callback(self) -> None:
        observed: queue.Queue[TranscriptEvent] = queue.Queue()
        ears = Ears(on_transcript=lambda event: observed.put(event))
        ears.start()

        try:
            ears.inject_transcript("start recording")
            event = observed.get(timeout=1.0)
            self.assertEqual("start recording", event.transcript)
            self.assertEqual("wake_word", event.source)
        finally:
            ears.stop()


if __name__ == "__main__":
    unittest.main()

import unittest

from core.eyes import Eyes, HeadPose, Point


SETTINGS = {
    "dormant_fps": 15,
    "gesture_cooldown_ms": 200,
    "pinch_threshold": 0.045,
    "scroll_deadzone": 0.03,
    "scroll_scale": 1200,
    "nod_threshold": 12,
    "shake_threshold": 18,
}


class EyesGestureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.eyes = Eyes(settings=SETTINGS, on_gesture=lambda _: None, frame_source=lambda: iter(()))

    def test_detects_click_from_pinch(self) -> None:
        event = self.eyes.interpret_observation(
            hand_landmarks={
                4: Point(x=0.10, y=0.10),
                8: Point(x=0.12, y=0.12),
            }
        )
        self.assertIsNotNone(event)
        self.assertEqual("click", event.action)

    def test_detects_scroll_when_two_fingers_move_vertically(self) -> None:
        event = self.eyes.interpret_observation(
            hand_landmarks={
                8: Point(x=0.45, y=0.20),
                12: Point(x=0.47, y=0.40),
            }
        )
        self.assertIsNotNone(event)
        self.assertEqual("scroll", event.action)
        self.assertLess(event.value, 0)

    def test_detects_head_nod_confirmation(self) -> None:
        event = self.eyes.interpret_observation(head_pose=HeadPose(pitch=15.0))
        self.assertIsNotNone(event)
        self.assertEqual("confirm", event.action)
        self.assertEqual("yes", event.value)


if __name__ == "__main__":
    unittest.main()

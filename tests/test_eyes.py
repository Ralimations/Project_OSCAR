import unittest

from core.eyes import Eyes, HeadPose, Point


SETTINGS = {
    "dormant_fps": 15,
    "gesture_cooldown_ms": 200,
    "max_hands": 2,
    "pinch_threshold": 0.045,
    "scroll_deadzone": 0.03,
    "scroll_scale": 1200,
    "zoom_distance_threshold": 0.08,
    "zoom_hotkey_in": ["ctrl", "+"],
    "zoom_hotkey_out": ["ctrl", "-"],
    "nod_threshold": 12,
    "shake_threshold": 18,
    "hand_landmarker_task_path": "models/mediapipe/hand_landmarker.task",
    "face_landmarker_task_path": "models/mediapipe/face_landmarker.task",
    "require_calibration": True,
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

    def test_detects_zoom_in_from_two_hands_moving_apart(self) -> None:
        initial = [
            {0: Point(x=0.30, y=0.50), 5: Point(x=0.32, y=0.45), 9: Point(x=0.34, y=0.50), 13: Point(x=0.32, y=0.55), 17: Point(x=0.28, y=0.55)},
            {0: Point(x=0.60, y=0.50), 5: Point(x=0.58, y=0.45), 9: Point(x=0.56, y=0.50), 13: Point(x=0.58, y=0.55), 17: Point(x=0.62, y=0.55)},
        ]
        moved = [
            {0: Point(x=0.20, y=0.50), 5: Point(x=0.22, y=0.45), 9: Point(x=0.24, y=0.50), 13: Point(x=0.22, y=0.55), 17: Point(x=0.18, y=0.55)},
            {0: Point(x=0.70, y=0.50), 5: Point(x=0.68, y=0.45), 9: Point(x=0.66, y=0.50), 13: Point(x=0.68, y=0.55), 17: Point(x=0.72, y=0.55)},
        ]
        self.assertIsNone(self.eyes._detect_zoom(initial))
        event = self.eyes._detect_zoom(moved)
        self.assertIsNotNone(event)
        self.assertEqual("zoom_in", event.name)
        self.assertEqual("hotkey", event.action)
        self.assertEqual(["ctrl", "+"], event.value)

    def test_detects_head_nod_confirmation(self) -> None:
        event = self.eyes.interpret_observation(head_pose=HeadPose(pitch=15.0))
        self.assertIsNotNone(event)
        self.assertEqual("confirm", event.action)
        self.assertEqual("yes", event.value)

    def test_detects_open_hand_for_calibration(self) -> None:
        detected = self.eyes._interpreter.detect_open_hand(
            {
                3: Point(x=0.25, y=0.55),
                4: Point(x=0.12, y=0.50),
                6: Point(x=0.35, y=0.55),
                8: Point(x=0.35, y=0.30),
                10: Point(x=0.45, y=0.58),
                12: Point(x=0.45, y=0.28),
                14: Point(x=0.55, y=0.60),
                16: Point(x=0.55, y=0.32),
                18: Point(x=0.65, y=0.62),
                20: Point(x=0.65, y=0.34),
            }
        )
        self.assertTrue(detected)

    def test_calibration_progresses_through_required_steps(self) -> None:
        self.eyes._process_calibration(
            {
                3: Point(x=0.25, y=0.55),
                4: Point(x=0.12, y=0.50),
                6: Point(x=0.35, y=0.55),
                8: Point(x=0.35, y=0.30),
                10: Point(x=0.45, y=0.58),
                12: Point(x=0.45, y=0.28),
                14: Point(x=0.55, y=0.60),
                16: Point(x=0.55, y=0.32),
                18: Point(x=0.65, y=0.62),
                20: Point(x=0.65, y=0.34),
            },
            None,
        )
        self.eyes._process_calibration(None, HeadPose(pitch=0.0, yaw=0.0, roll=0.0))
        self.eyes._process_calibration(None, HeadPose(pitch=15.0, yaw=0.0, roll=0.0))
        self.eyes._process_calibration(None, HeadPose(pitch=-15.0, yaw=0.0, roll=0.0))
        self.eyes._process_calibration(None, HeadPose(pitch=15.0, yaw=0.0, roll=0.0))
        self.eyes._process_calibration(None, HeadPose(pitch=0.0, yaw=0.0, roll=0.0))
        self.eyes._process_calibration(None, HeadPose(pitch=0.0, yaw=25.0, roll=0.0))
        self.eyes._process_calibration(None, HeadPose(pitch=0.0, yaw=-25.0, roll=0.0))
        status = self.eyes._process_calibration(None, HeadPose(pitch=0.0, yaw=25.0, roll=0.0))
        self.assertTrue(status.completed)

    def test_head_calibration_requires_motion_from_baseline(self) -> None:
        self.eyes._calibration_index = 1
        status = self.eyes._process_calibration(None, HeadPose(pitch=0.0, yaw=0.0, roll=0.0))
        self.assertTrue(status.active)
        self.assertEqual("nod_yes", status.current_step)
        status = self.eyes._process_calibration(None, HeadPose(pitch=14.5, yaw=0.0, roll=0.0))
        self.assertTrue(status.active)
        self.assertEqual("nod_yes", status.current_step)
        status = self.eyes._process_calibration(None, HeadPose(pitch=-20.0, yaw=0.0, roll=0.0))
        self.assertEqual("nod_yes", status.current_step)
        status = self.eyes._process_calibration(None, HeadPose(pitch=30.0, yaw=0.0, roll=0.0))
        self.assertEqual("shake_no", status.current_step)

    def test_debug_skip_can_bypass_head_steps(self) -> None:
        settings = dict(SETTINGS)
        settings["calibration_skip_steps"] = ["nod_yes", "shake_no"]
        eyes = Eyes(settings=settings, on_gesture=lambda _: None, frame_source=lambda: iter(()))
        status = eyes.calibration_status()
        self.assertEqual("show_open_hand", status.current_step)
        eyes._process_calibration(
            {
                3: Point(x=0.25, y=0.55),
                4: Point(x=0.12, y=0.50),
                6: Point(x=0.35, y=0.55),
                8: Point(x=0.35, y=0.30),
                10: Point(x=0.45, y=0.58),
                12: Point(x=0.45, y=0.28),
                14: Point(x=0.55, y=0.60),
                16: Point(x=0.55, y=0.32),
                18: Point(x=0.65, y=0.62),
                20: Point(x=0.65, y=0.34),
            },
            None,
        )
        status = eyes.calibration_status()
        self.assertTrue(status.completed)


if __name__ == "__main__":
    unittest.main()

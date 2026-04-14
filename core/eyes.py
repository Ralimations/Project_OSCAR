from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import mediapipe as mp  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    mp = None


LOGGER = logging.getLogger(__name__)
HAS_MEDIAPIPE_TASKS = bool(mp and hasattr(mp, "tasks"))


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class HeadPose:
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0


@dataclass(frozen=True, slots=True)
class GestureEvent:
    name: str
    action: str
    value: float | str | int | None


@dataclass(frozen=True, slots=True)
class CalibrationStatus:
    active: bool
    completed: bool
    current_step: str
    completed_steps: tuple[str, ...]
    instruction: str


CALIBRATION_SEQUENCE = ("show_open_hand", "nod_yes", "shake_no")
CALIBRATION_INSTRUCTIONS = {
    "show_open_hand": "Show one open hand with all 5 fingers",
    "nod_yes": "Do 2 strong nods up and down",
    "shake_no": "Do 2 strong head shakes left and right",
    "complete": "Calibration complete",
}
FACE_LEFT_EYE = 33
FACE_RIGHT_EYE = 263
FACE_NOSE = 1
FACE_CHIN = 152
HAND_TIPS = (4, 8, 12, 16, 20)
HAND_PIPS = (3, 6, 10, 14, 18)


def _distance(point_a: Point, point_b: Point) -> float:
    return ((point_a.x - point_b.x) ** 2 + (point_a.y - point_b.y) ** 2) ** 0.5


class GestureInterpreter:
    """Pure gesture math that can be unit-tested without camera dependencies."""

    def __init__(self, settings: Mapping[str, float | int | str | bool | Sequence[str]]) -> None:
        self._pinch_threshold = float(settings["pinch_threshold"])
        self._scroll_deadzone = float(settings["scroll_deadzone"])
        self._scroll_scale = float(settings["scroll_scale"])
        self._nod_threshold = float(settings["nod_threshold"])
        self._shake_threshold = float(settings["shake_threshold"])

    def interpret(
        self,
        hand_landmarks: Mapping[int, Point] | Sequence[Mapping[int, Point]] | None = None,
        head_pose: HeadPose | None = None,
    ) -> GestureEvent | None:
        hands = self._normalize_hands(hand_landmarks)
        for hand in hands:
            click_event = self._detect_click(hand)
            if click_event:
                return click_event

            scroll_event = self._detect_scroll(hand)
            if scroll_event:
                return scroll_event

        if head_pose:
            if head_pose.pitch >= self._nod_threshold:
                return GestureEvent(name="confirm_yes", action="confirm", value="yes")
            if abs(head_pose.yaw) >= self._shake_threshold:
                return GestureEvent(name="confirm_no", action="confirm", value="no")

        return None

    def _normalize_hands(
        self,
        hand_landmarks: Mapping[int, Point] | Sequence[Mapping[int, Point]] | None,
    ) -> list[Mapping[int, Point]]:
        if hand_landmarks is None:
            return []
        if isinstance(hand_landmarks, Mapping):
            return [hand_landmarks]
        return [hand for hand in hand_landmarks if hand]

    def detect_open_hand(self, hand_landmarks: Mapping[int, Point] | None) -> bool:
        if not hand_landmarks:
            return False

        index_tip = hand_landmarks.get(8)
        index_pip = hand_landmarks.get(6)
        middle_tip = hand_landmarks.get(12)
        middle_pip = hand_landmarks.get(10)
        ring_tip = hand_landmarks.get(16)
        ring_pip = hand_landmarks.get(14)
        pinky_tip = hand_landmarks.get(20)
        pinky_pip = hand_landmarks.get(18)
        thumb_tip = hand_landmarks.get(4)
        thumb_ip = hand_landmarks.get(3)

        if not all([index_tip, index_pip, middle_tip, middle_pip, ring_tip, ring_pip, pinky_tip, pinky_pip, thumb_tip, thumb_ip]):
            return False

        fingers_extended = [
            index_tip.y < index_pip.y,
            middle_tip.y < middle_pip.y,
            ring_tip.y < ring_pip.y,
            pinky_tip.y < pinky_pip.y,
            abs(thumb_tip.x - thumb_ip.x) > 0.04,
        ]
        return all(fingers_extended)

    def calibration_signal(self, hand_landmarks: Mapping[int, Point] | None, head_pose: HeadPose | None, step: str) -> bool:
        if step == "show_open_hand":
            return self.detect_open_hand(hand_landmarks)
        if head_pose is None:
            return False
        if step == "nod_yes":
            return head_pose.pitch >= self._nod_threshold
        if step == "shake_no":
            return abs(head_pose.yaw) >= self._shake_threshold
        return False

    def _detect_click(self, hand_landmarks: Mapping[int, Point]) -> GestureEvent | None:
        thumb_tip = hand_landmarks.get(4)
        index_tip = hand_landmarks.get(8)
        if not thumb_tip or not index_tip:
            return None

        if _distance(thumb_tip, index_tip) <= self._pinch_threshold:
            return GestureEvent(name="click", action="click", value=None)
        return None

    def _detect_scroll(self, hand_landmarks: Mapping[int, Point]) -> GestureEvent | None:
        index_tip = hand_landmarks.get(8)
        middle_tip = hand_landmarks.get(12)
        if not index_tip or not middle_tip:
            return None

        vertical_delta = middle_tip.y - index_tip.y
        if abs(vertical_delta) < self._scroll_deadzone:
            return None

        amount = int(-vertical_delta * self._scroll_scale)
        if amount == 0:
            return None
        return GestureEvent(name="scroll", action="scroll", value=amount)


class Eyes:
    """Continuous gesture loop with startup calibration."""

    def __init__(
        self,
        settings: Mapping[str, float | int | str | bool],
        on_gesture: Callable[[GestureEvent], None],
        frame_source: Callable[[], Iterable[object]] | None = None,
        frame_observer: Callable[[object, list[dict[int, tuple[int, int]]] | None, list[tuple[int, int]] | None, str | None, CalibrationStatus, HeadPose | None], None] | None = None,
    ) -> None:
        self._fps = int(settings["dormant_fps"])
        self._cooldown_seconds = float(settings["gesture_cooldown_ms"]) / 1000.0
        self._mirror_camera = bool(settings.get("mirror_camera", True))
        self._hand_landmarker_task_path = str(settings.get("hand_landmarker_task_path", ""))
        self._face_landmarker_task_path = str(settings.get("face_landmarker_task_path", ""))
        self._require_calibration = bool(settings.get("require_calibration", True))
        self._max_hands = int(settings.get("max_hands", 2))
        self._zoom_distance_threshold = float(settings.get("zoom_distance_threshold", 0.08))
        self._zoom_hotkey_in = list(settings.get("zoom_hotkey_in", ["ctrl", "+"]))  # type: ignore[arg-type]
        self._zoom_hotkey_out = list(settings.get("zoom_hotkey_out", ["ctrl", "-"]))  # type: ignore[arg-type]
        self._calibration_skip_steps = {
            str(step) for step in settings.get("calibration_skip_steps", [])  # type: ignore[arg-type]
        }
        self._interpreter = GestureInterpreter(settings)  # type: ignore[arg-type]
        self._on_gesture = on_gesture
        self._frame_source = frame_source or self._camera_frames
        self._frame_observer = frame_observer
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="oscar-eyes", daemon=True)
        self._paused = threading.Event()
        self._last_event_at = 0.0
        self._hands = None
        self._face = None
        self._calibration_index = 0
        self._calibration_completed: list[str] = []
        self._calibration_head_baseline: HeadPose | None = None
        self._motion_direction: str | None = None
        self._motion_switches = 0
        self._last_two_hand_distance: float | None = None

    def start(self) -> None:
        self._thread.start()

    def diagnostics(self) -> dict[str, bool]:
        hand_task_path = Path(self._hand_landmarker_task_path) if self._hand_landmarker_task_path else None
        face_task_path = Path(self._face_landmarker_task_path) if self._face_landmarker_task_path else None
        return {
            "mediapipe_tasks_available": HAS_MEDIAPIPE_TASKS,
            "hand_landmarker_model_present": bool(hand_task_path and hand_task_path.exists()),
            "face_landmarker_model_present": bool(face_task_path and face_task_path.exists()),
            "calibration_required": self._require_calibration,
        }

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.5)
        if self._hands is not None:
            self._hands.close()
            self._hands = None
        if self._face is not None:
            self._face.close()
            self._face = None

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def interpret_observation(
        self,
        hand_landmarks: Mapping[int, Point] | Sequence[Mapping[int, Point]] | None = None,
        head_pose: HeadPose | None = None,
    ) -> GestureEvent | None:
        return self._interpreter.interpret(hand_landmarks=hand_landmarks, head_pose=head_pose)

    def calibration_status(self) -> CalibrationStatus:
        self._advance_skipped_steps()
        active = self._require_calibration and len(self._calibration_completed) < len(CALIBRATION_SEQUENCE)
        current_step = CALIBRATION_SEQUENCE[self._calibration_index] if active else "complete"
        return CalibrationStatus(
            active=active,
            completed=not active,
            current_step=current_step,
            completed_steps=tuple(self._calibration_completed),
            instruction=CALIBRATION_INSTRUCTIONS[current_step],
        )

    def _run(self) -> None:
        frame_interval = 1.0 / max(self._fps, 1)
        for frame in self._frame_source():
            if self._stop_event.is_set():
                return
            if self._paused.is_set():
                time.sleep(frame_interval)
                continue

            event = self._detect_from_frame(frame)
            now = time.monotonic()
            if event and (now - self._last_event_at) >= self._cooldown_seconds:
                self._last_event_at = now
                self._on_gesture(event)
            time.sleep(frame_interval)

    def _detect_from_frame(self, frame: object) -> GestureEvent | None:
        if mp is None or cv2 is None or not HAS_MEDIAPIPE_TASKS:
            return None

        if self._hands is None:
            self._hands = self._create_hand_landmarker()
        if self._face is None:
            self._face = self._create_face_landmarker()
        if self._hands is None:
            return None

        processed_frame = cv2.flip(frame, 1) if self._mirror_camera else frame
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        hand_landmarks = self._extract_hand_landmarks(self._hands.detect(mp_image))
        head_pose, face_landmarks = self._extract_head_pose(self._face.detect(mp_image) if self._face is not None else None)

        primary_hand = hand_landmarks[0] if hand_landmarks else None
        calibration_status = self._process_calibration(primary_hand, head_pose)
        event = None
        if not calibration_status.active:
            event = self._detect_zoom(hand_landmarks)
            if event is None:
                event = self._interpreter.interpret(hand_landmarks=hand_landmarks, head_pose=head_pose)
        else:
            self._last_two_hand_distance = None

        if self._frame_observer is not None:
            preview_points = None
            if hand_landmarks:
                preview_points = []
                for hand in hand_landmarks:
                    preview_points.append(
                        {
                            index: (int(point.x * processed_frame.shape[1]), int(point.y * processed_frame.shape[0]))
                            for index, point in hand.items()
                        }
                    )
            preview_face_points = None
            if face_landmarks:
                preview_face_points = [
                    (int(point.x * processed_frame.shape[1]), int(point.y * processed_frame.shape[0]))
                    for point in face_landmarks
                ]
            self._frame_observer(
                processed_frame,
                preview_points,
                preview_face_points,
                event.name if event else None,
                calibration_status,
                head_pose,
            )
        return event

    def _process_calibration(self, hand_landmarks: Mapping[int, Point] | None, head_pose: HeadPose | None) -> CalibrationStatus:
        if not self._require_calibration:
            return CalibrationStatus(active=False, completed=True, current_step="complete", completed_steps=tuple(), instruction=CALIBRATION_INSTRUCTIONS["complete"])

        self._advance_skipped_steps()
        if self._calibration_index >= len(CALIBRATION_SEQUENCE):
            return self.calibration_status()

        current_step = CALIBRATION_SEQUENCE[self._calibration_index]
        if current_step == "show_open_hand":
            if self._interpreter.calibration_signal(hand_landmarks, head_pose, current_step):
                self._complete_calibration_step(current_step)
            return self.calibration_status()

        if head_pose is None:
            self._calibration_head_baseline = None
            self._reset_motion_tracker()
            return self.calibration_status()

        if self._calibration_head_baseline is None:
            self._calibration_head_baseline = head_pose
            self._reset_motion_tracker()
            return self.calibration_status()

        if self._head_motion_matches_step(head_pose, self._calibration_head_baseline, current_step):
            self._complete_calibration_step(current_step)
            self._calibration_head_baseline = None
            self._reset_motion_tracker()
        return self.calibration_status()

    def _complete_calibration_step(self, step: str) -> None:
        self._calibration_completed.append(step)
        self._calibration_index += 1
        if self._calibration_index >= len(CALIBRATION_SEQUENCE):
            LOGGER.info("Calibration completed: %s", ", ".join(self._calibration_completed))
        else:
            LOGGER.info("Calibration step completed: %s", step)
        self._advance_skipped_steps()

    def _advance_skipped_steps(self) -> None:
        while self._calibration_index < len(CALIBRATION_SEQUENCE):
            step = CALIBRATION_SEQUENCE[self._calibration_index]
            if step not in self._calibration_skip_steps:
                return
            LOGGER.info("Calibration step skipped by configuration: %s", step)
            self._calibration_completed.append(f"{step}(skipped)")
            self._calibration_index += 1

    def _head_motion_matches_step(self, head_pose: HeadPose, baseline: HeadPose, step: str) -> bool:
        pitch_delta = head_pose.pitch - baseline.pitch
        yaw_delta = head_pose.yaw - baseline.yaw
        if step == "nod_yes":
            return self._count_repeated_motion(
                value=pitch_delta,
                threshold=self._interpreter._nod_threshold,
                cross_value=yaw_delta,
                cross_limit=self._interpreter._shake_threshold * 0.7,
            )
        if step == "shake_no":
            return self._count_repeated_motion(
                value=yaw_delta,
                threshold=self._interpreter._shake_threshold,
                cross_value=pitch_delta,
                cross_limit=self._interpreter._nod_threshold * 0.7,
            )
        return False

    def _count_repeated_motion(self, value: float, threshold: float, cross_value: float, cross_limit: float) -> bool:
        if abs(cross_value) > abs(cross_limit):
            self._reset_motion_tracker()
            return False

        if value >= threshold:
            direction = "positive"
        elif value <= -threshold:
            direction = "negative"
        else:
            direction = None

        if direction is None:
            return False

        if self._motion_direction is None:
            self._motion_direction = direction
            self._motion_switches = 0
            return False

        if direction != self._motion_direction:
            self._motion_direction = direction
            self._motion_switches += 1

        return self._motion_switches >= 2

    def _reset_motion_tracker(self) -> None:
        self._motion_direction = None
        self._motion_switches = 0

    def _detect_zoom(self, hand_landmarks: Sequence[Mapping[int, Point]] | None) -> GestureEvent | None:
        if not hand_landmarks or len(hand_landmarks) < 2:
            self._last_two_hand_distance = None
            return None

        first_center = self._hand_center(hand_landmarks[0])
        second_center = self._hand_center(hand_landmarks[1])
        if first_center is None or second_center is None:
            self._last_two_hand_distance = None
            return None

        distance = _distance(first_center, second_center)
        if self._last_two_hand_distance is None:
            self._last_two_hand_distance = distance
            return None

        delta = distance - self._last_two_hand_distance
        self._last_two_hand_distance = distance
        if abs(delta) < self._zoom_distance_threshold:
            return None
        if delta > 0:
            return GestureEvent(name="zoom_in", action="hotkey", value=self._zoom_hotkey_in)
        return GestureEvent(name="zoom_out", action="hotkey", value=self._zoom_hotkey_out)

    def _hand_center(self, hand_landmarks: Mapping[int, Point]) -> Point | None:
        anchors = [hand_landmarks.get(index) for index in (0, 5, 9, 13, 17)]
        points = [point for point in anchors if point is not None]
        if not points:
            return None
        return Point(
            x=sum(point.x for point in points) / len(points),
            y=sum(point.y for point in points) / len(points),
        )

    def _extract_hand_landmarks(self, results) -> list[Mapping[int, Point]] | None:
        if not getattr(results, "hand_landmarks", None):
            return None
        hands: list[Mapping[int, Point]] = []
        for hand in results.hand_landmarks[: self._max_hands]:
            hands.append({index: Point(x=landmark.x, y=landmark.y) for index, landmark in enumerate(hand)})
        return hands or None

    def _extract_head_pose(self, results) -> tuple[HeadPose | None, list[Point] | None]:
        if results is None or not getattr(results, "face_landmarks", None):
            return None, None

        face_landmarks = results.face_landmarks[0]
        try:
            left_eye = face_landmarks[FACE_LEFT_EYE]
            right_eye = face_landmarks[FACE_RIGHT_EYE]
            nose = face_landmarks[FACE_NOSE]
            chin = face_landmarks[FACE_CHIN]
        except IndexError:
            return None, None

        eye_mid_x = (left_eye.x + right_eye.x) / 2.0
        eye_mid_y = (left_eye.y + right_eye.y) / 2.0
        eye_dx = right_eye.x - left_eye.x
        eye_dy = right_eye.y - left_eye.y
        eye_distance = max(math.hypot(eye_dx, eye_dy), 1e-6)
        yaw = (nose.x - eye_mid_x) / eye_distance * 100.0
        pitch = (nose.y - ((eye_mid_y + chin.y) / 2.0)) / eye_distance * -120.0
        roll = math.degrees(math.atan2(eye_dy, eye_dx))
        return HeadPose(pitch=pitch, yaw=yaw, roll=roll), [Point(x=landmark.x, y=landmark.y) for landmark in face_landmarks]

    def _create_hand_landmarker(self):
        task_path = Path(self._hand_landmarker_task_path) if self._hand_landmarker_task_path else None
        if not task_path or not task_path.exists():
            LOGGER.warning("Hand landmarker task file is missing: %s. Gesture camera loop is idle.", task_path or "<unset>")
            return None

        try:
            base_options = mp.tasks.BaseOptions(model_asset_path=str(task_path.resolve()))
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_hands=self._max_hands,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return mp.tasks.vision.HandLandmarker.create_from_options(options)
        except Exception:
            LOGGER.exception("Failed to initialize MediaPipe HandLandmarker.")
            return None

    def _create_face_landmarker(self):
        task_path = Path(self._face_landmarker_task_path) if self._face_landmarker_task_path else None
        if not task_path or not task_path.exists():
            LOGGER.warning("Face landmarker task file is missing: %s. Head calibration will be limited.", task_path or "<unset>")
            return None

        try:
            base_options = mp.tasks.BaseOptions(model_asset_path=str(task_path.resolve()))
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return mp.tasks.vision.FaceLandmarker.create_from_options(options)
        except Exception:
            LOGGER.exception("Failed to initialize MediaPipe FaceLandmarker.")
            return None

    def _camera_frames(self) -> Iterable[object]:
        if cv2 is None:
            LOGGER.warning("OpenCV is not installed; Eyes loop is running without camera input.")
            while not self._stop_event.is_set():
                yield object()
            return
        if not HAS_MEDIAPIPE_TASKS:
            LOGGER.warning("Installed mediapipe package does not expose the Tasks API; gesture camera loop is idle.")
            while not self._stop_event.is_set():
                yield object()
            return

        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            LOGGER.warning("Camera could not be opened; Eyes loop is idle.")
            while not self._stop_event.is_set():
                yield object()
            return

        try:
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    LOGGER.debug("Camera frame read failed; retrying.")
                    continue
                yield frame
        finally:
            capture.release()

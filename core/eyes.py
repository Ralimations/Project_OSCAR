from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import mediapipe as mp  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    mp = None


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class HeadPose:
    pitch: float = 0.0
    yaw: float = 0.0


@dataclass(frozen=True, slots=True)
class GestureEvent:
    name: str
    action: str
    value: float | str | int | None


def _distance(point_a: Point, point_b: Point) -> float:
    return ((point_a.x - point_b.x) ** 2 + (point_a.y - point_b.y) ** 2) ** 0.5


class GestureInterpreter:
    """Pure gesture math that can be unit-tested without camera dependencies."""

    def __init__(self, settings: Mapping[str, float]) -> None:
        self._pinch_threshold = float(settings["pinch_threshold"])
        self._scroll_deadzone = float(settings["scroll_deadzone"])
        self._scroll_scale = float(settings["scroll_scale"])
        self._nod_threshold = float(settings["nod_threshold"])
        self._shake_threshold = float(settings["shake_threshold"])

    def interpret(
        self,
        hand_landmarks: Mapping[int, Point] | None = None,
        head_pose: HeadPose | None = None,
    ) -> GestureEvent | None:
        if hand_landmarks:
            click_event = self._detect_click(hand_landmarks)
            if click_event:
                return click_event

            scroll_event = self._detect_scroll(hand_landmarks)
            if scroll_event:
                return scroll_event

        if head_pose:
            if head_pose.pitch >= self._nod_threshold:
                return GestureEvent(name="confirm_yes", action="confirm", value="yes")
            if abs(head_pose.yaw) >= self._shake_threshold:
                return GestureEvent(name="confirm_no", action="confirm", value="no")

        return None

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
    """Continuous gesture loop for OSCAR's dormant state."""

    def __init__(
        self,
        settings: Mapping[str, float | int],
        on_gesture: Callable[[GestureEvent], None],
        frame_source: Callable[[], Iterable[object]] | None = None,
    ) -> None:
        self._fps = int(settings["dormant_fps"])
        self._cooldown_seconds = float(settings["gesture_cooldown_ms"]) / 1000.0
        self._interpreter = GestureInterpreter(settings)
        self._on_gesture = on_gesture
        self._frame_source = frame_source or self._camera_frames
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="oscar-eyes", daemon=True)
        self._paused = threading.Event()
        self._last_event_at = 0.0
        self._hands = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.5)
        if self._hands is not None:
            self._hands.close()
            self._hands = None

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def interpret_observation(
        self,
        hand_landmarks: Mapping[int, Point] | None = None,
        head_pose: HeadPose | None = None,
    ) -> GestureEvent | None:
        return self._interpreter.interpret(hand_landmarks=hand_landmarks, head_pose=head_pose)

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
        if mp is None or cv2 is None:
            return None
        if self._hands is None:
            # Keep the model complexity at zero to stay inside the i3 dormant-state budget.
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb_frame)
        if not results.multi_hand_landmarks:
            return None

        first_hand = results.multi_hand_landmarks[0]
        hand_landmarks = {
            index: Point(x=landmark.x, y=landmark.y)
            for index, landmark in enumerate(first_hand.landmark)
        }
        return self._interpreter.interpret(hand_landmarks=hand_landmarks)

    def _camera_frames(self) -> Iterable[object]:
        if cv2 is None:
            LOGGER.warning("OpenCV is not installed; Eyes loop is running without camera input.")
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

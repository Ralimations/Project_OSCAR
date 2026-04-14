from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SandboxState:
    cursor_x: int = 320
    cursor_y: int = 180
    last_action: str = "idle"
    last_value: str = ""
    click_flash_until: float = 0.0
    scroll_events: list[str] = field(default_factory=list)
    calibration_step: str = "show_open_hand"
    calibration_completed: list[str] = field(default_factory=list)
    calibration_active: bool = True


class DevOverlay:
    """Development-only UI for reviewing gesture tracking and sandboxed actions."""

    def __init__(
        self,
        tracking_window: bool = True,
        sandbox_window: bool = True,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self._tracking_window = tracking_window
        self._sandbox_window = sandbox_window
        self._on_close = on_close
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="oscar-dev-overlay", daemon=True)
        self._tracking_frame = None
        self._sandbox = SandboxState()
        self._lock = threading.Lock()
        self._events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._started = False
        self._close_notified = False
        self._tracking_window_shown = False
        self._sandbox_window_shown = False

    def start(self) -> None:
        if self._started or cv2 is None or np is None:
            if cv2 is None or np is None:
                LOGGER.warning("Dev overlay disabled because OpenCV or NumPy is unavailable.")
            return
        self._started = True
        self._thread.start()

    def stop(self) -> None:
        if not self._started:
            return
        self._stop_event.set()
        self._events.put(("shutdown", None))
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)

    def update_tracking(
        self,
        frame,
        hand_landmarks: list[dict[int, tuple[int, int]]] | None,
        face_landmarks: list[tuple[int, int]] | None,
        event_name: str | None,
        calibration_step: str,
        calibration_completed: tuple[str, ...],
        calibration_active: bool,
        head_pose,
    ) -> None:
        if not self._started or cv2 is None:
            return
        display = frame.copy()
        if face_landmarks:
            for x, y in face_landmarks:
                cv2.circle(display, (x, y), 1, (120, 255, 120), -1)
        if hand_landmarks:
            colors = ((255, 180, 0), (80, 255, 160))
            for hand_index, hand in enumerate(hand_landmarks):
                base_color = colors[hand_index % len(colors)]
                for index, (x, y) in hand.items():
                    radius = 6 if index in {4, 8, 12} else 3
                    color = (0, 255, 255) if index in {4, 8, 12} else base_color
                    cv2.circle(display, (x, y), radius, color, -1)
                for key in (4, 8, 12):
                    if key in hand:
                        x, y = hand[key]
                        cv2.putText(display, f"{hand_index}:{key}", (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            index_tip = hand_landmarks[0].get(8)
            if index_tip:
                self._events.put(("cursor", index_tip))
        status = calibration_step if calibration_active else (event_name or "tracking")
        cv2.putText(display, f"OSCAR Tracking | {status}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        instruction = calibration_step.replace("_", " ") if calibration_active else "tracking"
        cv2.putText(display, instruction, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 230, 160), 2)
        if head_pose is not None:
            cv2.putText(
                display,
                f"pitch={head_pose.pitch:.1f} yaw={head_pose.yaw:.1f} roll={head_pose.roll:.1f}",
                (12, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 220, 255),
                1,
            )
        with self._lock:
            self._sandbox.calibration_step = calibration_step
            self._sandbox.calibration_completed = list(calibration_completed)
            self._sandbox.calibration_active = calibration_active
        with self._lock:
            self._tracking_frame = display

    def record_action(self, action: str, value: object) -> None:
        if not self._started:
            return
        self._events.put(("action", (action, value)))

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    event_type, payload = self._events.get(timeout=0.03)
                    self._apply_event(event_type, payload)
                except queue.Empty:
                    pass

                if self._tracking_window:
                    with self._lock:
                        tracking_frame = None if self._tracking_frame is None else self._tracking_frame.copy()
                    if tracking_frame is not None:
                        cv2.imshow("OSCAR Tracking", tracking_frame)
                        self._tracking_window_shown = True

                if self._sandbox_window:
                    cv2.imshow("OSCAR Sandbox", self._render_sandbox())
                    self._sandbox_window_shown = True

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    self._request_close()
                    break

                if self._tracking_window and self._tracking_window_shown and self._window_closed("OSCAR Tracking"):
                    self._request_close()
                    break
                if self._sandbox_window and self._sandbox_window_shown and self._window_closed("OSCAR Sandbox"):
                    self._request_close()
                    break
        except Exception:
            LOGGER.exception("Dev overlay loop failed.")
            self._request_close()
        finally:
            self._safe_destroy_window("OSCAR Tracking")
            self._safe_destroy_window("OSCAR Sandbox")

    def _window_closed(self, name: str) -> bool:
        try:
            return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
        except Exception:
            return True

    def _safe_destroy_window(self, name: str) -> None:
        try:
            cv2.destroyWindow(name)
        except Exception:
            return

    def _request_close(self) -> None:
        self._stop_event.set()
        if self._close_notified:
            return
        self._close_notified = True
        if self._on_close is not None:
            try:
                self._on_close()
            except Exception:
                LOGGER.exception("Dev overlay close callback failed.")

    def _apply_event(self, event_type: str, payload: object) -> None:
        now = time.monotonic()
        if event_type == "cursor" and isinstance(payload, tuple):
            x, y = payload
            self._sandbox.cursor_x = max(20, min(620, int(x)))
            self._sandbox.cursor_y = max(20, min(340, int(y)))
            return
        if event_type == "action" and isinstance(payload, tuple):
            action, value = payload
            self._sandbox.last_action = str(action)
            self._sandbox.last_value = "" if value is None else str(value)
            if action == "click":
                self._sandbox.click_flash_until = now + 0.25
            if action == "scroll":
                self._sandbox.scroll_events.append(str(value))
                self._sandbox.scroll_events = self._sandbox.scroll_events[-6:]

    def _render_sandbox(self):
        canvas = np.zeros((360, 640, 3), dtype=np.uint8)
        canvas[:] = (24, 24, 30)
        cv2.rectangle(canvas, (18, 18), (622, 342), (70, 70, 90), 2)
        cv2.putText(canvas, "OSCAR Sandbox", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
        cv2.putText(canvas, f"Last action: {self._sandbox.last_action}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)
        cv2.putText(canvas, f"Value: {self._sandbox.last_value}", (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 220), 1)
        cv2.putText(canvas, "Cursor preview", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.putText(canvas, "Calibration", (340, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        cv2.putText(canvas, f"Current: {self._sandbox.calibration_step.replace('_', ' ')}", (340, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)
        completed_label = ", ".join(self._sandbox.calibration_completed) if self._sandbox.calibration_completed else "none"
        cv2.putText(canvas, f"Done: {completed_label}", (340, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 220), 1)
        skipped_steps = [step for step in self._sandbox.calibration_completed if "(skipped)" in step]
        skipped_label = ", ".join(skipped_steps) if skipped_steps else "none"
        cv2.putText(canvas, f"Skipped: {skipped_label}", (340, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 190, 120), 1)
        if self._sandbox.calibration_active:
            cv2.putText(canvas, "Follow prompt before control is enabled", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 210, 120), 1)

        click_active = time.monotonic() < self._sandbox.click_flash_until
        cursor_color = (0, 80, 255) if click_active else (0, 220, 255)
        cv2.circle(canvas, (self._sandbox.cursor_x, self._sandbox.cursor_y), 10, cursor_color, 2)
        cv2.line(canvas, (self._sandbox.cursor_x - 14, self._sandbox.cursor_y), (self._sandbox.cursor_x + 14, self._sandbox.cursor_y), cursor_color, 1)
        cv2.line(canvas, (self._sandbox.cursor_x, self._sandbox.cursor_y - 14), (self._sandbox.cursor_x, self._sandbox.cursor_y + 14), cursor_color, 1)

        cv2.putText(canvas, "Recent scroll values", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        for idx, value in enumerate(reversed(self._sandbox.scroll_events)):
            cv2.putText(canvas, value, (24, 238 + (idx * 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)

        cv2.putText(canvas, "Development mode sandboxed actions", (20, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 210, 120), 1)
        return canvas

from __future__ import annotations

import logging
import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    import pyautogui  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pyautogui = None


LOGGER = logging.getLogger(__name__)
KNOWN_APP_PATHS: dict[str, list[str]] = {
    "brave": [
        r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
        r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
    ],
    "telegram": [
        str(Path.home() / "AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Telegram Desktop/Telegram.lnk"),
        str(Path.home() / "AppData/Local/Programs/Telegram Desktop/Telegram.exe"),
        str(Path.home() / "AppData/Roaming/Telegram Desktop/Telegram.exe"),
    ],
    "notepad": [
        r"C:\Windows\System32\notepad.exe",
    ],
}


@dataclass(frozen=True, slots=True)
class ActionRequest:
    action: str
    value: Any = None
    source: str = "system"


class ActionController:
    """Threaded action dispatcher so the orchestrator never blocks on IO."""

    def __init__(self, dry_run: bool = False, action_observer: Callable[[ActionRequest], None] | None = None) -> None:
        self._queue: queue.Queue[ActionRequest] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="oscar-actions", daemon=True)
        self._dry_run = dry_run
        self._action_observer = action_observer
        self._handlers: dict[str, Callable[[Any], None]] = {
            "click": self._click,
            "scroll": self._scroll,
            "press_key": self._press_key,
            "hotkey": self._hotkey,
            "open_app": self._open_app,
            "write_text": self._write_text,
            "wait": self._wait,
            "sequence": self._sequence,
        }

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._queue.put(ActionRequest(action="__shutdown__"))
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def dispatch(self, request: ActionRequest) -> None:
        self._queue.put(request)

    def register_handler(self, action: str, handler: Callable[[Any], None]) -> None:
        self._handlers[action] = handler

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            request = self._queue.get()
            if request.action == "__shutdown__":
                return
            self._execute_request(request)

    def _execute_request(self, request: ActionRequest) -> None:
        if self._action_observer is not None:
            self._action_observer(request)
        handler = self._handlers.get(request.action)
        if handler is None:
            LOGGER.warning("Unsupported action '%s' from %s.", request.action, request.source)
            return

        try:
            handler(request.value)
        except Exception:  # pragma: no cover - defensive runtime logging
            LOGGER.exception("Action '%s' failed.", request.action)

    def _click(self, _: Any) -> None:
        if pyautogui is None or self._dry_run:
            LOGGER.info("Simulated click action.")
            return
        pyautogui.click()

    def _scroll(self, amount: Any) -> None:
        if pyautogui is None or self._dry_run:
            LOGGER.info("Simulated scroll: %s", amount)
            return
        pyautogui.scroll(int(amount))

    def _press_key(self, key: Any) -> None:
        if pyautogui is None or self._dry_run:
            LOGGER.info("Simulated key press: %s", key)
            return
        pyautogui.press(str(key))

    def _hotkey(self, keys: Any) -> None:
        if pyautogui is None or self._dry_run:
            LOGGER.info("Simulated hotkey: %s", keys)
            return
        if not isinstance(keys, (list, tuple)):
            raise TypeError("Hotkey action requires a list or tuple of keys.")
        pyautogui.hotkey(*[str(key) for key in keys])

    def _open_app(self, command: Any) -> None:
        if self._dry_run:
            LOGGER.info("Simulated app launch: %s", command)
            return
        if isinstance(command, str):
            resolved = self._resolve_app_command(command)
            self._launch_windows_target(resolved)
            return
        if isinstance(command, (list, tuple)):
            subprocess.Popen(list(command), shell=False)
            return
        raise TypeError("open_app requires a command string or argv list.")

    def _resolve_app_command(self, command: str) -> str:
        normalized = command.strip()
        lowered = normalized.lower()

        if os.path.exists(normalized):
            return normalized

        if shutil.which(normalized):
            return normalized

        candidates = KNOWN_APP_PATHS.get(lowered, [])
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        return normalized

    def _launch_windows_target(self, target: str) -> None:
        if os.name == "nt":
            if os.path.exists(target):
                os.startfile(target)  # type: ignore[attr-defined]
                return
            subprocess.Popen(["cmd", "/c", "start", "", target], shell=False)
            return
        subprocess.Popen(target, shell=True)

    def _write_text(self, text: Any) -> None:
        if pyautogui is None or self._dry_run:
            LOGGER.info("Simulated text input: %s", text)
            return
        pyautogui.write(str(text))

    def _wait(self, seconds: Any) -> None:
        time.sleep(float(seconds))

    def _sequence(self, steps: Any) -> None:
        if not isinstance(steps, list):
            raise TypeError("sequence action requires a list of action steps.")

        for step in steps:
            if not isinstance(step, dict) or "action" not in step:
                raise TypeError("Each sequence step must be a dict with an action key.")
            nested = ActionRequest(
                action=str(step["action"]),
                value=step.get("value"),
                source="sequence",
            )
            self._execute_request(nested)

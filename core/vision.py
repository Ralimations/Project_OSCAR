from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable

try:
    import pyautogui  # type: ignore
except ImportError:  # pragma: no cover
    pyautogui = None


class Vision:
    """Tactical screenshot handler for the visual route."""

    def __init__(self, answer_backend: Callable[[str, str], str] | None = None) -> None:
        self._answer_backend = answer_backend

    def answer_visual_question(self, prompt: str) -> str:
        screenshot_path = self.capture_snapshot()
        try:
            if self._answer_backend is not None:
                return self._answer_backend(prompt, screenshot_path)
            return "Visual routing is wired, but no VLM backend is configured yet."
        finally:
            if screenshot_path and os.path.exists(screenshot_path):
                os.remove(screenshot_path)

    def capture_snapshot(self) -> str:
        if pyautogui is None:
            return ""
        image = pyautogui.screenshot()
        handle, path = tempfile.mkstemp(prefix="oscar_snapshot_", suffix=".png")
        os.close(handle)
        image.save(Path(path))
        return path

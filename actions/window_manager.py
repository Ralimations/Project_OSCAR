from __future__ import annotations

try:
    import win32gui  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    win32gui = None


class WindowManager:
    """Thin wrapper for active-window lookup used by macro routing."""

    def get_active_window_title(self) -> str | None:
        if win32gui is None:
            return None

        handle = win32gui.GetForegroundWindow()
        if not handle:
            return None
        title = win32gui.GetWindowText(handle)
        return title or None

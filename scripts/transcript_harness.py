from __future__ import annotations

import ctypes
import os
import time

from main import OSCAR, configure_logging, configure_threading, load_runtime_config


def _should_release_focus(transcript: str) -> bool:
    normalized = transcript.strip().lower()
    if not normalized:
        return False
    if normalized.startswith("open "):
        return False
    return normalized in {
        "new tab",
        "add new tab",
        "open new tab",
        "close tab",
        "next tab",
        "previous tab",
        "switch tab",
        "go back",
        "back",
        "go forward",
        "forward",
        "refresh",
        "focus address bar",
        "address bar",
        "go to address bar",
        "zoom in",
        "zoom out",
    }


def _temporarily_release_console_focus(transcript: str) -> None:
    if os.name != "nt":
        return
    if not _should_release_focus(transcript):
        return
    try:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        console = kernel32.GetConsoleWindow()
        if not console:
            return
        user32.ShowWindow(console, 6)  # SW_MINIMIZE
        time.sleep(0.35)
    except Exception:
        return


def _restore_console() -> None:
    if os.name != "nt":
        return
    try:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        console = kernel32.GetConsoleWindow()
        if not console:
            return
        user32.ShowWindow(console, 9)  # SW_RESTORE
        user32.SetForegroundWindow(console)
    except Exception:
        return


def main() -> int:
    config = load_runtime_config()
    configure_threading(int(config.settings["hardware"]["cpu_threads"]))
    configure_logging(config.settings["runtime"]["log_level"])
    oscar = OSCAR(config)
    oscar.start()
    print("OSCAR transcript harness")
    print("Type a command and press Enter. Type 'exit' to quit.")
    try:
        while True:
            transcript = input("> ").strip()
            if not transcript:
                continue
            if transcript.lower() in {"exit", "quit"}:
                break
            _temporarily_release_console_focus(transcript)
            oscar._ears.inject_transcript(transcript)  # Intentional dev harness shortcut.
            if _should_release_focus(transcript):
                time.sleep(0.6)
                _restore_console()
    except KeyboardInterrupt:
        pass
    finally:
        oscar.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

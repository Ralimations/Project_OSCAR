from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


class Voice:
    """Piper subprocess wrapper with text-only fallback."""

    def __init__(self, model_path: str | Path | None = None, piper_executable: str | None = None) -> None:
        self._model_path = str(model_path) if model_path else None
        self._piper_executable = piper_executable or "piper"

    def is_ready(self) -> bool:
        return self._has_piper() or self._has_windows_builtin_tts()

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        if self._has_piper():
            subprocess.Popen(
                [self._piper_executable, "--model", self._model_path],
                stdin=subprocess.PIPE,
                text=True,
            ).communicate(text)
            return
        if self._has_windows_builtin_tts():
            self._speak_windows_builtin(text)
            return
        print(f"OSCAR: {text}")

    def _has_piper(self) -> bool:
        return bool(self._model_path and os.path.exists(self._model_path) and shutil.which(self._piper_executable))

    def _has_windows_builtin_tts(self) -> bool:
        return os.name == "nt" and shutil.which("powershell.exe") is not None

    def _speak_windows_builtin(self, text: str) -> None:
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$s.Speak([Console]::In.ReadToEnd())"
        )
        try:
            subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", script],
                input=text,
                text=True,
                check=False,
            )
        except Exception:
            print(f"OSCAR: {text}")

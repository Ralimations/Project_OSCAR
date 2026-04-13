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
        return bool(self._model_path and os.path.exists(self._model_path) and shutil.which(self._piper_executable))

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        if self.is_ready():
            subprocess.Popen(
                [self._piper_executable, "--model", self._model_path],
                stdin=subprocess.PIPE,
                text=True,
            ).communicate(text)
            return
        print(f"OSCAR: {text}")

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    from faster_whisper import WhisperModel  # type: ignore
except ImportError:  # pragma: no cover
    WhisperModel = None

try:
    from openwakeword.model import Model as OpenWakeWordModel  # type: ignore
except ImportError:  # pragma: no cover
    OpenWakeWordModel = None


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TranscriptEvent:
    transcript: str
    source: str = "wake_word"


class Ears:
    """Wake-word and transcription seam with injectable offline fallbacks."""

    def __init__(
        self,
        on_transcript: Callable[[TranscriptEvent], None] | None = None,
        wake_word_model_path: str | Path | None = None,
        whisper_model_path: str | Path | None = None,
    ) -> None:
        self._on_transcript = on_transcript
        self._wake_word_model_path = str(wake_word_model_path) if wake_word_model_path else None
        self._whisper_model_path = str(whisper_model_path) if whisper_model_path else None
        self._queue: queue.Queue[TranscriptEvent] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="oscar-ears", daemon=True)
        self._started = False
        self._wake_word_model = self._load_wake_word_model()
        self._whisper_model = self._load_whisper_model()

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def stop(self) -> None:
        if not self._started:
            return
        self._stop_event.set()
        self._queue.put(TranscriptEvent(transcript="", source="shutdown"))
        if self._thread.is_alive():
            self._thread.join(timeout=1.5)

    def inject_transcript(self, transcript: str, source: str = "wake_word") -> None:
        if not transcript.strip():
            return
        self._queue.put(TranscriptEvent(transcript=transcript.strip(), source=source))

    def trigger_audio_file(self, audio_path: str | Path) -> None:
        transcript = self.transcribe_audio_file(audio_path)
        if transcript:
            self.inject_transcript(transcript=transcript, source="stt")

    def transcribe_audio_file(self, audio_path: str | Path) -> str:
        if self._whisper_model is None:
            return ""
        segments, _ = self._whisper_model.transcribe(str(audio_path), vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments if segment.text.strip())

    def detect_wake_word(self, audio_chunk: list[float]) -> bool:
        if self._wake_word_model is None:
            return False
        predictions = self._wake_word_model.predict(audio_chunk)
        if not predictions:
            return False
        return any(float(score) >= 0.5 for score in predictions.values())

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            event = self._queue.get()
            if self._stop_event.is_set():
                return
            if self._on_transcript is None:
                LOGGER.info("Transcript received without handler: %s", event.transcript)
                continue
            self._on_transcript(event)

    def _load_wake_word_model(self):
        if OpenWakeWordModel is None or not self._wake_word_model_path:
            return None
        try:
            return OpenWakeWordModel(wakeword_models=[self._wake_word_model_path])
        except Exception:
            LOGGER.exception("Failed to initialize openWakeWord model.")
            return None

    def _load_whisper_model(self):
        if WhisperModel is None or not self._whisper_model_path:
            return None
        try:
            return WhisperModel(self._whisper_model_path, device="cpu", compute_type="int8")
        except Exception:
            LOGGER.exception("Failed to initialize Faster-Whisper model.")
            return None

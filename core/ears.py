from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None

try:
    import sounddevice as sd  # type: ignore
except ImportError:  # pragma: no cover
    sd = None

try:
    from openwakeword.model import Model as OpenWakeWordModel  # type: ignore
except ImportError:  # pragma: no cover
    OpenWakeWordModel = None


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AudioCaptureEvent:
    audio_samples: object
    sample_rate: int
    source: str = "wake_word"
    transcript: str | None = None


class Ears:
    """Low-usage wake-word listener that emits raw audio segments."""

    def __init__(
        self,
        on_audio_captured: Callable[[AudioCaptureEvent], None] | None = None,
        wake_word_model_path: str | Path | None = None,
        audio_settings: dict | None = None,
    ) -> None:
        self._on_audio_captured = on_audio_captured
        self._wake_word_model_path = str(wake_word_model_path) if wake_word_model_path else None
        self._audio_settings = audio_settings or {}
        self._audio_enabled = bool(self._audio_settings.get("enabled", True))
        self._sample_rate = int(self._audio_settings.get("sample_rate", 16000))
        self._chunk_ms = int(self._audio_settings.get("chunk_ms", 80))
        self._capture_seconds = float(self._audio_settings.get("capture_seconds", 3.0))
        self._wake_threshold = float(self._audio_settings.get("wake_threshold", 0.5))
        self._input_device = self._audio_settings.get("input_device")
        self._chunk_samples = max(1, int(self._sample_rate * (self._chunk_ms / 1000.0)))
        self._queue: queue.Queue[AudioCaptureEvent] = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker, name="oscar-audio-events", daemon=True)
        self._audio_thread = threading.Thread(target=self._audio_loop, name="oscar-wakeword", daemon=True)
        self._started = False
        self._wake_word_model = self._load_wake_word_model()

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._worker_thread.start()
        if self._audio_enabled:
            self._audio_thread.start()

    def stop(self) -> None:
        if not self._started:
            return
        self._stop_event.set()
        self._queue.put(AudioCaptureEvent(audio_samples=[], sample_rate=self._sample_rate, source="shutdown"))
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.5)
        if self._audio_thread.is_alive():
            self._audio_thread.join(timeout=1.5)

    def diagnostics(self) -> dict[str, object]:
        return {
            "wake_word_model_loaded": self._wake_word_model is not None,
            "audio_capture_enabled": self._audio_enabled,
            "audio_backend_available": sd is not None and np is not None,
            "capture_seconds": self._capture_seconds,
            "audio_input_device": self._input_device if self._input_device is not None else "default",
        }

    def inject_audio_capture(self, audio_samples, source: str = "wake_word", transcript: str | None = None) -> None:
        self._queue.put(
            AudioCaptureEvent(
                audio_samples=audio_samples,
                sample_rate=self._sample_rate,
                source=source,
                transcript=transcript,
            )
        )

    def inject_transcript(self, transcript: str, source: str = "transcript") -> None:
        self.inject_audio_capture(audio_samples=[], source=source, transcript=transcript)

    def detect_wake_word(self, audio_chunk) -> bool:
        if self._wake_word_model is None or np is None:
            return False
        predictions = self._wake_word_model.predict(np.asarray(audio_chunk, dtype=np.float32))
        if not predictions:
            return False
        return any(float(score) >= self._wake_threshold for score in predictions.values())

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            event = self._queue.get()
            if self._stop_event.is_set() or event.source == "shutdown":
                return
            if self._on_audio_captured is not None:
                self._on_audio_captured(event)

    def _audio_loop(self) -> None:
        if not self._audio_enabled:
            return
        if sd is None or np is None:
            LOGGER.warning("Wake-word listener disabled because sounddevice or numpy is unavailable.")
            return
        if self._wake_word_model is None:
            LOGGER.warning("Wake-word listener idle because no local openWakeWord model is loaded.")
            return

        capture_chunks = max(1, int((self._sample_rate * self._capture_seconds) / self._chunk_samples))
        try:
            with sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self._chunk_samples,
                device=self._input_device,
            ) as stream:
                LOGGER.info("Wake-word listener active on input device: %s", self._input_device or "default")
                while not self._stop_event.is_set():
                    frames, _overflowed = stream.read(self._chunk_samples)
                    mono = np.squeeze(frames).astype(np.float32)
                    if mono.size == 0:
                        continue
                    if not self.detect_wake_word(mono):
                        continue
                    LOGGER.info("Wake word detected. Capturing %.1f seconds of raw audio.", self._capture_seconds)
                    utterance_parts = [mono.copy()]
                    for _ in range(capture_chunks):
                        if self._stop_event.is_set():
                            return
                        next_frames, _ = stream.read(self._chunk_samples)
                        utterance_parts.append(np.squeeze(next_frames).astype(np.float32))
                    self.inject_audio_capture(np.concatenate(utterance_parts))
        except Exception:
            LOGGER.exception("Wake-word listener failed.")

    def _load_wake_word_model(self):
        if OpenWakeWordModel is None or not self._wake_word_model_path:
            return None
        model_path = Path(self._wake_word_model_path)
        if not model_path.exists():
            LOGGER.warning("Wake-word model path does not exist: %s", model_path)
            return None
        model_files = sorted(model_path.glob("*.onnx")) if model_path.is_dir() else [model_path]
        if not model_files:
            LOGGER.warning("Wake-word model path contains no ONNX files: %s", model_path)
            return None
        kwargs = {"wakeword_models": [str(path) for path in model_files], "inference_framework": "onnx"}
        feature_dir = model_path / "features" if model_path.is_dir() else model_path.parent / "features"
        melspec_model = feature_dir / "melspectrogram.onnx"
        embedding_model = feature_dir / "embedding_model.onnx"
        if melspec_model.exists():
            kwargs["melspec_model_path"] = str(melspec_model)
        if embedding_model.exists():
            kwargs["embedding_model_path"] = str(embedding_model)
        try:
            return OpenWakeWordModel(**kwargs)
        except Exception:
            LOGGER.exception("Failed to initialize openWakeWord model.")
            return None

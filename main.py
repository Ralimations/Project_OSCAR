from __future__ import annotations

import gc
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from actions.controller import ActionController, ActionRequest
from core.brain import Brain, BrainOutputError
from core.ears import AudioCaptureEvent, Ears
from core.eyes import Eyes, GestureEvent
from core.memory import Memory
from core.ollama_client import OllamaClient
from core.vision import Vision


class OrchestratorState(str, Enum):
    DORMANT = "dormant"
    TRIGGER = "trigger"
    AUDIO = "audio"
    OMNI_INFERENCE = "omni_inference"
    CLEANUP = "cleanup"
    SHUTDOWN = "shutdown"


@dataclass(slots=True)
class RuntimeConfig:
    settings: dict[str, Any]
    gestures: dict[str, Any]
    app_macros: dict[str, Any]


def ensure_runtime_models(model_path: str | Path) -> None:
    target_dir = Path(model_path)
    if target_dir.exists() and any(target_dir.iterdir()):
        return
    print("Downloading and compiling Gemma 4 E4B for Intel i3. Please wait...")
    subprocess.run([sys.executable, "scripts/download_models.py"], check=True)


def configure_threading(cpu_threads: int) -> None:
    thread_count = str(max(1, int(cpu_threads)))
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, thread_count)


def build_ollama_client(model_settings: dict[str, Any]) -> OllamaClient | None:
    backend = str(model_settings.get("inference_backend", "")).strip().lower()
    if backend != "ollama":
        return None
    model_name = str(model_settings.get("ollama_model", "")).strip()
    if not model_name:
        logging.warning("Ollama backend is enabled but no 'ollama_model' is configured.")
        return None
    base_url = str(model_settings.get("ollama_base_url", "http://127.0.0.1:11434")).strip()
    timeout_seconds = int(model_settings.get("ollama_timeout_seconds", 120))
    return OllamaClient(model=model_name, base_url=base_url, timeout_seconds=timeout_seconds)


class OSCAR:
    """Offline dormant/trigger/omni-action orchestrator."""

    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config
        self._runtime_settings = config.settings["runtime"]
        self._hardware_settings = config.settings["hardware"]
        self._screenpipe_settings = config.settings["screenpipe"]
        self._model_settings = config.settings["models"]
        active_profile_name = self._model_settings.get("active_profile", "aibox")
        active_profile_settings = self._model_settings.get("profiles", {}).get(active_profile_name, {})
        self._model_settings = {**self._model_settings, **active_profile_settings}
        self._state = OrchestratorState.DORMANT
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._busy = threading.Lock()
        ollama_client = build_ollama_client(self._model_settings)
        self._controller = ActionController(dry_run=bool(self._runtime_settings.get("sandbox_actions_only", False)))
        self._memory = Memory(database_path=self._screenpipe_settings["database_path"])
        self._vision = Vision(
            answer_backend=ollama_client.answer_visual_question if ollama_client is not None else None
        )
        self._brain = Brain(
            prompt_template_path=Path("config") / "prompt_template.txt",
            model_path=self._model_settings.get("gemma_model_path"),
            device=str(self._hardware_settings.get("openvino_device", "AUTO")),
            max_new_tokens=int(self._model_settings.get("gemma_max_new_tokens", 128)),
            temperature=float(self._model_settings.get("gemma_temperature", 0.0)),
            infer_backend=ollama_client.infer_action if ollama_client is not None else None,
            backend_name=f"ollama:{ollama_client.model}" if ollama_client is not None else None,
        )
        self._ears = Ears(
            on_audio_captured=self._handle_audio_capture,
            wake_word_model_path=self._model_settings["wake_word_model_path"],
            audio_settings=config.settings["audio"],
        )
        self._eyes = Eyes(
            settings={
                **self._runtime_settings,
                **config.settings["gestures"],
                "hand_landmarker_task_path": self._model_settings["hand_landmarker_task_path"],
                "face_landmarker_task_path": self._model_settings["face_landmarker_task_path"],
            },
            on_gesture=self._handle_gesture,
        )

    def start(self) -> None:
        self.log_startup_diagnostics()
        self._controller.start()
        self._ears.start()
        self._eyes.start()

    def stop(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        with self._state_lock:
            self._state = OrchestratorState.SHUTDOWN
        self._eyes.stop()
        self._ears.stop()
        self._controller.stop()

    def run(self) -> None:
        self.start()
        try:
            while not self._stop_event.is_set():
                time.sleep(0.1)
        finally:
            self.stop()

    def _handle_gesture(self, event: GestureEvent) -> None:
        with self._state_lock:
            if self._state != OrchestratorState.DORMANT:
                logging.debug("Ignoring gesture '%s' while state is %s.", event.name, self._state)
                return
            self._state = OrchestratorState.TRIGGER
        try:
            self._controller.dispatch(ActionRequest(action=event.action, value=event.value, source="gesture"))
        finally:
            with self._state_lock:
                self._state = OrchestratorState.DORMANT

    def _handle_audio_capture(self, event: AudioCaptureEvent) -> None:
        if not self._busy.acquire(blocking=False):
            logging.debug("Wake-word trigger ignored because inference is already active.")
            return
        threading.Thread(target=self._process_audio_trigger, args=(event,), name="oscar-omni", daemon=True).start()

    def _process_audio_trigger(self, event: AudioCaptureEvent) -> None:
        screenshot_path = ""
        self._eyes.pause()
        try:
            with self._state_lock:
                self._state = OrchestratorState.AUDIO
            screenshot_path = self._vision.capture_snapshot()
            ocr_history = self._memory.recent_history(limit=int(self._screenpipe_settings.get("history_limit", 8)))
            with self._state_lock:
                self._state = OrchestratorState.OMNI_INFERENCE
            response = self._brain.infer_action(
                audio_samples=event.audio_samples,
                sample_rate=event.sample_rate,
                screenshot_path=screenshot_path or None,
                ocr_history=ocr_history,
            )
            self._controller.dispatch(ActionRequest(action=response.action, value=response.value, source="gemma"))
        except BrainOutputError:
            logging.exception("Gemma action output was not valid JSON.")
        except Exception:
            logging.exception("Omni-inference pipeline failed.")
        finally:
            with self._state_lock:
                self._state = OrchestratorState.CLEANUP
            if screenshot_path:
                Path(screenshot_path).unlink(missing_ok=True)
            self._brain.flush_context()
            gc.collect()
            self._eyes.resume()
            with self._state_lock:
                self._state = OrchestratorState.DORMANT
            if self._busy.locked():
                self._busy.release()

    def log_startup_diagnostics(self) -> None:
        ear_diag = self._ears.diagnostics()
        eye_diag = self._eyes.diagnostics()
        brain_diag = self._brain.diagnostics()
        logging.info("OSCAR startup diagnostics:")
        logging.info("  python: %s", sys.executable)
        logging.info("  wake_word_model_loaded: %s", ear_diag["wake_word_model_loaded"])
        logging.info("  audio_capture_enabled: %s", ear_diag["audio_capture_enabled"])
        logging.info("  mediapipe_tasks_available: %s", eye_diag["mediapipe_tasks_available"])
        logging.info("  hand_landmarker_model_present: %s", eye_diag["hand_landmarker_model_present"])
        logging.info("  face_landmarker_model_present: %s", eye_diag["face_landmarker_model_present"])
        logging.info("  gemma_model_exists: %s", brain_diag["model_path_exists"])
        logging.info("  inference_backend: %s", brain_diag["backend_name"])
        logging.info("  openvino_device: %s", brain_diag["openvino_device"])


def load_runtime_config() -> RuntimeConfig:
    settings = json.loads(Path("config/settings.yaml").read_text(encoding="utf-8"))
    gestures = json.loads(Path("config/gestures.json").read_text(encoding="utf-8"))
    app_macros = json.loads(Path("config/app_macros.json").read_text(encoding="utf-8"))
    return RuntimeConfig(settings=settings, gestures=gestures, app_macros=app_macros)


def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> int:
    config = load_runtime_config()
    configure_threading(int(config.settings["hardware"]["cpu_threads"]))
    configure_logging(config.settings["runtime"]["log_level"])
    active_profile_name = config.settings["models"].get("active_profile", "aibox")
    active_profile = config.settings["models"].get("profiles", {}).get(active_profile_name, {})
    
    if str(active_profile.get("inference_backend", "")).strip().lower() != "ollama":
        model_path = config.settings["models"].get("gemma_model_path", "models/gemma")
        ensure_runtime_models(model_path)
    oscar = OSCAR(config)

    def _shutdown_handler(_signum, _frame) -> None:
        oscar.stop()

    signal.signal(signal.SIGINT, _shutdown_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown_handler)
    oscar.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

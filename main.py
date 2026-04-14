from __future__ import annotations

import gc
import json
import logging
import os
import re
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
from actions.window_manager import WindowManager
from core.brain import Brain, BrainOutputError
from core.dev_overlay import DevOverlay
from core.ears import AudioCaptureEvent, Ears
from core.eyes import Eyes, GestureEvent
from core.memory import Memory
from core.ollama_client import OllamaClient
from core.router import DirectCommandRouter, TranscriptIntent, classify_transcript_intent
from core.vision import Vision
from core.voice import Voice


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
    print(f"Downloading local OSCAR model assets into {target_dir}. Please wait...")
    subprocess.run([sys.executable, "scripts/download_models.py", "--skip-ollama"], check=True)


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


def resolve_active_model_settings(settings: dict[str, Any]) -> dict[str, Any]:
    model_settings = settings["models"]
    active_profile_name = model_settings.get("active_profile", "aibox")
    active_profile_settings = model_settings.get("profiles", {}).get(active_profile_name, {})
    return {**model_settings, **active_profile_settings}


class OSCAR:
    """Offline dormant/trigger/omni-action orchestrator."""

    _WAKE_PHRASE_RE = re.compile(r"^\s*(?:hey|hi|hello)\s+oscar[!,.?\s]*$", re.IGNORECASE)

    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config
        self._runtime_settings = config.settings["runtime"]
        self._hardware_settings = config.settings["hardware"]
        self._screenpipe_settings = config.settings["screenpipe"]
        self._voice_settings = config.settings.get("voice", {})
        self._model_settings = resolve_active_model_settings(config.settings)
        self._state = OrchestratorState.DORMANT
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._busy = threading.Lock()
        self._ready_file = os.environ.get("OSCAR_READY_FILE")
        self._overlay = self._build_overlay()
        ollama_client = build_ollama_client(self._model_settings)
        self._controller = ActionController(
            dry_run=bool(self._runtime_settings.get("sandbox_actions_only", False)),
            action_observer=self._observe_action if self._overlay is not None else None,
        )
        self._memory = Memory(database_path=self._screenpipe_settings["database_path"])
        self._vision = Vision(
            answer_backend=ollama_client.answer_visual_question if ollama_client is not None else None
        )
        self._voice = Voice(
            model_path=self._voice_settings.get("model_path"),
            piper_executable=self._voice_settings.get("piper_executable"),
        )
        self._brain = Brain(
            prompt_template_path=Path("config") / "prompt_template.txt",
            model_path=self._model_settings.get("model_path") or self._model_settings.get("gemma_model_path"),
            device=str(self._hardware_settings.get("openvino_device", "AUTO")),
            max_new_tokens=int(self._model_settings.get("max_new_tokens") or self._model_settings.get("gemma_max_new_tokens", 128)),
            temperature=float(self._model_settings.get("temperature") or self._model_settings.get("gemma_temperature", 0.0)),
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
            frame_observer=self._observe_frame if self._overlay is not None else None,
        )
        self._router = DirectCommandRouter()
        self._window_manager = WindowManager()
        self._controller.register_handler("speak", self._voice.speak)

    def start(self) -> None:
        self.log_startup_diagnostics()
        if self._overlay is not None:
            self._overlay.start()
        self._controller.start()
        self._ears.start()
        self._eyes.start()
        self._signal_ready()

    def stop(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        with self._state_lock:
            self._state = OrchestratorState.SHUTDOWN
        self._eyes.stop()
        self._ears.stop()
        self._controller.stop()
        if self._overlay is not None:
            self._overlay.stop()
        self._clear_ready_signal()

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
        started_at = time.monotonic()
        self._eyes.pause()
        try:
            with self._state_lock:
                self._state = OrchestratorState.AUDIO
            transcript = (event.transcript or "").strip()
            if self._should_acknowledge(event.source, transcript):
                self._dispatch_acknowledgement(source=event.source)
                self._log_latency(route="wake_ack", started_at=started_at)
                return
            if transcript:
                handled = self._route_transcript(transcript, event, started_at=started_at)
                if not handled:
                    with self._state_lock:
                        self._state = OrchestratorState.OMNI_INFERENCE
                    brain_started_at = time.monotonic()
                    response = self._brain.infer_action(
                        audio_samples=event.audio_samples,
                        sample_rate=event.sample_rate,
                        screenshot_path=None,
                        ocr_history="",
                        transcript=transcript,
                    )
                    self._log_latency(
                        route="planner_fallback",
                        started_at=started_at,
                        stage_started_at=brain_started_at,
                    )
                    self._controller.dispatch(ActionRequest(action=response.action, value=response.value, source="brain:fallback"))
            else:
                with self._state_lock:
                    self._state = OrchestratorState.OMNI_INFERENCE
                brain_started_at = time.monotonic()
                response = self._brain.infer_action(
                    audio_samples=event.audio_samples,
                    sample_rate=event.sample_rate,
                    screenshot_path=None,
                    ocr_history="",
                    transcript=None,
                )
                self._log_latency(
                    route="audio_brain",
                    started_at=started_at,
                    stage_started_at=brain_started_at,
                )
                self._controller.dispatch(ActionRequest(action=response.action, value=response.value, source="brain:audio"))
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

    def _route_transcript(self, transcript: str, event: AudioCaptureEvent, started_at: float) -> bool:
        del event
        direct = self._router.route(transcript)
        if direct is not None:
            self._controller.dispatch(direct.request)
            self._log_latency(route=f"direct:{direct.matched_rule}", started_at=started_at)
            return True

        macro_request = self._match_app_macro(transcript)
        if macro_request is not None:
            self._controller.dispatch(macro_request)
            self._log_latency(route=macro_request.source, started_at=started_at)
            return True

        intent = classify_transcript_intent(transcript)
        if intent == TranscriptIntent.MEMORY:
            memory_started_at = time.monotonic()
            history = self._memory.query_recent_text(
                transcript,
                limit=int(self._screenpipe_settings.get("history_limit", 8)),
            )
            self._controller.dispatch(ActionRequest(action="speak", value=history, source="memory"))
            self._log_latency(route="memory", started_at=started_at, stage_started_at=memory_started_at)
            return True

        if intent == TranscriptIntent.VISION:
            vision_started_at = time.monotonic()
            answer = self._vision.answer_visual_question(transcript)
            self._controller.dispatch(ActionRequest(action="speak", value=answer, source="vision"))
            self._log_latency(route="vision", started_at=started_at, stage_started_at=vision_started_at)
            return True

        return False

    def _match_app_macro(self, transcript: str) -> ActionRequest | None:
        title = (self._window_manager.get_active_window_title() or "").strip().lower()
        normalized = transcript.strip().lower()
        if not title or not normalized:
            return None

        for app_name, macros in self._config.app_macros.items():
            if str(app_name).strip().lower() not in title:
                continue
            if not isinstance(macros, dict):
                continue
            for phrase, payload in macros.items():
                if normalized != str(phrase).strip().lower():
                    continue
                if not isinstance(payload, dict) or "action" not in payload:
                    continue
                return ActionRequest(
                    action=str(payload["action"]),
                    value=payload.get("value"),
                    source=f"macro:{app_name}",
                )
        return None

    def _build_overlay(self) -> DevOverlay | None:
        if not bool(self._runtime_settings.get("development_mode", False)):
            return None
        return DevOverlay(
            tracking_window=bool(self._runtime_settings.get("show_tracking_window", True)),
            sandbox_window=bool(self._runtime_settings.get("show_sandbox_window", True)),
            on_close=self.stop,
        )

    def _observe_action(self, request: ActionRequest) -> None:
        if self._overlay is None:
            return
        self._overlay.record_action(request.action, request.value)

    def _observe_frame(
        self,
        frame: object,
        hand_landmarks: list[dict[int, tuple[int, int]]] | None,
        face_landmarks: list[tuple[int, int]] | None,
        event_name: str | None,
        calibration_status,
        head_pose,
    ) -> None:
        if self._overlay is None:
            return
        self._overlay.update_tracking(
            frame=frame,
            hand_landmarks=hand_landmarks,
            face_landmarks=face_landmarks,
            event_name=event_name,
            calibration_step=calibration_status.current_step,
            calibration_completed=calibration_status.completed_steps,
            calibration_active=calibration_status.active,
            head_pose=head_pose,
        )

    def _log_latency(self, route: str, started_at: float, stage_started_at: float | None = None) -> None:
        total_ms = (time.monotonic() - started_at) * 1000.0
        if stage_started_at is None:
            logging.info("Route '%s' completed in %.1f ms", route, total_ms)
            return
        stage_ms = (time.monotonic() - stage_started_at) * 1000.0
        logging.info("Route '%s' completed in %.1f ms (stage %.1f ms)", route, total_ms, stage_ms)

    def _should_acknowledge(self, source: str, transcript: str) -> bool:
        if transcript and self._WAKE_PHRASE_RE.match(transcript):
            return True
        return source == "wake_word" and not transcript

    def _dispatch_acknowledgement(self, source: str) -> None:
        self._controller.dispatch(
            ActionRequest(
                action="speak",
                value="Yes? How can I help you?",
                source=f"{source}:ack",
            )
        )

    def _signal_ready(self) -> None:
        if not self._ready_file:
            return
        try:
            Path(self._ready_file).write_text("ready\n", encoding="utf-8")
        except Exception:
            logging.exception("Failed to write OSCAR ready file: %s", self._ready_file)

    def _clear_ready_signal(self) -> None:
        if not self._ready_file:
            return
        try:
            Path(self._ready_file).unlink(missing_ok=True)
        except Exception:
            logging.exception("Failed to remove OSCAR ready file: %s", self._ready_file)

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
        logging.info("  local_model_exists: %s", brain_diag["model_path_exists"])
        logging.info("  local_model_ready: %s", brain_diag["model_files_present"])
        logging.info("  inference_backend: %s", brain_diag["backend_name"])
        logging.info("  openvino_device: %s", brain_diag["openvino_device"])
        logging.info("  voice_ready: %s", self._voice.is_ready())


def load_runtime_config() -> RuntimeConfig:
    settings = json.loads(Path("config/settings.yaml").read_text(encoding="utf-8"))
    gestures = json.loads(Path("config/gestures.json").read_text(encoding="utf-8"))
    app_macros = json.loads(Path("config/app_macros.json").read_text(encoding="utf-8"))
    runtime = settings.get("runtime", {})
    env_overrides = {
        "development_mode": os.environ.get("OSCAR_DEVELOPMENT_MODE"),
        "show_tracking_window": os.environ.get("OSCAR_SHOW_TRACKING_WINDOW"),
        "show_sandbox_window": os.environ.get("OSCAR_SHOW_SANDBOX_WINDOW"),
        "sandbox_actions_only": os.environ.get("OSCAR_SANDBOX_ACTIONS_ONLY"),
    }
    for key, raw_value in env_overrides.items():
        if raw_value is None:
            continue
        runtime[key] = raw_value.strip().lower() in {"1", "true", "yes", "on"}
    settings["runtime"] = runtime
    model_profile_override = os.environ.get("OSCAR_MODEL_PROFILE")
    if model_profile_override:
        settings.setdefault("models", {})["active_profile"] = model_profile_override.strip()
    return RuntimeConfig(settings=settings, gestures=gestures, app_macros=app_macros)


def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> int:
    config = load_runtime_config()
    configure_threading(int(config.settings["hardware"]["cpu_threads"]))
    configure_logging(config.settings["runtime"]["log_level"])
    active_profile = resolve_active_model_settings(config.settings)

    if str(active_profile.get("inference_backend", "")).strip().lower() != "ollama":
        model_path = active_profile.get("model_path") or active_profile.get("gemma_model_path", "models/gemma")
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

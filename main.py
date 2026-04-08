from __future__ import annotations

import json
import logging
import signal
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from actions.controller import ActionController, ActionRequest
from actions.window_manager import WindowManager
from core.brain import Brain, BrainOutputError
from core.ears import Ears, TranscriptEvent
from core.eyes import Eyes, GestureEvent
from core.memory import Memory
from core.router import DirectCommandRouter
from core.vision import Vision
from core.voice import Voice


class OrchestratorState(str, Enum):
    DORMANT = "dormant"
    ROUTING = "routing"
    REASONING = "reasoning"
    CLEANUP = "cleanup"
    SHUTDOWN = "shutdown"


@dataclass(slots=True)
class RuntimeConfig:
    settings: dict[str, Any]
    gestures: dict[str, Any]
    app_macros: dict[str, Any]


class OSCAR:
    """Offline orchestrator implementing the PRD control loop."""

    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config
        self._state = OrchestratorState.DORMANT
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._controller = ActionController()
        self._window_manager = WindowManager()
        self._router = DirectCommandRouter()
        self._brain = Brain(prompt_template_path=Path("config") / "prompt_template.txt")
        self._memory = Memory(database_path=config.settings["screenpipe"]["database_path"])
        self._vision = Vision()
        self._voice = Voice(
            model_path=config.settings["voice"]["model_path"],
            piper_executable=config.settings["voice"]["piper_executable"],
        )
        self._ears = Ears(
            on_transcript=self._handle_transcript_event,
            wake_word_model_path=config.settings["models"]["wake_word_model_path"],
            whisper_model_path=config.settings["models"]["whisper_model_path"],
        )
        self._eyes = Eyes(
            settings={
                **config.settings["runtime"],
                **config.settings["gestures"],
            },
            on_gesture=self._handle_gesture,
        )
        self._controller.register_handler("speak", self._voice.speak)

    def start(self) -> None:
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
            current_state = self._state

        if current_state != OrchestratorState.DORMANT:
            logging.debug("Ignoring gesture '%s' while state is %s.", event.name, current_state)
            return

        request = ActionRequest(action=event.action, value=event.value, source="gesture")
        self._controller.dispatch(request)
        logging.info("Gesture routed: %s -> %s", event.name, request.action)

    def _handle_transcript_event(self, event: TranscriptEvent) -> None:
        self.handle_wake_word(event.transcript)

    def handle_wake_word(self, transcript: str) -> None:
        with self._state_lock:
            self._state = OrchestratorState.ROUTING
        self._eyes.pause()
        logging.info("Wake word path requested with transcript: %s", transcript)

        try:
            if self._dispatch_direct_route(transcript):
                return
            if self._dispatch_macro_route(transcript):
                return
            self._dispatch_reasoned_route(transcript)
        finally:
            self._cleanup_after_reasoning()

    def _dispatch_direct_route(self, transcript: str) -> bool:
        direct_result = self._router.route(transcript)
        if direct_result is None:
            return False
        self._controller.dispatch(direct_result.request)
        logging.info("Direct route matched %s for transcript '%s'.", direct_result.matched_rule, transcript)
        return True

    def _dispatch_macro_route(self, transcript: str) -> bool:
        macro_request = self.resolve_macro_request(transcript)
        if macro_request is None:
            return False
        self._controller.dispatch(macro_request)
        logging.info("Macro routed from transcript '%s': %s", transcript, macro_request.action)
        return True

    def _dispatch_reasoned_route(self, transcript: str) -> None:
        with self._state_lock:
            self._state = OrchestratorState.REASONING

        route_decision = self._brain.decide_route(transcript)
        if route_decision.route == "memory":
            context = self._memory.query_recent_text(route_decision.context_prompt)
        elif route_decision.route == "vision":
            context = self._vision.answer_visual_question(route_decision.context_prompt)
        else:
            context = ""

        try:
            plan = self._brain.plan(transcript=transcript, context=context)
        except BrainOutputError:
            logging.exception("Brain output sanitization failed.")
            self._controller.dispatch(ActionRequest(action="speak", value="I could not safely execute that request.", source="brain"))
            return

        self._controller.dispatch(ActionRequest(action=plan.action, value=plan.value, source="brain"))

    def _cleanup_after_reasoning(self) -> None:
        with self._state_lock:
            self._state = OrchestratorState.CLEANUP
        self._eyes.resume()
        with self._state_lock:
            self._state = OrchestratorState.DORMANT

    def resolve_macro_request(self, transcript: str) -> ActionRequest | None:
        normalized_transcript = transcript.strip().lower()
        if not normalized_transcript:
            return None

        active_window = self.active_window_title()
        if not active_window:
            return None

        app_name, macros = self._match_app_macros(active_window)
        if not macros:
            return None

        for command_name, payload in macros.items():
            spoken_command = command_name.replace("_", " ").strip().lower()
            if spoken_command == normalized_transcript:
                return ActionRequest(
                    action=str(payload["action"]),
                    value=payload.get("value"),
                    source=f"macro:{app_name}",
                )
        return None

    def _match_app_macros(self, active_window: str) -> tuple[str | None, dict[str, Any] | None]:
        normalized_window = active_window.lower()
        for app_name, macros in self._config.app_macros.items():
            if app_name.lower() in normalized_window:
                return app_name, macros
        return None, None

    def active_window_title(self) -> str | None:
        return self._window_manager.get_active_window_title()


def load_config(base_dir: Path) -> RuntimeConfig:
    config_dir = base_dir / "config"
    with (config_dir / "settings.yaml").open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    with (config_dir / "gestures.json").open("r", encoding="utf-8") as handle:
        gestures = json.load(handle)
    with (config_dir / "app_macros.json").open("r", encoding="utf-8") as handle:
        app_macros = json.load(handle)
    return RuntimeConfig(settings=settings, gestures=gestures, app_macros=app_macros)


def configure_logging(base_dir: Path, log_level: str) -> None:
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(logs_dir / "oscar.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    config = load_config(base_dir)
    configure_logging(base_dir, config.settings["runtime"]["log_level"])
    oscar = OSCAR(config)

    def _shutdown_handler(*_: Any) -> None:
        oscar.stop()

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)
    oscar.run()


if __name__ == "__main__":
    main()

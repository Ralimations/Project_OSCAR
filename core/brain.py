from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
VISUAL_QUERY_RE = re.compile(r"\b(where|look|see|visible|screen|button|icon|window|submit|click on)\b", re.IGNORECASE)
MEMORY_QUERY_RE = re.compile(r"\b(what was|earlier|before|previous|email|address|phone|history|recent|last)\b", re.IGNORECASE)


@dataclass(slots=True)
class BrainResponse:
    action: str
    value: Any


@dataclass(frozen=True, slots=True)
class RouteDecision:
    route: str
    context_prompt: str


class BrainOutputError(ValueError):
    """Raised when the LLM output does not match OSCAR's action schema."""


class Brain:
    """Offline planning shim with strict JSON enforcement and heuristic routing."""

    def __init__(
        self,
        prompt_template_path: str | Path,
        model_infer: Callable[[str], str] | None = None,
    ) -> None:
        self._prompt_template = Path(prompt_template_path).read_text(encoding="utf-8")
        self._model_infer = model_infer

    def decide_route(self, transcript: str) -> RouteDecision:
        normalized = transcript.strip()
        if VISUAL_QUERY_RE.search(normalized):
            return RouteDecision(route="vision", context_prompt=normalized)
        if MEMORY_QUERY_RE.search(normalized):
            return RouteDecision(route="memory", context_prompt=normalized)
        return RouteDecision(route="brain", context_prompt=normalized)

    def plan(self, transcript: str, context: str = "") -> BrainResponse:
        if self._model_infer is not None:
            prompt = self._build_prompt(transcript=transcript, context=context)
            return sanitize_llm_output(self._model_infer(prompt))

        fallback = self._fallback_plan(transcript=transcript, context=context)
        return sanitize_llm_output(json.dumps(fallback))

    def _build_prompt(self, transcript: str, context: str) -> str:
        return "\n".join(
            [
                self._prompt_template.strip(),
                f"Transcript: {transcript}",
                f"Context: {context}",
            ]
        )

    def _fallback_plan(self, transcript: str, context: str) -> dict[str, Any]:
        normalized = transcript.strip().lower()
        if normalized.startswith("say "):
            return {"action": "speak", "value": transcript[4:].strip()}
        if "open " in normalized and " and " not in normalized:
            app_name = normalized.replace("open ", "", 1).strip()
            return {"action": "open_app", "value": app_name}
        if context:
            return {"action": "speak", "value": context}
        return {"action": "speak", "value": "I cannot do that offline."}


def sanitize_llm_output(raw_text: str) -> BrainResponse:
    """Extract and validate the first JSON object from model output."""
    match = JSON_OBJECT_RE.search(raw_text)
    if not match:
        raise BrainOutputError("No JSON object found in model output.")

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise BrainOutputError("Model output contains invalid JSON.") from exc

    if not isinstance(payload, dict):
        raise BrainOutputError("Model output must be a JSON object.")

    action = payload.get("action")
    value = payload.get("value")
    if not isinstance(action, str) or not action.strip():
        raise BrainOutputError("JSON output must include a non-empty 'action' string.")

    if isinstance(value, (dict, list)) and action not in {"hotkey", "sequence"}:
        raise BrainOutputError("Only 'hotkey' and 'sequence' actions may carry list or object values.")

    return BrainResponse(action=action.strip(), value=value)

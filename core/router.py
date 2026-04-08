from __future__ import annotations

import re
from dataclasses import dataclass

from actions.controller import ActionRequest


@dataclass(frozen=True, slots=True)
class DirectRouteResult:
    request: ActionRequest
    matched_rule: str


class DirectCommandRouter:
    """Deterministic transcript router for cheap offline commands before LLM fallback."""

    _OPEN_APP_RE = re.compile(r"^open\s+(?P<app>[a-z0-9 ._-]+?)(?:\s+and\s+(?P<task>.+))?$", re.IGNORECASE)

    def route(self, transcript: str) -> DirectRouteResult | None:
        normalized = transcript.strip()
        if not normalized:
            return None

        open_app_match = self._OPEN_APP_RE.match(normalized)
        if open_app_match:
            app = open_app_match.group("app").strip()
            task = (open_app_match.group("task") or "").strip()
            if task:
                return DirectRouteResult(
                    request=self._build_open_and_search_request(app=app, task=task),
                    matched_rule="open_app_with_task",
                )
            return DirectRouteResult(
                request=ActionRequest(action="open_app", value=self._normalize_app_command(app), source="direct:open_app"),
                matched_rule="open_app",
            )

        return None

    def _build_open_and_search_request(self, app: str, task: str) -> ActionRequest:
        normalized_app = self._normalize_app_command(app)
        search_text = self._normalize_task_text(app=app, task=task)
        return ActionRequest(
            action="sequence",
            value=[
                {"action": "open_app", "value": normalized_app},
                {"action": "wait", "value": 1.0},
                {"action": "hotkey", "value": ["ctrl", "l"]},
                {"action": "write_text", "value": search_text},
                {"action": "press_key", "value": "enter"},
            ],
            source="direct:open_app_with_task",
        )

    def _normalize_app_command(self, app: str) -> str:
        lowered = app.strip().lower()
        aliases = {
            "brave": "brave",
            "bandlab": "bandlab",
            "notepad": "notepad",
        }
        return aliases.get(lowered, app.strip())

    def _normalize_task_text(self, app: str, task: str) -> str:
        lowered_app = app.strip().lower()
        if lowered_app == "brave" and task.lower().startswith("watch "):
            return f"youtube {task}"
        return task

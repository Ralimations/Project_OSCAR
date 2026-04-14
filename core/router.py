from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from actions.controller import ActionRequest


@dataclass(frozen=True, slots=True)
class DirectRouteResult:
    request: ActionRequest
    matched_rule: str


class DirectCommandRouter:
    """Deterministic transcript router for cheap offline commands before LLM fallback."""

    _OPEN_APP_RE = re.compile(r"^open\s+(?P<app>[a-z0-9 ._-]+?)(?:\s+and\s+(?P<task>.+))?$", re.IGNORECASE)
    _OPEN_IN_APP_RE = re.compile(r"^open\s+(?P<task>.+?)\s+in\s+(?P<app>[a-z0-9 ._-]+)$", re.IGNORECASE)
    _SCROLL_RE = re.compile(r"^(?:scroll|move)\s+(?P<direction>up|down)(?:\s+(?P<count>\d+))?$", re.IGNORECASE)
    _PRESS_KEY_RE = re.compile(r"^(?:press|hit)\s+(?P<key>[a-z0-9 _-]+)$", re.IGNORECASE)
    _WRITE_TEXT_RE = re.compile(r"^(?:type|write)\s+(?P<text>.+)$", re.IGNORECASE)
    _SAY_RE = re.compile(r"^(?:say|speak)\s+(?P<text>.+)$", re.IGNORECASE)

    def route(self, transcript: str) -> DirectRouteResult | None:
        normalized = transcript.strip()
        if not normalized:
            return None

        open_in_app_match = self._OPEN_IN_APP_RE.match(normalized)
        if open_in_app_match:
            task = open_in_app_match.group("task").strip()
            app = open_in_app_match.group("app").strip()
            return DirectRouteResult(
                request=self._build_open_and_search_request(app=app, task=task),
                matched_rule="open_task_in_app",
            )

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

        if normalized.lower() in {"click", "left click"}:
            return DirectRouteResult(
                request=ActionRequest(action="click", value=None, source="direct:click"),
                matched_rule="click",
            )

        if normalized.lower() in {"focus address bar", "address bar", "go to address bar"}:
            return DirectRouteResult(
                request=ActionRequest(action="hotkey", value=["ctrl", "l"], source="direct:focus_address_bar"),
                matched_rule="focus_address_bar",
            )

        hotkey_routes = {
            "go back": ["alt", "left"],
            "back": ["alt", "left"],
            "go forward": ["alt", "right"],
            "forward": ["alt", "right"],
            "refresh": ["ctrl", "r"],
            "new tab": ["ctrl", "t"],
            "add new tab": ["ctrl", "t"],
            "open new tab": ["ctrl", "t"],
            "close tab": ["ctrl", "w"],
            "delete tab": ["ctrl", "w"],
            "remove tab": ["ctrl", "w"],
            "next tab": ["ctrl", "tab"],
            "previous tab": ["ctrl", "shift", "tab"],
            "switch tab": ["ctrl", "tab"],
            "zoom in": ["ctrl", "+"],
            "zoom out": ["ctrl", "-"],
        }
        route = hotkey_routes.get(normalized.lower())
        if route is not None:
            return DirectRouteResult(
                request=ActionRequest(action="hotkey", value=route, source="direct:hotkey"),
                matched_rule="hotkey",
            )

        scroll_match = self._SCROLL_RE.match(normalized)
        if scroll_match:
            direction = scroll_match.group("direction").lower()
            count = int(scroll_match.group("count") or 1)
            amount = 600 * count
            if direction == "down":
                amount = -amount
            return DirectRouteResult(
                request=ActionRequest(action="scroll", value=amount, source="direct:scroll"),
                matched_rule="scroll",
            )

        key_match = self._PRESS_KEY_RE.match(normalized)
        if key_match:
            key = self._normalize_key_name(key_match.group("key"))
            return DirectRouteResult(
                request=ActionRequest(action="press_key", value=key, source="direct:press_key"),
                matched_rule="press_key",
            )

        write_match = self._WRITE_TEXT_RE.match(normalized)
        if write_match:
            return DirectRouteResult(
                request=ActionRequest(action="write_text", value=write_match.group("text").strip(), source="direct:write_text"),
                matched_rule="write_text",
            )

        say_match = self._SAY_RE.match(normalized)
        if say_match:
            return DirectRouteResult(
                request=ActionRequest(action="speak", value=say_match.group("text").strip(), source="direct:speak"),
                matched_rule="speak",
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

    def _normalize_key_name(self, key: str) -> str:
        lowered = key.strip().lower()
        aliases = {
            "enter": "enter",
            "return": "enter",
            "escape": "esc",
            "esc": "esc",
            "space": "space",
            "spacebar": "space",
            "tab": "tab",
            "backspace": "backspace",
            "delete": "delete",
            "up": "up",
            "down": "down",
            "left": "left",
            "right": "right",
        }
        return aliases.get(lowered, lowered)


class TranscriptIntent(str, Enum):
    DIRECT = "direct"
    MEMORY = "memory"
    VISION = "vision"
    PLANNER = "planner"


def classify_transcript_intent(transcript: str) -> TranscriptIntent:
    normalized = transcript.strip().lower()
    if not normalized:
        return TranscriptIntent.PLANNER

    memory_markers = (
        "what was",
        "what did i",
        "earlier",
        "before",
        "history",
        "on my screen",
        "email address",
        "remember",
        "last time",
    )
    if any(marker in normalized for marker in memory_markers):
        return TranscriptIntent.MEMORY

    vision_markers = (
        "what is this",
        "what's this",
        "where is",
        "find the",
        "which button",
        "what do you see",
        "on the screen",
        "look at",
    )
    if any(marker in normalized for marker in vision_markers):
        return TranscriptIntent.VISION

    return TranscriptIntent.PLANNER

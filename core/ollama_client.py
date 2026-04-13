from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any
from urllib import request


class OllamaClient:
    """Minimal Ollama HTTP client for OSCAR's local text and vision backends."""

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434", timeout_seconds: int = 120) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    def list_models(self) -> list[str]:
        payload = self._post_json("/api/tags", {})
        models = payload.get("models") or []
        return [str(model.get("name")) for model in models if model.get("name")]

    def infer_action(self, prompt: str, screenshot_path: str | None, _audio_samples: Any, _sample_rate: int) -> str:
        message: dict[str, Any] = {"role": "user", "content": prompt}
        if screenshot_path:
            message["images"] = [self._encode_image(screenshot_path)]
        payload = {
            "model": self._model,
            "messages": [message],
            "stream": False,
            "options": {
                "temperature": 0,
            },
        }
        response = self._post_json("/api/chat", payload)
        return self._extract_message_content(response)

    def answer_visual_question(self, prompt: str, screenshot_path: str) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [self._encode_image(screenshot_path)],
                }
            ],
            "stream": False,
        }
        response = self._post_json("/api/chat", payload)
        return self._extract_message_content(response)

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self._base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self._timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def _encode_image(self, image_path: str) -> str:
        return base64.b64encode(Path(image_path).read_bytes()).decode("ascii")

    def _extract_message_content(self, response: dict[str, Any]) -> str:
        message = response.get("message")
        if not isinstance(message, dict):
            raise ValueError("Ollama response is missing a message payload.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Ollama response did not include text content.")
        return content.strip()

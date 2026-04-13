from __future__ import annotations

import gc
import importlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(slots=True)
class BrainResponse:
    action: str
    value: Any


class BrainOutputError(ValueError):
    """Raised when the model output does not match OSCAR's action schema."""


class Brain:
    """Gemma 4 OpenVINO wrapper with strict JSON action routing."""

    def __init__(
        self,
        prompt_template_path: str | Path,
        model_path: str | Path | None = None,
        device: str = "AUTO",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        infer_backend: Callable[[str, str | None, Sequence[float] | None, int], str] | None = None,
        backend_name: str | None = None,
    ) -> None:
        self._prompt_template = Path(prompt_template_path).read_text(encoding="utf-8")
        self._model_path = Path(model_path).expanduser().resolve() if model_path else None
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._infer_backend = infer_backend
        self._backend_name = backend_name or ("custom" if infer_backend is not None else "fallback")
        self._lazy_backend = None

    def infer_action(
        self,
        audio_samples: Sequence[float] | None,
        sample_rate: int,
        screenshot_path: str | None,
        ocr_history: str,
    ) -> BrainResponse:
        prompt = self._build_prompt(ocr_history=ocr_history)
        raw_output = None
        if self._infer_backend is not None:
            raw_output = self._infer_backend(prompt, screenshot_path, audio_samples, sample_rate)
        else:
            backend = self._ensure_backend()
            if backend is not None:
                try:
                    raw_output = backend(prompt, screenshot_path, audio_samples, sample_rate)
                except Exception:
                    logging.exception("Gemma backend failed; falling back to deterministic controller.")
        if raw_output is None:
            raw_output = json.dumps(self._fallback_action(ocr_history))
        return sanitize_llm_output(raw_output)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "backend_name": self._backend_name,
            "model_path": str(self._model_path) if self._model_path else None,
            "model_path_exists": self._model_path.exists() if self._model_path else False,
            "openvino_device": self._device,
        }

    def flush_context(self) -> None:
        gc.collect()

    def _build_prompt(self, ocr_history: str) -> str:
        return "\n".join(
            [
                self._prompt_template.strip(),
                "Return one JSON object only.",
                "Actions: click, scroll, press_key, hotkey, open_app, write_text, wait, sequence.",
                f"OCR history:\n{ocr_history or 'No OCR history available.'}",
            ]
        )

    def _fallback_action(self, ocr_history: str) -> dict[str, Any]:
        lowered = ocr_history.lower()
        if "address bar" in lowered:
            return {"action": "hotkey", "value": ["ctrl", "l"]}
        if "submit" in lowered:
            return {"action": "click", "value": None}
        return {"action": "wait", "value": 0.1}

    def _ensure_backend(self):
        if self._lazy_backend is not None:
            return self._lazy_backend
        loader = getattr(self, "_load_openvino_backend", None)
        if loader is None:
            loader = getattr(self, "_load_gguf_backend", None)
        self._lazy_backend = loader() if loader is not None else None
        return self._lazy_backend

    def _load_gguf_backend(self):
        if self._model_path is None or not self._model_path.exists():
            logging.warning("Gemma model path is missing: %s", self._model_path or "<unset>")
            return None
        try:
            from llama_cpp import Llama
        except ImportError:
            logging.warning("llama-cpp-python is unavailable.")
            return None

        try:
            model = Llama(
                model_path=str(self._model_path),
                n_ctx=4096,
                n_threads=4,
                verbose=False
            )
            self._backend_name = "Llama.cpp"
        except Exception:
            logging.exception("Failed loading GGUF model from %s", self._model_path)
            return None

        def _infer(prompt: str, screenshot_path: str | None, audio_samples: Sequence[float] | None, sample_rate: int) -> str:
            # GGUF CPU fallback mandates string text-only paths natively.
            # Rich multimodal bytes (vision/audio) are bypassed to maintain stable CPU execution.
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]

            try:
                # Issue completion command via strictly AVX2 bounds
                response = model.create_chat_completion(
                    messages=messages,
                    max_tokens=self._max_new_tokens,
                    temperature=self._temperature if self._temperature > 0 else 0.1,
                    top_p=0.95,
                    top_k=64
                )
                
                output_text = response["choices"][0]["message"]["content"]

                if not output_text or not output_text.strip():
                    raise BrainOutputError("Gemma backend returned empty text.")
                return output_text.strip()

            except Exception:
                logging.exception("Llama.cpp native generation failed.")
                raise BrainOutputError("Gemma text generation failed.")

        return _infer


def sanitize_llm_output(raw_text: str) -> BrainResponse:
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

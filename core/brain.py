from __future__ import annotations

import gc
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
    """Local planner backend with strict JSON action routing."""

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
        transcript: str | None = None,
    ) -> BrainResponse:
        prompt = self._build_prompt(ocr_history=ocr_history, transcript=transcript)
        raw_output = None
        if self._infer_backend is not None:
            try:
                raw_output = self._infer_backend(prompt, screenshot_path, audio_samples, sample_rate)
            except Exception:
                logging.exception("Configured inference backend failed; falling back to deterministic controller.")
        else:
            backend = self._ensure_backend()
            if backend is not None:
                try:
                    raw_output = backend(prompt, screenshot_path, audio_samples, sample_rate)
                except Exception:
                    logging.exception("Local model backend failed; falling back to deterministic controller.")
        if raw_output is None:
            raw_output = json.dumps(self._fallback_action(ocr_history=ocr_history, transcript=transcript))
        return sanitize_llm_output(raw_output)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "backend_name": self._backend_name,
            "model_path": str(self._model_path) if self._model_path else None,
            "model_path_exists": self._model_path.exists() if self._model_path else False,
            "model_files_present": self._has_transformers_weights(),
            "openvino_device": self._device,
        }

    def flush_context(self) -> None:
        gc.collect()

    def _build_prompt(self, ocr_history: str, transcript: str | None) -> str:
        return "\n".join(
            [
                self._prompt_template.strip(),
                "Return one JSON object only.",
                "Actions: click, scroll, press_key, hotkey, open_app, write_text, wait, sequence.",
                f"Transcript:\n{transcript or 'No transcript available.'}",
                f"OCR history:\n{ocr_history or 'No OCR history available.'}",
            ]
        )

    def _fallback_action(self, ocr_history: str, transcript: str | None) -> dict[str, Any]:
        normalized = (transcript or "").strip().lower()
        if normalized:
            if normalized.startswith("open "):
                return {"action": "open_app", "value": normalized.removeprefix("open ").strip()}
            if normalized.startswith("type ") or normalized.startswith("write "):
                _, _, text = normalized.partition(" ")
                return {"action": "write_text", "value": text.strip()}
            if normalized.startswith("say ") or normalized.startswith("speak "):
                _, _, text = normalized.partition(" ")
                return {"action": "speak", "value": text.strip()}
            if "address bar" in normalized:
                return {"action": "hotkey", "value": ["ctrl", "l"]}
            if "scroll down" in normalized:
                return {"action": "scroll", "value": -600}
            if "scroll up" in normalized:
                return {"action": "scroll", "value": 600}
            if "press enter" in normalized:
                return {"action": "press_key", "value": "enter"}

        lowered = ocr_history.lower()
        if "address bar" in lowered:
            return {"action": "hotkey", "value": ["ctrl", "l"]}
        if "submit" in lowered:
            return {"action": "click", "value": None}
        return {"action": "wait", "value": 0.1}

    def _ensure_backend(self):
        if self._lazy_backend is not None:
            return self._lazy_backend
        loader = getattr(self, "_load_transformers_backend", None)
        if loader is not None:
            backend = loader()
            if backend is not None:
                self._lazy_backend = backend
                return self._lazy_backend
        loader = getattr(self, "_load_openvino_backend", None)
        if loader is None:
            loader = getattr(self, "_load_gguf_backend", None)
        self._lazy_backend = loader() if loader is not None else None
        return self._lazy_backend

    def _load_transformers_backend(self):
        if self._model_path is None or not self._model_path.exists():
            return None
        config_path = self._model_path / "config.json"
        if not config_path.exists():
            return None
        if not self._has_transformers_weights():
            logging.warning("Transformers model weights are missing in %s", self._model_path)
            return None

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            logging.warning("transformers/torch backend is unavailable.")
            return None

        try:
            tokenizer = AutoTokenizer.from_pretrained(str(self._model_path), local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(self._model_path),
                local_files_only=True,
            )
            model.eval()
            self._backend_name = "Transformers"
        except Exception:
            logging.exception("Failed loading Transformers model from %s", self._model_path)
            return None

        def _infer(prompt: str, screenshot_path: str | None, audio_samples: Sequence[float] | None, sample_rate: int) -> str:
            del screenshot_path, audio_samples, sample_rate
            try:
                inputs = tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self._max_new_tokens,
                        do_sample=self._temperature > 0,
                        temperature=max(self._temperature, 0.1),
                        pad_token_id=tokenizer.eos_token_id,
                    )
                prompt_tokens = inputs["input_ids"].shape[1]
                generated = outputs[0][prompt_tokens:]
                text = tokenizer.decode(generated, skip_special_tokens=True).strip()
                if not text:
                    raise BrainOutputError("Transformers backend returned empty text.")
                return text
            except Exception:
                logging.exception("Transformers generation failed.")
                raise BrainOutputError("Transformers text generation failed.")

        return _infer

    def _has_transformers_weights(self) -> bool:
        if self._model_path is None or not self._model_path.exists():
            return False
        weight_patterns = (
            "model.safetensors",
            "*.safetensors",
            "pytorch_model.bin",
            "pytorch_model-*.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
        for pattern in weight_patterns:
            if any(self._model_path.glob(pattern)):
                return True
        return False

    def _load_gguf_backend(self):
        if self._model_path is None or not self._model_path.exists():
            logging.warning("Local model path is missing: %s", self._model_path or "<unset>")
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
                    raise BrainOutputError("Local model backend returned empty text.")
                return output_text.strip()

            except Exception:
                logging.exception("Llama.cpp native generation failed.")
                raise BrainOutputError("Local model text generation failed.")

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

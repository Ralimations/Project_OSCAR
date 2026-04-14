from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve


OPENWAKEWORD_RELEASE = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
MEDIAPIPE_HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
MEDIAPIPE_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
OPENWAKEWORD_ONNX_FILES = [
    ("alexa_v0.1.onnx", "alexa_v0.1.onnx"),
    ("melspectrogram.onnx", "features/melspectrogram.onnx"),
    ("embedding_model.onnx", "features/embedding_model.onnx"),
]
SMOLLM2_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
SMOLLM2_TARGET_DIR = "smollm2-360m-instruct-hf"
SMOLLM2_ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
]


def remove_legacy_models(models_dir: Path) -> None:
    for directory in (
        models_dir / "llama",
        models_dir / "moondream2",
        models_dir / "faster-whisper",
        models_dir / "faster-whisper-tiny",
    ):
        if directory.exists():
            print(f"Removing legacy model directory: {directory}")
            shutil.rmtree(directory, ignore_errors=True)

    for path in models_dir.rglob("*"):
        if not path.is_file():
            continue
        lowered = path.name.lower()
        if path.suffix.lower() == ".gguf":
            path.unlink(missing_ok=True)
        elif "moondream" in lowered and path.suffix.lower() == ".onnx":
            path.unlink(missing_ok=True)
        elif "whisper" in lowered and path.suffix.lower() == ".bin":
            path.unlink(missing_ok=True)


def download_openwakeword_models(models_dir: Path) -> None:
    wakeword_dir = models_dir / "openwakeword"
    wakeword_dir.mkdir(parents=True, exist_ok=True)
    for source_name, relative_target in OPENWAKEWORD_ONNX_FILES:
        target_path = wakeword_dir / relative_target
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists():
            continue
        url = f"{OPENWAKEWORD_RELEASE}/{source_name}"
        print(f"Downloading {url} -> {target_path}")
        urlretrieve(url, target_path)


def download_mediapipe_asset(models_dir: Path, filename: str, source_url: str) -> None:
    mediapipe_dir = models_dir / "mediapipe"
    mediapipe_dir.mkdir(parents=True, exist_ok=True)
    target_path = mediapipe_dir / filename
    if target_path.exists():
        return
    print(f"Downloading {source_url} -> {target_path}")
    urlretrieve(source_url, target_path)


def _optimum_cli_exe() -> str:
    """Return the path to optimum-cli, preferring the Scripts/ sibling of the running interpreter."""
    scripts_dir = Path(sys.executable).parent / "Scripts"
    for candidate in (scripts_dir / "optimum-cli.exe", scripts_dir / "optimum-cli"):
        if candidate.exists():
            return str(candidate)
    return "optimum-cli"  # hope it's on PATH for system installs


def pull_ollama_model(model_name: str) -> None:
    print(f"Pulling Ollama model: {model_name}...")
    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
        print(f"Successfully pulled {model_name}.")
    except FileNotFoundError:
        print("\n[BLOCKED] Ollama is not installed or not in PATH! Please install Ollama first.\n")
    except subprocess.CalledProcessError as exc:
        print(f"\n[BLOCKED] Ollama pull failed for {model_name}: {exc}\n")

def download_small_local_model(models_dir: Path) -> None:
    target_dir = models_dir / SMOLLM2_TARGET_DIR
    if (target_dir / "model.safetensors").exists():
        print(f"Small local model already present: {target_dir}")
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("\n[BLOCKED] huggingface_hub is not available in the embedded runtime.\n")
        return

    print(f"Downloading local small model {SMOLLM2_MODEL_ID} -> {target_dir}")
    snapshot_download(
        repo_id=SMOLLM2_MODEL_ID,
        local_dir=str(target_dir),
        allow_patterns=SMOLLM2_ALLOW_PATTERNS,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OSCAR runtime models.")
    parser.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Skip Ollama pulls and only download repo-local assets.",
    )
    parser.add_argument(
        "--skip-small-local",
        action="store_true",
        help="Skip the bundled small local Transformers model download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    remove_legacy_models(models_dir)
    download_openwakeword_models(models_dir)
    download_mediapipe_asset(models_dir, "hand_landmarker.task", MEDIAPIPE_HAND_LANDMARKER_URL)
    download_mediapipe_asset(models_dir, "face_landmarker.task", MEDIAPIPE_FACE_LANDMARKER_URL)
    if not args.skip_small_local:
        download_small_local_model(models_dir)
    if not args.skip_ollama:
        pull_ollama_model("gemma4:e2b")
        pull_ollama_model("llama3.2:1b")
    print("Model bootstrap completed.")



if __name__ == "__main__":
    main()

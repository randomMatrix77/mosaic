"""Native inference boundary; no desktop Python extensions are used on Android."""

from __future__ import annotations

from typing import Protocol


class InferenceProvider(Protocol):
    def load_model(self, model_path: str) -> None: ...

    def generate(self, prompt: str) -> str: ...

    def unload_model(self) -> None: ...


class AndroidInferenceProvider:
    """Calls the app's JNI bridge through Chaquopy's supported Java interop."""

    def __init__(self) -> None:
        from java import jclass  # Chaquopy-only adapter import, kept out of core.py.

        self._native = jclass("dev.mosaic.poc.NativeLlama")

    @property
    def is_loaded(self) -> bool:
        return bool(self._native.isModelLoaded())

    def load_model(self, model_path: str) -> None:
        outcome = str(self._native.loadModel(model_path))
        if outcome.startswith("ERROR:"):
            raise RuntimeError(outcome)

    def generate(self, prompt: str) -> str:
        outcome = str(self._native.generate(prompt))
        if outcome.startswith("ERROR:"):
            raise RuntimeError(outcome)
        return outcome

    def generate_chat(self, messages: list[dict[str, str]]) -> str:
        """Ask JNI to apply Qwen's native chat template to persisted history."""
        import json

        if not messages:
            raise ValueError("At least one chat message is required.")
        outcome = str(self._native.generateChat(json.dumps(messages, ensure_ascii=False)))
        if outcome.startswith("ERROR:"):
            raise RuntimeError(outcome)
        return outcome

    def unload_model(self) -> None:
        self._native.unloadModel()

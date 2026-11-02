"""Native inference boundary; no desktop Python extensions are used on Android."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Protocol

from mosaic.events import GenerationCancelled


class InferenceProvider(Protocol):
    def load_model(self, model_path: str) -> None: ...

    def generate(self, prompt: str) -> str: ...

    def unload_model(self) -> None: ...


@dataclass(frozen=True)
class ModelPaths:
    qwen: str
    embedding: str
    projector: str
    qwen_projector: str = ""
    debug_images: str = ""


_model_paths: ModelPaths | None = None


def configure_model_paths(paths: ModelPaths) -> None:
    global _model_paths
    _model_paths = paths


def get_model_paths() -> ModelPaths:
    if _model_paths is None:
        raise RuntimeError("Android model paths have not been configured.")
    return _model_paths


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

    def generate_chat(self, messages: list[dict[str, Any]]) -> str:
        """Ask JNI to apply Qwen's native chat template to persisted history."""
        import json

        if not messages:
            raise ValueError("At least one chat message is required.")
        outcome = str(self._native.generateChat(json.dumps(messages, ensure_ascii=False)))
        if outcome.startswith("ERROR:"):
            raise RuntimeError(outcome)
        return outcome

    def generate_chat_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        """Use llama.cpp's model-provided tool template and native output parser."""
        if not messages:
            raise ValueError("At least one chat message is required.")
        request = {"messages": messages, "tools": tools}
        outcome = str(self._native.generateChatWithTools(json.dumps(request, ensure_ascii=False)))
        if outcome.startswith("ERROR:"):
            raise RuntimeError(outcome)
        try:
            parsed = json.loads(outcome)
        except json.JSONDecodeError as error:
            raise RuntimeError("Native tool-chat response was not JSON.") from error
        if not isinstance(parsed, dict):
            raise RuntimeError("Native tool-chat response was not an object.")
        return parsed

    def generate_chat_with_tools_stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                                        token_listener: Any) -> dict[str, Any]:
        """Run native tool chat with a Kotlin listener for decoded token pieces.

        The listener deliberately stays on the Java/Kotlin side while this
        synchronous Python-to-JNI call is active. Re-entering Python through a
        Chaquopy dynamic proxy during that call can deadlock its bridge.
        """
        if not messages:
            raise ValueError("At least one chat message is required.")
        request = {"messages": messages, "tools": tools}
        outcome = str(self._native.generateChatWithToolsStream(
            json.dumps(request, ensure_ascii=False), token_listener))
        if outcome.startswith("CANCELLED:"):
            raise GenerationCancelled(outcome.removeprefix("CANCELLED:").strip())
        if outcome.startswith("ERROR:"):
            raise RuntimeError(outcome)
        try:
            parsed = json.loads(outcome)
        except json.JSONDecodeError as error:
            raise RuntimeError("Native streaming tool-chat response was not JSON.") from error
        if not isinstance(parsed, dict):
            raise RuntimeError("Native streaming tool-chat response was not an object.")
        return parsed

    def prepare_generation(self) -> None:
        self._native.prepareGeneration()

    def load_generation_projector(self, projector_path: str) -> None:
        outcome = str(self._native.loadGenerationProjector(projector_path))
        if outcome.startswith("ERROR:"):
            raise RuntimeError(outcome)

    def generate_with_image(self, message: str, image_path: str) -> str:
        outcome = str(self._native.generateWithImage(message, image_path))
        if outcome.startswith("ERROR:"):
            raise RuntimeError(outcome)
        return outcome

    def unload_model(self) -> None:
        self._native.unloadModel()

    def cancel_generation(self) -> None:
        self._native.cancelGeneration()


class AndroidAgentGateway:
    """Switches the single native model slot for safe Qwen/tool orchestration."""

    def __init__(self, paths: ModelPaths, event_handler: Callable[..., Any] | None = None,
                 token_listener: Any | None = None) -> None:
        self._paths = paths
        self._native = AndroidInferenceProvider()
        self._event_handler = event_handler
        self._token_listener = token_listener

    def _emit(self, event_type: str, **payload: Any) -> None:
        if self._event_handler:
            self._event_handler(event_type, **payload)

    @staticmethod
    def _require_file(path: str, description: str) -> None:
        if not path or not Path(path).is_file():
            raise RuntimeError(f"Import the {description} before using this capability.")

    def _ensure_qwen(self) -> float:
        if self._native.is_loaded:
            self._emit("model_ready", model="Qwen")
            return 0.0
        self._require_file(self._paths.qwen, "Qwen GGUF")
        self._emit("model_loading", model="Qwen")
        started = perf_counter()
        self._native.load_model(self._paths.qwen)
        elapsed = (perf_counter() - started) * 1000
        self._emit("model_ready", model="Qwen", elapsed_ms=elapsed)
        return elapsed

    def _ensure_embedding(self) -> float:
        if bool(self._native._native.isEmbeddingModelLoaded()) and bool(self._native._native.isEmbeddingProjectorLoaded()):
            self._emit("model_ready", model="EmbeddingGemma")
            return 0.0
        self._require_file(self._paths.embedding, "EmbeddingGemma GGUF")
        self._require_file(self._paths.projector, "EmbeddingGemma multimodal projector")
        self._emit("model_loading", model="EmbeddingGemma")
        started = perf_counter()
        # The JNI loader owns one model slot, so this intentionally unloads Qwen.
        outcome = str(self._native._native.loadEmbeddingModel(self._paths.embedding))
        if outcome.startswith("ERROR:"):
            raise RuntimeError(outcome)
        outcome = str(self._native._native.loadEmbeddingProjector(self._paths.projector))
        if outcome.startswith("ERROR:"):
            raise RuntimeError(outcome)
        elapsed = (perf_counter() - started) * 1000
        self._emit("model_ready", model="EmbeddingGemma", elapsed_ms=elapsed)
        return elapsed

    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                 on_delta: Callable[[str], None] | None = None) -> Mapping[str, Any]:
        switch_ms = self._ensure_qwen()
        if on_delta:
            self._native.prepare_generation()
        self._emit("prompt_processing", model="Qwen")
        started = perf_counter()
        reply = (self._native.generate_chat_with_tools_stream(messages, tools, self._token_listener)
                 if on_delta else self._native.generate_chat_with_tools(messages, tools))
        reply["metrics"] = {
            "qwen_load_or_restore_ms": switch_ms,
            "qwen_generation_ms": (perf_counter() - started) * 1000,
        }
        return reply

    def generate_skill(self, messages: list[dict[str, Any]]) -> tuple[str, dict[str, float]]:
        """Run a distinct task prompt using the same single Qwen model slot."""
        if not messages:
            raise ValueError("Skill generation requires prompt messages.")
        switch_ms = self._ensure_qwen()
        started = perf_counter()
        reply = self._native.generate_chat(messages)
        return reply, {
            "qwen_load_or_restore_ms": switch_ms,
            "qwen_skill_generation_ms": (perf_counter() - started) * 1000,
        }

    def generate_with_image(self, message: str, image_path: str) -> tuple[str, dict[str, float]]:
        """Run one real image-plus-text Qwen request through the matching projector."""
        qwen_switch_ms = self._ensure_qwen()
        self._require_file(self._paths.qwen_projector, "matching Qwen multimodal projector")
        projector_load_ms = 0.0
        if not bool(self._native._native.isGenerationProjectorLoaded()):
            started = perf_counter()
            self._native.load_generation_projector(self._paths.qwen_projector)
            projector_load_ms = (perf_counter() - started) * 1000
        started = perf_counter()
        answer = self._native.generate_with_image(message, image_path)
        return answer, {
            "qwen_load_or_restore_ms": qwen_switch_ms,
            "qwen_projector_load_ms": projector_load_ms,
            "qwen_vision_generation_ms": (perf_counter() - started) * 1000,
        }

    def embed_query(self, text: str) -> tuple[list[float], dict[str, float]]:
        switch_ms = self._ensure_embedding()
        from mosaic.embeddings import EmbeddingGemmaProvider

        provider = EmbeddingGemmaProvider()
        vector = provider.embed_text(text)
        return vector, {
            "embedding_model_load_ms": switch_ms,
            "embedding_query_ms": provider.last_latency_ms or 0.0,
        }

    def embed_image(self, image_path: str) -> tuple[list[float], dict[str, float]]:
        """Create a real EmbeddingGemma vector for an app-private image."""
        switch_ms = self._ensure_embedding()
        from mosaic.embeddings import EmbeddingGemmaProvider

        provider = EmbeddingGemmaProvider()
        vector = provider.embed_image(image_path)
        return vector, {
            "embedding_model_load_ms": switch_ms,
            "embedding_image_ms": provider.last_latency_ms or 0.0,
        }

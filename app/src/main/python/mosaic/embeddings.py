"""Model-agnostic embedding API for Mosaic's embedded Python backend."""

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Sequence


class EmbeddingProvider(ABC):
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load this provider's embedding model."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Return a normalized embedding for text."""

    @abstractmethod
    def embed_image(self, image_path: str) -> list[float]:
        """Return a normalized embedding for image content."""

    def embed_audio(self, audio_path: str) -> list[float]:
        raise NotImplementedError("Audio embeddings are not implemented in this experiment.")

    def embed_video(self, video_path: str) -> list[float]:
        raise NotImplementedError("Video embeddings are not implemented in this experiment.")

    @abstractmethod
    def unload_model(self) -> None:
        """Release native model resources."""


class EmbeddingGemmaProvider(EmbeddingProvider):
    """EmbeddingGemma 2 adapter over the Android llama.cpp JNI boundary."""

    SENTENCE_SIMILARITY_TASK = "task: sentence similarity | query: "

    def __init__(self) -> None:
        from java import jclass

        self._native = jclass("dev.mosaic.poc.NativeLlama")
        self.last_latency_ms: float | None = None

    def load_model(self, model_path: str) -> None:
        status = str(self._native.loadEmbeddingModel(model_path))
        if status.startswith("ERROR:"):
            raise RuntimeError(status)

    def embed_text(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Cannot embed empty text.")
        formatted = self.SENTENCE_SIMILARITY_TASK + text
        started = perf_counter()
        vector = self._native.embedText(formatted)
        self.last_latency_ms = (perf_counter() - started) * 1000
        if vector is None:
            raise RuntimeError("llama.cpp did not return an EmbeddingGemma text embedding.")
        return [float(value) for value in vector]

    def embed_image(self, image_path: str) -> list[float]:
        if not self._native.isEmbeddingProjectorLoaded():
            raise RuntimeError("Import the official EmbeddingGemma 2 multimodal projector first.")
        started = perf_counter()
        vector = self._native.embedImage(image_path)
        self.last_latency_ms = (perf_counter() - started) * 1000
        if vector is None:
            raise RuntimeError("llama.cpp did not return an EmbeddingGemma image embedding.")
        return [float(value) for value in vector]

    def unload_model(self) -> None:
        self._native.unloadModel()


def cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("Cosine similarity requires equal, non-empty vectors.")
    return sum(a * b for a, b in zip(first, second))

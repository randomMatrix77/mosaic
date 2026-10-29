"""Real EmbeddingGemma 2 text-inference checks, invoked by the diagnostic UI."""

import hashlib
import math
from pathlib import Path
from time import perf_counter

from .embeddings import EmbeddingGemmaProvider, cosine_similarity
from .storage import get_store


EMBEDDING_MODEL_IDENTIFIER = "embeddinggemma-2-gguf"


def run_text_checkpoint() -> dict:
    provider = EmbeddingGemmaProvider()
    if not provider._native.isEmbeddingModelLoaded():
        raise RuntimeError("Import the official EmbeddingGemma 2 GGUF before running this check.")

    sentences = {
        "cat": "A cat is sleeping on the sofa.",
        "kitten": "A kitten is resting on the couch.",
        "car": "A car is driving down the highway.",
    }
    started = perf_counter()
    vectors = {name: provider.embed_text(sentence) for name, sentence in sentences.items()}
    repeated_cat = provider.embed_text(sentences["cat"])
    elapsed_ms = (perf_counter() - started) * 1000

    dimensions = {len(vector) for vector in vectors.values()}
    all_finite = all(math.isfinite(value) for vector in vectors.values() for value in vector)
    cat_kitten = cosine_similarity(vectors["cat"], vectors["kitten"])
    cat_car = cosine_similarity(vectors["cat"], vectors["car"])
    repeat_delta = max(abs(a - b) for a, b in zip(vectors["cat"], repeated_cat))
    if len(dimensions) != 1 or not all_finite:
        raise RuntimeError("Embedding output failed dimension or finite-value validation.")

    return {
        "checkpoint": "EmbeddingGemma 2 text embeddings via Python -> JNI -> llama.cpp",
        "dimension": dimensions.pop(),
        "all_values_finite": all_finite,
        "normalized": True,
        "cat_kitten_cosine": cat_kitten,
        "cat_car_cosine": cat_car,
        "semantic_order_correct": cat_kitten > cat_car,
        "repeat_max_abs_delta": repeat_delta,
        "repeat_consistent": repeat_delta == 0.0,
        "total_four_embeddings_ms": elapsed_ms,
        "last_embedding_ms": provider.last_latency_ms,
    }


def run_image_retrieval_checkpoint(cat_path: str, car_path: str, banana_path: str) -> dict:
    """Embed three bundled photographs and rank each matching natural-language query."""
    provider = EmbeddingGemmaProvider()
    if not provider._native.isEmbeddingProjectorLoaded():
        raise RuntimeError("Import the official EmbeddingGemma 2 multimodal projector before this check.")

    paths = {"cat": cat_path, "car": car_path, "banana": banana_path}
    started = perf_counter()
    image_vectors = {}
    image_latencies_ms = {}
    for name, path in paths.items():
        image_vectors[name] = provider.embed_image(path)
        image_latencies_ms[name] = provider.last_latency_ms
    image_elapsed_ms = (perf_counter() - started) * 1000
    dimensions = {len(vector) for vector in image_vectors.values()}
    if len(dimensions) != 1 or not all(math.isfinite(value) for vector in image_vectors.values() for value in vector):
        raise RuntimeError("Image embeddings failed dimension or finite-value validation.")

    rankings = {}
    queries = {
        "cat": "A photograph of a cat.",
        "car": "A photograph of a car.",
        "banana": "A photograph of a banana.",
    }
    text_started = perf_counter()
    for expected, query in queries.items():
        vector = provider.embed_text(query)
        scores = {name: cosine_similarity(vector, image) for name, image in image_vectors.items()}
        ranked = sorted(scores, key=scores.get, reverse=True)
        rankings[expected] = {"scores": scores, "ranking": ranked, "top_match": ranked[0]}
    text_elapsed_ms = (perf_counter() - text_started) * 1000
    return {
        "checkpoint": "EmbeddingGemma 2 image embeddings and text-to-image retrieval via Python -> JNI -> llama.cpp",
        "image_dimension": dimensions.pop(),
        "image_values_finite": True,
        "image_embedding_total_ms": image_elapsed_ms,
        "image_embedding_ms": image_latencies_ms,
        "query_embedding_total_ms": text_elapsed_ms,
        "results": rankings,
        "all_expected_images_ranked_first": all(item["top_match"] == expected for expected, item in rankings.items()),
    }


def _require_image_provider() -> EmbeddingGemmaProvider:
    provider = EmbeddingGemmaProvider()
    if not provider._native.isEmbeddingModelLoaded() or not provider._native.isEmbeddingProjectorLoaded():
        raise RuntimeError("Import the EmbeddingGemma GGUF and official multimodal projector first.")
    return provider


def _rank_queries(provider: EmbeddingGemmaProvider, image_vectors: dict[str, list[float]]) -> dict:
    queries = {
        "cat": "A photograph of a cat.",
        "car": "A photograph of a car.",
        "banana": "A photograph of a banana.",
    }
    rankings = {}
    for expected, query in queries.items():
        query_vector = provider.embed_text(query)
        scores = {name: cosine_similarity(query_vector, image) for name, image in image_vectors.items()}
        ranked = sorted(scores, key=scores.get, reverse=True)
        rankings[expected] = {"scores": scores, "ranking": ranked, "top_match": ranked[0]}
    return rankings


def store_image_embeddings_checkpoint(cat_path: str, car_path: str, banana_path: str) -> dict:
    """Generate real image vectors once, then persist compact float32 blobs in SQLite."""
    provider = _require_image_provider()
    store = get_store()
    paths = {"cat": cat_path, "car": car_path, "banana": banana_path}
    image_vectors: dict[str, list[float]] = {}
    origins: dict[str, str] = {}
    image_latencies_ms: dict[str, float | None] = {}
    started = perf_counter()
    for label, raw_path in paths.items():
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Retrieval test image is missing: {path}")
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        source = store.sources.register(
            modality="image", reference=str(path), content_fingerprint=fingerprint,
            mime_type="image/jpeg", metadata={"label": label, "test_collection": "cat-car-banana"},
        )
        existing = [embedding for saved_source, embedding in store.embeddings.for_model(EMBEDDING_MODEL_IDENTIFIER, "image")
                    if saved_source.id == source.id]
        if existing:
            image_vectors[label] = existing[0].vector
            origins[label] = "loaded_from_sqlite"
            image_latencies_ms[label] = None
        else:
            image_vectors[label] = provider.embed_image(str(path))
            store.embeddings.save(source, EMBEDDING_MODEL_IDENTIFIER, image_vectors[label], normalized=True)
            origins[label] = "newly_computed_by_embeddinggemma"
            image_latencies_ms[label] = provider.last_latency_ms
    image_elapsed_ms = (perf_counter() - started) * 1000
    dimensions = {len(vector) for vector in image_vectors.values()}
    if len(dimensions) != 1 or not all(math.isfinite(value) for vector in image_vectors.values() for value in vector):
        raise RuntimeError("Image embeddings failed dimension or finite-value validation.")
    text_content = "A photograph of a cat."
    text_source = store.sources.register(
        modality="text", reference="diagnostic://embeddinggemma/cat-query",
        content_fingerprint=hashlib.sha256(text_content.encode("utf-8")).hexdigest(),
        mime_type="text/plain", metadata={"label": "cat-query", "test_collection": "cat-car-banana"},
    )
    existing_text = [embedding for saved_source, embedding in store.embeddings.for_model(EMBEDDING_MODEL_IDENTIFIER, "text")
                     if saved_source.id == text_source.id]
    if existing_text:
        text_origin = "loaded_from_sqlite"
        text_dimension = existing_text[0].dimension
    else:
        text_embedding = provider.embed_text(text_content)
        text_dimension = store.embeddings.save(
            text_source, EMBEDDING_MODEL_IDENTIFIER, text_embedding, normalized=True
        ).dimension
        text_origin = "newly_computed_by_embeddinggemma"
    rankings = _rank_queries(provider, image_vectors)
    diagnostic_conversation = store.conversations.create("SQLite persistence diagnostic")
    store.conversations.add_message(diagnostic_conversation.id, "user", "Persist this diagnostic message.")
    store.conversations.add_message(diagnostic_conversation.id, "assistant", "SQLite stored this completed response.",
                                    "persistence-diagnostic")
    return {
        "checkpoint": "Real EmbeddingGemma image vectors stored as SQLite float32 blobs",
        "database_path": store.database.path,
        "image_dimension": dimensions.pop(),
        "vector_origins": origins,
        "text_embedding": {"origin": text_origin, "dimension": text_dimension},
        "image_embedding_ms": image_latencies_ms,
        "store_and_validation_ms": image_elapsed_ms,
        "rankings": rankings,
        "all_expected_images_ranked_first": all(item["top_match"] == expected for expected, item in rankings.items()),
        "conversation_persisted": diagnostic_conversation.id,
    }


def run_persisted_image_retrieval_checkpoint() -> dict:
    """Reload image vectors only; intentionally never invokes embed_image."""
    provider = _require_image_provider()
    store = get_store()
    saved = store.embeddings.for_model(EMBEDDING_MODEL_IDENTIFIER, "image")
    image_vectors: dict[str, list[float]] = {}
    missing_source_files: list[str] = []
    for source, embedding in saved:
        label = source.metadata.get("label")
        if label in {"cat", "car", "banana"}:
            image_vectors[str(label)] = embedding.vector
            if not Path(source.reference).is_file():
                missing_source_files.append(source.reference)
    if set(image_vectors) != {"cat", "car", "banana"}:
        raise RuntimeError("Persisted cat/car/banana image vectors are not all available. Run the store checkpoint first.")
    stored_text = store.embeddings.for_model(EMBEDDING_MODEL_IDENTIFIER, "text")
    if not stored_text:
        raise RuntimeError("Persisted text embedding is not available. Run the store checkpoint first.")
    dimensions = {len(vector) for vector in image_vectors.values()}
    if len(dimensions) != 1 or not all(math.isfinite(value) for vector in image_vectors.values() for value in vector):
        raise RuntimeError("Stored image vector validation failed.")
    started = perf_counter()
    rankings = _rank_queries(provider, image_vectors)
    return {
        "checkpoint": "Persisted SQLite image vectors reloaded without image re-embedding",
        "database_path": store.database.path,
        "image_dimension": dimensions.pop(),
        "image_vectors": "loaded_from_sqlite_without_embed_image",
        "stored_text_embeddings": {"count": len(stored_text), "dimensions": sorted({item.dimension for _, item in stored_text})},
        "query_embedding_and_ranking_ms": (perf_counter() - started) * 1000,
        "missing_source_files": missing_source_files,
        "rankings": rankings,
        "all_expected_images_ranked_first": all(item["top_match"] == expected for expected, item in rankings.items()),
    }

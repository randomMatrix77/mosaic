# Mosaic Android proof of concept

Mosaic is a single-screen Android proof of concept with a real Kotlin → Chaquopy → Python → JNI → llama.cpp path. Python owns chat application logic, SQLite persistence, history selection, embedding persistence, and retrieval scoring. Kotlin supplies the UI, app-private storage locations, lifecycle handling, and model-file import.

## Build and install

The project expects Android SDK API 36, NDK 27.2.12479018, CMake 3.22.1, and Java 17. From the repository root:

```sh
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Package ID: `dev.mosaic.poc`. Models are not bundled or mocked. The app uses Android's file picker and copies selected files to app-private storage.

For Qwen, select **Import Qwen GGUF** and choose the Qwen 3.5 4B Q4_K_M GGUF. On the target Fold 6, the app uses llama.cpp's OpenCL backend to offload the model to the Adreno 750 GPU, with a 2048-token context and 128-token cap.

For EmbeddingGemma 2, import both official Q8 artifacts, in this order:

1. **Import EmbeddingGemma GGUF**: `embeddinggemma-2-Q8_0.gguf` from [ggml-org/embeddinggemma-2-GGUF](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF).
2. **Import EmbeddingGemma projector**: `mmproj-embeddinggemma-2-Q8_0.gguf` from the same repository.

Use **Text embeddings** for real text vectors and **Image retrieval** for the bundled cat/car/banana photograph experiment. **Store image vectors** writes real image embeddings as SQLite float32 blobs, while **Load saved vectors** retrieves those image vectors without calling image embedding again and performs new text-query embeddings. The embedding model and Qwen intentionally share one native model slot: importing one unloads the other.

## Local persistence

Kotlin passes the app-private database path to Python at startup. Python's standard-library `sqlite3` backend stores:

* conversations and chronological messages, including role, model identifier, timestamp, and completion status;
* image sources with fingerprints and JSON metadata; and
* normalized embedding vectors as validated little-endian IEEE-754 float32 blobs.

On the Fold the database is at `files/mosaic/mosaic.sqlite3` (equivalent to `/data/user/0/dev.mosaic.poc/files/mosaic/mosaic.sqlite3`). It uses foreign keys, WAL mode, transactions with rollback, and schema versioning via `PRAGMA user_version`. Do not delete the app or clear its data when installing an update: use `adb install -r` to retain conversations and vectors.

Create a conversation before sending a chat message. Existing conversations can be opened after a process restart. Python bounds completed stored history before asking JNI to apply Qwen's native chat template; failed generation never produces a completed assistant-message record.

## Layout

* `app/src/main/java/dev/mosaic/poc/MainActivity.kt` — Compose UI and off-main-thread Chaquopy invocation.
* `app/src/main/python/mosaic/core.py` — Python-owned message handling.
* `app/src/main/python/mosaic/inference.py` — Python inference-provider boundary with the Android JNI adapter.
* `app/src/main/python/mosaic/embeddings.py` — model-agnostic Python embedding interface; audio and video explicitly raise `NotImplementedError`.
* `app/src/main/python/mosaic/embedding_diagnostics.py` — real text and image embedding/retrieval checks.
* `app/src/main/python/mosaic/storage.py` — portable standard-library SQLite schema, migrations, repositories, and float32 vector codec.
* `app/src/main/cpp/mosaic_jni.cpp` — llama.cpp JNI implementation, including model lifecycle, GPU/CPU fallback, generation, text embeddings, and libmtmd image embeddings.
* `app/src/main/cpp/llama.cpp/` — pinned official llama.cpp source (`6b9ba4f2e8c1436511956f0b3c309e2c9e642071`).

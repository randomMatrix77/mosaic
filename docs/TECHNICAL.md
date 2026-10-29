# Mosaic technical guide

This document describes the implementation behind the Android proof of concept. It is intentionally specific about what has been run on a physical device and what remains diagnostic or incomplete.

## Scope and status

Mosaic proves this local execution path:

```text
Compose UI
  → Chaquopy Python 3.11
    → Android JNI bridge
      → llama.cpp / libmtmd
    → sqlite3
  → Compose UI
```

There is no network request in inference. Models are selected through Android's file picker, copied into app-private storage, and opened locally by llama.cpp.

The application package is `dev.mosaic.poc`. It is an Android-only proof of concept, not a library or a production application.

## Repository and source layout

| Area | Location | Responsibility |
| --- | --- | --- |
| Kotlin UI and lifecycle | `app/src/main/java/dev/mosaic/poc/MainActivity.kt` | Compose screen, file picker, background calls into Python, and UI state. |
| Kotlin JNI API | `app/src/main/java/dev/mosaic/poc/NativeLlama.kt` | Boundary exposed to Chaquopy/Python. |
| Native implementation | `app/src/main/cpp/mosaic_jni.cpp` | Model lifecycle, text generation, text embedding, image embedding, and native metrics. |
| Native build | `app/src/main/cpp/CMakeLists.txt` | Builds llama.cpp and the JNI shared library. |
| Python chat logic | `app/src/main/python/mosaic/core.py` | Conversations, bounded history, completed-response persistence, and application errors. |
| Python inference adapter | `app/src/main/python/mosaic/inference.py` | Model-agnostic inference boundary backed by `NativeLlama`. |
| Python embeddings | `app/src/main/python/mosaic/embeddings.py` | Embedding provider interface and EmbeddingGemma adapter. |
| Embedding diagnostics | `app/src/main/python/mosaic/embedding_diagnostics.py` | Real embedding, similarity, retrieval, and persistence diagnostics. |
| SQLite persistence | `app/src/main/python/mosaic/storage.py` | Schema migration, repositories, transactions, and float32 vector codec. |

The native dependencies are Git submodules. `app/src/main/cpp/llama.cpp` is pinned to llama.cpp revision `6b9ba4f2e8c1436511956f0b3c309e2c9e642071`; Khronos OpenCL and Vulkan headers are pinned separately. Initialize submodules before building:

```bash
git submodule update --init --recursive
```

## Supported build environment

The tested setup is:

- macOS development host
- Java 17
- Android SDK API 36
- Android NDK `27.2.12479018`
- CMake `3.22.1`
- Android Gradle Plugin `9.0.1`
- Kotlin/Compose plugin `2.3.20`
- Chaquopy `17.1.0` with Python 3.11
- Samsung Galaxy Z Fold 6 (`SM-F956B`) with USB debugging

Build and install without destroying stored data:

```bash
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Use `adb install -r`, not uninstall/reinstall, when preserving imported weights, conversations, and embeddings matters.

## Architecture and execution flow

### Chat

1. Kotlin starts Chaquopy and passes app-private paths to Python.
2. `MosaicCore` creates or opens a conversation through the Python persistence layer.
3. The user message is stored inside a SQLite transaction.
4. Python reads completed prior messages, bounds the history conservatively, and delegates to `AndroidInferenceProvider`.
5. The adapter calls the JNI-backed `NativeLlama` API.
6. Native llama.cpp applies the Qwen chat template, performs generation, and returns a complete response and timing information.
7. Python stores an assistant message only after successful completion, then returns the result to Kotlin.

If model loading or generation fails, the Python layer returns a readable error. It does not create a falsely completed assistant message. The full history remains stored even though each native prompt is bounded for the context window.

### Embeddings

1. Kotlin imports an EmbeddingGemma model and its matching multimodal projector through file pickers.
2. Python calls `EmbeddingGemmaProvider`.
3. The provider invokes the JNI bridge, which uses llama.cpp for text embeddings and `libmtmd` for image preprocessing/multimodal embeddings.
4. Python validates and normalizes vectors, calculates cosine similarity, and manages diagnostic storage/reload checks.

`embed_audio` and `embed_video` are deliberately present as future-facing interface methods and raise `NotImplementedError`. They do not produce simulated vectors.

## Native inference

### Qwen 3.5 4B

The currently tested generation model is the Qwen 3.5 4B Q4_K_M GGUF linked from the top-level README. It is not bundled in the APK. The initial configuration is intentionally conservative:

- context size: 2048 tokens
- output limit: 128 tokens
- one request at a time
- thinking mode disabled for simple diagnostics
- one loaded native model at a time

On the verified Fold 6 build, Qwen uses the OpenCL backend and offloads all 33 layers to the Adreno 750 GPU. If GPU setup is unavailable, the JNI layer falls back to CPU execution rather than pretending the model loaded on GPU.

### EmbeddingGemma 2

Embedding tests use the official `embeddinggemma-2-Q8_0.gguf` plus matching `mmproj-embeddinggemma-2-Q8_0.gguf`. Both files are required for image embeddings. Text and image output vectors are 768 dimensions and are normalized before comparison.

Embedding inference currently uses the CPU path. This keeps its behavior separated from the tested Qwen OpenCL configuration and avoids claiming GPU support that has not been independently validated for the multimodal path.

### OpenCL GPU linkage

The project expects an ABI-specific link stub at:

```text
app/src/main/cpp/prebuilt/arm64-v8a/libOpenCL.so
```

This file is ignored by Git. It must be compatible with the target Android device and is used only to satisfy native linkage; Android loads the device's OpenCL implementation at runtime. It must not be treated as a portable desktop library or committed as a generic project artifact.

As a result, a fresh clone can build the CPU implementation after submodules are initialized, but an equivalent OpenCL build may require preparing that device-specific stub first. Verify GPU use through logcat, rather than relying only on a successful APK build.

Vulkan headers remain in the repository's native dependencies, but Vulkan is not the selected Qwen backend for the tested Fold 6 configuration.

### Model lifecycle and memory

To keep peak memory reasonable, the JNI bridge owns a single model slot. Loading Qwen unloads EmbeddingGemma; loading EmbeddingGemma unloads Qwen. Qwen's imported model path is retained and restored on startup when the private file remains available.

The UI dispatches Python, SQLite, model loading, and inference off the Android main thread. Native generation is serialized to prevent concurrent use of model/context resources.

## Model import and private files

Android's Storage Access Framework gives the app a selected file URI. Kotlin copies it to app-private model storage before telling Python/native code about the copied path. This avoids hard-coded device paths and avoids relying on transient URI permissions during later restarts.

Weights are intentionally excluded from Git and from the APK. The user must import:

- Qwen GGUF for chat;
- EmbeddingGemma GGUF for text embeddings; and
- its matching `mmproj` GGUF for image embeddings.

If a selected model or projector is missing, incompatible, or deleted, native loading reports an error to Python and then to the diagnostic UI.

## SQLite persistence

### Location and connection policy

Kotlin provides an app-private path to Python. On the tested Fold it resolves to:

```text
/data/user/0/dev.mosaic.poc/files/mosaic/mosaic.sqlite3
```

Within the app this is `files/mosaic/mosaic.sqlite3`. SQLite connections enable foreign keys, WAL journaling, a busy timeout, explicit transactions, and rollback on failure. Schema changes use `PRAGMA user_version`; migrations are additive and do not silently erase records.

When examining a live database outside the app, remember that WAL mode may have `-wal` and `-shm` sidecar files. Copy all relevant files only when safe to do so; do not clear app data as part of diagnosis.

### Schema

| Table | Purpose |
| --- | --- |
| `conversations` | ID, optional title, and creation/update timestamps. |
| `messages` | ID, conversation ID, role, content, model ID, timestamp, and completion state. |
| `sources` | ID, modality, app-private/reference path, optional MIME type, fingerprint, and JSON metadata. |
| `embeddings` | ID, source ID, model identifier, dimension, normalization state, float32 vector blob, and creation timestamp. |

Foreign keys connect messages to conversations and embeddings to sources. Deleting a source cascades to its embeddings. Uniqueness constraints protect against duplicate vectors for the same source/content version/model combination, so incompatible model outputs cannot be silently mixed.

### Vector format

Embedding vectors are validated to ensure that dimensions match and every value is finite. They are encoded as contiguous little-endian IEEE-754 `float32` blobs, with their dimension and normalization information stored alongside. This keeps the database independent of NumPy and usable from a future desktop Python application.

Cosine similarity is calculated in Python against vectors loaded from SQLite; there is no ANN index or vector database in this proof of concept.

## Verified results

The following tests were run on the physical Samsung Galaxy Z Fold 6. They are real native executions, not fixtures or mock responses.

| Check | Result |
| --- | --- |
| Kotlin/Python/native generation path | Verified end to end with a local Qwen response. |
| Qwen GPU loading | OpenCL offloaded 33/33 layers; reported OpenCL model buffer: `2106.31 MiB`. |
| Embedding vector shape | Text and image embeddings were 768 dimensions. |
| Text semantic check | Cat/kitten cosine: `0.947830`; cat/car cosine: `0.763599`; identical repeated input had delta `0`. |
| Cat image query | cat `0.716594`, banana `0.616985`, car `0.604613`; cat ranked first. |
| Car image query | car `0.679837`, cat `0.617319`, banana `0.612644`; car ranked first. |
| Banana image query | banana `0.744346`, cat `0.640275`, car `0.596801`; banana ranked first. |
| SQLite image-vector reload | After force-stop/restart, saved 768-dimensional image vectors were loaded without image re-embedding; the fresh query embedding took `211.6868 ms` and all three expected first-place rankings were reproduced. |
| Conversation persistence | A saved conversation and completed response survived force-stop/restart. A subsequent question referring to the earlier message was answered using stored history. |

The embedding persistence diagnostic has been physically verified with image vectors. Text embeddings themselves and the common vector codec/repository path are real, but a dedicated post-change, on-device text-vector save/reload run remains a useful follow-up check.

## Running diagnostics

The Compose screen is intentionally a diagnostic UI rather than a finished interface.

### Chat and conversations

Create a conversation, send a message, and wait for the completed assistant response. The interface can list, open, rename, and delete conversations. Reopen a prior conversation to verify persistence and bounded history handling.

### Text embeddings

After importing EmbeddingGemma, run the text diagnostic. It embeds similar and unrelated sentences, verifies finite 768-dimensional vectors, checks repeated-input consistency, and prints cosine similarities.

### Image retrieval and persistence

Import both the embedding model and matching projector, then run image retrieval. The diagnostic uses actual cat, car, and banana image content, not filenames or synthetic descriptions. Use the storage action to persist the image vectors. After process restart, use the load action: it loads stored image vectors, creates only a new text-query embedding, then prints scores and rankings.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| CMake cannot find llama.cpp or headers | Run `git submodule update --init --recursive`. |
| APK builds but OpenCL fails to link or load | Confirm the ABI-specific ignored `libOpenCL.so` stub exists, then inspect device logcat for the active backend. CPU fallback is expected if GPU initialization cannot succeed. |
| Image embedding fails | Confirm both EmbeddingGemma GGUF and exactly matching `mmproj` GGUF were imported. |
| Model does not restore | Confirm the app has not been uninstalled/cleared and its private model file still exists. Re-import if needed. |
| Data appears missing | Install with `adb install -r`; do not use uninstall or clear-storage operations. Check the private SQLite path and WAL sidecars. |
| UI freezes | Collect logcat and confirm work is still dispatched outside the main thread; native inference should remain single-request-at-a-time. |

Useful commands:

```bash
adb devices
adb logcat | rg 'Mosaic|llama|ggml|OpenCL'
adb shell am force-stop dev.mosaic.poc
```

## Known limitations and next steps

- Output is returned as a completed response; token streaming is not implemented.
- Qwen GPU support is validated on the Fold 6 OpenCL configuration only. It is not a portability guarantee for every Android GPU.
- EmbeddingGemma runs on CPU in the current application.
- The retrieval corpus is a small diagnostic set held in memory plus SQLite persistence; it is not an indexing system.
- Audio and video embeddings intentionally report unsupported capability.
- There is no document ingestion, RAG, automatic memory, cloud fallback, or semantic chat search.
- Model/profile configuration is deliberately hard-coded for this experiment rather than exposed as a broad model manager.

The intended next step is to continue testing and hardening these verified local paths, while preserving existing private data and keeping future retrieval/memory work separate from this persistence foundation.

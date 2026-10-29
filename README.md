# Mosaic

Mosaic is an Android proof of concept for a local-first personal intelligence application. It demonstrates that application logic can live in embedded Python while models run fully on an Android device:

```text
Kotlin + Jetpack Compose → Chaquopy → Python → JNI → llama.cpp
                                  ↘ SQLite
```

This is deliberately a small, diagnostic-oriented application—not the full Mosaic product. It is useful as a working reference for local Qwen generation, multimodal embeddings, and Python-owned persistence on Android.

## Current capabilities

- Offline Qwen 3.5 4B generation using llama.cpp, with the chat flow owned by Python.
- Qwen GPU execution on the tested Galaxy Z Fold 6 via OpenCL, with a CPU fallback.
- EmbeddingGemma 2 text and image embeddings through the same native backend.
- A real text-to-image retrieval diagnostic over cat, car, and banana test photos.
- Python `sqlite3` persistence for conversations, messages, sources, and float32 embedding vectors.
- A minimal Compose interface for chat, conversation management, model import, and diagnostics.

## Intentionally not included

Mosaic does **not** yet include RAG, document ingestion, automatic indexing, memory extraction, semantic chat search, a vector database, cloud services, authentication, audio/video embeddings, or a production-polished interface.

## Clone the repository

This repository uses Git submodules for llama.cpp and native header dependencies. Clone recursively:

```bash
git clone --recurse-submodules https://github.com/randomMatrix77/mosaic.git
cd mosaic
```

If you already cloned it without submodules, initialize them before building:

```bash
git submodule update --init --recursive
```

The checked-in submodules are:

- [`llama.cpp`](https://github.com/ggml-org/llama.cpp), the native inference runtime.
- [`OpenCL-Headers`](https://github.com/KhronosGroup/OpenCL-Headers), used by the selected Android GPU backend.
- [`Vulkan-Headers`](https://github.com/KhronosGroup/Vulkan-Headers), retained as a native dependency for experimentation; Vulkan is not the active Qwen backend.

Model weights are intentionally not in Git or the APK. Obtain them separately and comply with their respective licences and terms:

- [Qwen 3.5 4B Q4_K_M GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf)
- [EmbeddingGemma 2 GGUF repository](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF) — import both the embedding GGUF and its matching `mmproj` file for image embeddings.

## Requirements

The currently tested development environment is macOS with Java 17, Android SDK API 36, Android NDK `27.2.12479018`, and CMake `3.22.1`. A physical Android device with USB debugging is recommended; the implementation was verified on a Samsung Galaxy Z Fold 6 (`SM-F956B`).

There is one additional native prerequisite for the current OpenCL configuration: the project links against an Android-device-compatible `libOpenCL.so` stub in `app/src/main/cpp/prebuilt/arm64-v8a/`. That binary is intentionally ignored by Git because it is device/vendor specific. Read the [technical guide](docs/TECHNICAL.md#opencl-gpu-linkage) before attempting a fresh GPU build on another phone.

## Build, install, and run

From the repository root:

```bash
./gradlew :app:assembleDebug
adb devices
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

`adb install -r` updates the app without clearing its data. Do not uninstall or clear app data if you want to retain imported models, conversations, and the SQLite database.

On first launch:

1. Use the Qwen import action to select the downloaded Qwen GGUF.
2. Wait for model loading, then create a conversation and send a message.
3. To run embedding diagnostics, import the EmbeddingGemma GGUF and matching multimodal projector (`mmproj`) using the corresponding file-picker actions.

The app copies selected model files into its private storage; it does not bundle multi-gigabyte weights in the APK.

## What a new contributor should know

The Android app is intentionally single-module and compact. Kotlin handles UI, lifecycle, file selection, and native bindings. Python owns message handling, bounded conversation-history preparation, persistence, embedding interfaces, retrieval calculations, and errors. llama.cpp owns actual model loading and inference.

The project uses only one native model slot at a time to keep memory use practical. Loading EmbeddingGemma unloads Qwen and vice versa. Qwen is restored automatically on startup when its imported file is still present.

For component layout, data formats, model settings, testing evidence, and troubleshooting, read the [technical guide](docs/TECHNICAL.md).

## Verification snapshot

The following were run on the physical Fold 6, not mocked:

- Qwen generation through Kotlin → Python → JNI → llama.cpp → Python → Kotlin, with all 33 model layers offloaded through OpenCL.
- 768-dimensional EmbeddingGemma text and image vectors.
- Text-to-image retrieval ranked the matching cat, car, and banana photos first for their respective text queries.
- Saved image embeddings were reloaded from SQLite after an application process restart and produced the expected rankings without regenerating image vectors.
- Conversations and completed assistant replies persisted through a force-stop/restart; a follow-up response used earlier stored context.

The [technical guide](docs/TECHNICAL.md#verified-results) records the measured diagnostic results and the remaining limitations.

## Project map

```text
app/
  src/main/java/       Kotlin Compose UI and JNI bridge
  src/main/python/     Mosaic application, inference, embeddings, and storage
  src/main/cpp/        Native bridge, CMake, llama.cpp and header submodules
docs/                  Implementation and operational documentation
```

## Status

This is an actively evolving Android experiment. Preserve existing app data while testing, keep model files outside Git, and treat the diagnostic UI and benchmark results as proof-of-concept tooling rather than production UX.

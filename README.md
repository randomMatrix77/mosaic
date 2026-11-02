# Mosaic

Mosaic is an Android proof of concept for a local-first personal intelligence application. It demonstrates that application logic can live in embedded Python while models run fully on an Android device:

```text
Kotlin + Jetpack Compose → Chaquopy → Python → JNI → llama.cpp
                                  ↘ SQLite
```

This is deliberately a small, diagnostic-oriented application—not the full Mosaic product. It is useful as a working reference for local Qwen generation, multimodal embeddings, and Python-owned persistence on Android.

## Current capabilities

- Offline Qwen 3.5 4B generation using llama.cpp, with the chat flow owned by Python.
- Qwen 3.5 4B image-plus-text requests using its matching multimodal projector.
- Qwen GPU execution on the tested Galaxy Z Fold 6 via OpenCL, with a CPU fallback.
- EmbeddingGemma 2 text and image embeddings through the same native backend.
- A real text-to-image retrieval diagnostic over cat, car, and banana test photos.
- A Python-owned Qwen tool loop with safe, read-only image and memory retrieval tools; Qwen decides whether to call them.
- Python `sqlite3` persistence for conversations, messages, sources, and float32 embedding vectors.
- An evidence-backed SQLite memory graph with entities, claims, corrections, provenance, and semantic claim retrieval, verified on the Fold 6.
- A minimal Compose interface for chat, conversation management, model import, and diagnostics.

## Intentionally not included

Mosaic does **not** yet include RAG, document ingestion, automatic indexing, background memory processing, a vector database, cloud services, authentication, audio/video embeddings, or a production-polished interface.

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
- [Qwen 3.5 4B multimodal projector (`mmproj-F16.gguf`)](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/mmproj-F16.gguf) — required for Qwen image input; this exact file is distinct from EmbeddingGemma's projector.
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

### ADB diagnostic interface

The debug APK includes a deliberately narrow, JSON-based command surface for rapid iteration without operating the Compose diagnostic controls. It is not a shell: all dispatch remains in the Python backend and only the documented commands are accepted. A request is sent to a debug-only, transparent Activity so long-running local inference is not subject to Android's broadcast receiver timeout. Its result is written to app-private storage:

```bash
mosaic_debug() {
  request_b64="$(printf '%s' "$1" | base64 | tr -d '\n')"
  adb shell am start -n dev.mosaic.poc/.DebugCommandActivity \
    -a dev.mosaic.poc.DEBUG_COMMAND --es request_b64 "$request_b64"
  adb exec-out run-as dev.mosaic.poc cat files/mosaic-debug/last-result.json
}

mosaic_debug '{"operation":"status"}'
mosaic_debug '{"operation":"agent.chat","title":"Debug chat","message":"Hello"}'
mosaic_debug '{"operation":"images.search","query":"a photograph of a cat","top_k":3}'
mosaic_debug '{"operation":"qwen.vision","image_name":"cat.jpg","message":"Describe this image in one sentence."}'
mosaic_debug '{"operation":"memory.claims"}'
mosaic_debug '{"operation":"memory.search","query":"What technologies does Mosaic use?"}'
```

`am start` returns immediately; read the result file after the command completes. Only issue one inference command at a time. The bundled diagnostic images are `cat.jpg`, `car.jpg`, and `banana.jpg`. To supply another image while using the debug build, copy it deliberately into the app-private debug directory, then pass only its filename to `qwen.vision`:

```bash
adb push ~/Pictures/example.jpg /data/local/tmp/mosaic-example.jpg
adb shell run-as dev.mosaic.poc mkdir -p files/mosaic-debug/images
adb shell run-as dev.mosaic.poc cp /data/local/tmp/mosaic-example.jpg files/mosaic-debug/images/example.jpg
```

For the same debug-only workflow, the Qwen projector belongs at `files/models/selected-qwen35-mmproj-f16.gguf`. Verify downloads before import: the official F16 projector is 672,423,616 bytes with SHA-256 `cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864`.

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
- Qwen autonomously issued native-parsed `search_images` tool calls for banana and car requests. Python validated the calls, searched the saved EmbeddingGemma vectors, and Qwen generated the final answers after the tool results were added to context.
- Qwen received real `cat.jpg` and `car.jpg` pixels through libmtmd and the matching Qwen projector. It described a light-colored kitten with blue eyes for the cat and identified the blue BMW car for the car input.

The [technical guide](docs/TECHNICAL.md#verified-results) records the measured diagnostic results and the remaining limitations.

The memory graph was verified on the Fold 6 through the full local path: Qwen emitted a native `search_memories` tool call, Python validated and executed it, EmbeddingGemma searched persisted claim vectors, and Qwen generated a grounded final response. The test completed in 86.45 seconds; this is a diagnostic proof rather than an interactive-latency target.

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

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
| Memory skills | `app/src/main/python/mosaic/skills/` | Packaged versioned Markdown instructions for responder, extractor, and curator tasks. |
| Memory orchestration | `app/src/main/python/mosaic/memory.py` | Strict extraction/curation parsing, graph persistence, and semantic retrieval. |
| Python debug commands | `app/src/main/python/mosaic/debug.py` | Closed ADB JSON command surface, including vision diagnostics. |
| Debug command Activity | `app/src/main/java/dev/mosaic/poc/DebugCommandActivity.kt` | Debug-build-only Android Activity which initializes Chaquopy off the UI thread and persists structured command results without a receiver timeout. |
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

### Agent tool loop

The Python backend now has a small ReAct-style loop in `app/src/main/python/mosaic/agent.py`. The responder skill receives declared, read-only functions for `search_images(query, top_k=3)`, `search_memories(query, top_k=5)`, `web_search(query, max_results=4)`, and `web_fetch(url)`. Python validates the name and argument schema, embeds local queries with EmbeddingGemma, and returns structured image or evidence-backed memory results to Qwen for a final response. Web tools use DeepSeek Search & Fetch only for external retrieval; Qwen reasoning remains local. The DeepSeek API key is stored in Android-backed secure settings and is never logged or written to SQLite.

The model has no route to execute Python, shell commands, or arbitrary filesystem access. The registry is a closed allow-list, unknown tools and invalid arguments become structured tool errors, and each user request is limited to two tool calls with a cooperative 90-second orchestration deadline. Tool calls and tool results are stored as distinct SQLite message kinds, not as user messages or completed assistant replies.

The JNI bridge uses the pinned llama.cpp revision's `common_chat_session`, which renders the GGUF-provided Qwen 3.5 tool template and parses its structured output. Qwen's official current template uses `<tool_call>` with nested `<function=...>` and `<parameter=...>` tags; the native parser converts that output to structured function calls before Python validation.

This path was verified on the physical Fold 6 on 2026-10-30 with the imported Qwen, EmbeddingGemma, matching projector, and the three persisted image vectors. No native crash or malformed tool call occurred in the four requested checks:

- `Hello! How are you?` produced a direct Qwen answer, saved after about 12.5 seconds, with no tool event.
- `Find the picture of a banana in my indexed images.` produced `search_images({"query":"banana","top_k":3})`. Python accepted the arguments, loaded EmbeddingGemma in 1507.5 ms, embedded the query in 67.9 ms, and returned `banana.jpg` first at 0.7606 similarity. Qwen's final answer was saved after about 55.9 seconds total.
- `Which of my indexed photos shows a car?` produced `search_images({"query":"car","top_k":3})`. Python accepted the arguments, loaded EmbeddingGemma in 1400.1 ms, embedded the query in 48.9 ms, and returned `car.jpg` first at 0.6996 similarity. Qwen's final answer was saved after about 72.4 seconds total.
- `Describe exactly what's happening in the retrieved cat photograph.` produced no tool call and, after about 19.9 seconds, explicitly said it could not access the specific photograph rather than invent visual details.

Tool calls and results were persisted as distinct `tool_call` and `tool_result` messages. The actual tool-result payloads contain only source IDs, filenames, similarity scores, and embedding timing; they deliberately do not grant Qwen image inspection. Local test doubles cover malformed JSON arguments, unknown tools, invalid argument types, tool failures, the two-call bound, migration, and history replay. Those are not device-verification substitutes.

### Debug command surface

The debug manifest registers `DebugCommandActivity` only in debug builds. ADB sends one JSON request using the `dev.mosaic.poc.DEBUG_COMMAND` action; Kotlin runs it off the UI thread and passes it to `mosaic.debug`, which accepts only status, conversation, inference, and explicitly named memory operations. The response is atomically written to `files/mosaic-debug/last-result.json`. This Activity route is used for inference because a broadcast receiver may be killed by Android's timeout. It is a development harness, not a remotely exposed production API or a generic Python/shell endpoint.

`qwen.vision` accepts an app-private image filename, never an arbitrary path. It resolves files exclusively below `files/mosaic-debug/images`, preventing the debug interface from reading unrelated app files. The receiver copies the three retrieval fixtures into this directory on demand.

To invoke a debug command, encode one JSON request and send it to the debug-only Activity. The result is written to app-private storage:

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
```

## Native inference

### Qwen 3.5 4B

The currently tested generation model is the Qwen 3.5 4B Q4_K_M GGUF linked from the top-level README. It is not bundled in the APK. The initial configuration is intentionally conservative:

- context size: 2048 tokens
- output limit: 128 tokens
- one request at a time
- thinking mode disabled for simple diagnostics
- one loaded native model at a time

On the verified Fold 6 build, Qwen uses the OpenCL backend and offloads all 33 layers to the Adreno 750 GPU. If GPU setup is unavailable, the JNI layer falls back to CPU execution rather than pretending the model loaded on GPU.

#### Image input

Qwen image input uses the official matching Qwen projector, not the EmbeddingGemma projector:

```text
https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/mmproj-F16.gguf
```

The verified F16 file is 672,423,616 bytes and SHA-256 `cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864`. It is installed in app-private storage as `files/models/selected-qwen35-mmproj-f16.gguf`; it is not bundled in the APK or committed to Git.

The JNI path loads the file through llama.cpp's `libmtmd`, decodes the supplied image, inserts `mtmd_default_marker()` into the Qwen chat template, evaluates image embeddings in the same native context as text, then generates the answer with the already-loaded Qwen model. The projector itself is currently initialized on CPU; the Qwen model remains on its existing verified OpenCL path. Qwen and EmbeddingGemma still use separate matching projectors.

This standalone diagnostic entry point does not yet add images to persisted chat history or provide Qwen vision to the autonomous agent. It is intentionally bounded to one image and 128 generated tokens.

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

## Knowledge graph and provenance

Schema version 3 adds a derived memory graph without modifying the authoritative conversation ledger. `graph_entities` records canonical typed entities; `graph_claims` records atomic entity-to-entity or entity-to-literal facts; and `claim_evidence` attaches each claim to one or more original conversation/message records. A claim may have many evidence references, and one message may support many claims.

Every evidence row stores stable conversation and message UUIDs, an SHA-256 fingerprint of the original message text, and optional character offsets. When extraction cannot provide reliable offsets, it stores message-level provenance with `span_available = 0`; it never invents a span. Evidence resolution verifies the fingerprint before returning an exact quote plus surrounding messages.

`claim_embeddings` stores normalized float32 vectors with the existing validation/encoding discipline. `search_memories` returns only current claims, similarity scores, and source-reference IDs; it does not inject the entire graph into normal prompts. `memory_events` records create, evidence-link, supersession, deletion, and conversation-deletion history. Explicitly deleted claims remain tombstones, so reprocessing retained older text cannot recreate them silently.

Conversation deletion removes derived evidence links before the ledger cascade. Claims which lose all evidence are tombstoned and lose their embedding; claims with evidence in another conversation survive. Deleting a memory removes its evidence links and embedding but does not rewrite the original conversation.

Three package-local Markdown skills are bundled in the Chaquopy APK: responder, memory extractor, and memory curator. `SkillRegistry` assigns each a fixed Python-enforced tool allowlist. Extractor/curator output uses closed JSON schemas and is validated against stored message IDs, conversation IDs, and offsets before writes. Obvious email/phone contact data is refused by automatic extraction until a consent flow exists.

The graph has passed local test-double tests and real Fold 6 verification. Qwen emitted a native `search_memories` function call, Python validated its arguments, EmbeddingGemma searched persisted claim embeddings, and Qwen used the returned evidence-backed results in its final answer. The complete two-pass diagnostic took 86.45 seconds: 6.48 s Qwen load/restore plus 20.60 s first generation; 1.41 s EmbeddingGemma load plus 79 ms query embedding; then 7.63 s Qwen load/restore plus 50.22 s final generation.

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
| Qwen vision: cat | `qwen.vision` passed actual `cat.jpg` pixels through the Qwen F16 projector and Qwen described a light-colored kitten with blue eyes. Generation: `57,538.7 ms`. |
| Qwen vision: car | `qwen.vision` passed actual `car.jpg` pixels and Qwen identified a blue BMW car (rather than cat or banana). Generation: `39,660.0 ms`. |

The embedding persistence diagnostic has been physically verified with image vectors. Text embeddings themselves and the common vector codec/repository path are real, but a dedicated post-change, on-device text-vector save/reload run remains a useful follow-up check.

## Running diagnostics

The Compose screen is intentionally a diagnostic UI rather than a finished interface.

### Chat and conversations

Create a conversation, send a message, and wait for the completed assistant response. The interface can list, open, rename, and delete conversations. Reopen a prior conversation to verify persistence and bounded history handling.

### Text embeddings

After importing EmbeddingGemma, run the text diagnostic. It embeds similar and unrelated sentences, verifies finite 768-dimensional vectors, checks repeated-input consistency, and prints cosine similarities.

### Image retrieval and persistence

Import both the embedding model and matching projector, then run image retrieval. The diagnostic uses actual cat, car, and banana image content, not filenames or synthetic descriptions. Use the storage action to persist the image vectors. After process restart, use the load action: it loads stored image vectors, creates only a new text-query embedding, then prints scores and rankings.

### ADB commands and Qwen vision

Use the debug-command procedure above to invoke Python-owned diagnostics without interacting with the Compose screen. For Qwen vision, first copy the official projector into `files/models/selected-qwen35-mmproj-f16.gguf` and supply an image inside `files/mosaic-debug/images`. The receiver creates that image directory and populates the three test images automatically. Only a basename is accepted by Python, so a debug request cannot select arbitrary filesystem paths.

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

- Final-answer tokens stream to the UI during generation. Streaming is intentionally limited to user-visible answer text: tool payloads and model reasoning are never shown as a token trace.
- Qwen vision is verified as a standalone image-plus-text diagnostic, but it is not yet multimodal persisted chat, agent vision, or multi-image conversation support. Qwen sometimes emits reasoning-style prose even when the native template requests non-thinking output; the bounded 256-token cap can therefore truncate a response.
- Qwen GPU support is validated on the Fold 6 OpenCL configuration only. It is not a portability guarantee for every Android GPU.
- EmbeddingGemma runs on CPU in the current application.
- The retrieval corpus is a small diagnostic set held in memory plus SQLite persistence; it is not an indexing system.
- Audio and video embeddings intentionally report unsupported capability.
- Memory extraction, curation, and cross-conversation retrieval are locally covered with explicit test doubles. The first full Fold retrieval proof is complete; broader extraction/curation scenario coverage still requires device tests. There is no consent UI, automatic background ingestion, graph visualization, document ingestion, RAG, or cloud fallback.
- Model/profile configuration is deliberately hard-coded for this experiment rather than exposed as a broad model manager.

The intended next step is to continue testing and hardening these verified local paths, while preserving existing private data and keeping future retrieval/memory work separate from this persistence foundation.

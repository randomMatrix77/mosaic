# vibeOS Intelligence Core v0.1 — implementation plan

## Verified baseline

- Target device: Samsung Galaxy Z Fold 6 (`SM-F956B`), Android 16, `arm64-v8a`.
- Host: macOS on Apple Silicon, Temurin JDK 17, Node 23, Android API 36/build-tools 36.
- Official MNN source reviewed at commit `45a002ab19f9021e982ed153826586bb79e41837`.
- MNN Chat builds `libMNN.so` with `MNN_BUILD_LLM`, transformer fuse, Arm82, and vision enabled. Its native bridge owns an `MNN::Transformer::Llm`, calls `Llm::createLLM(config.json)`, forwards UTF-8 chunks through a callback, and reads `LlmContext` performance fields after generation.

## Architecture

```text
apps/mobile                  React/Vite UI and Capacitor bridge
packages/core                Runtime-agnostic chat, inference, tool, and attachment types
packages/models              Model registry and MNN-package validation rules
packages/storage             Repository interfaces and SQLite schema/migrations
packages/embeddings          Independent LiteRT-LM embedding-provider contract
native/android               Capacitor plugin, Kotlin runtime owner, JNI/CMake bridge
```

`MnnInferencePlugin` is deliberately narrow: it accepts an installed, validated model directory; loads only one native session; sends token/error/completion events; supports cancellation; and releases native resources. The React layer never calls JNI directly.

## MNN model contract

The initial registry entry is `taobao-mnn/Qwen3.5-4B-MNN` (the official MNN Chat registry lists it as a 2,845,946,907-byte, Think + Vision package). A normal Hugging Face Qwen checkout is rejected. An installed package must include `config.json`, a non-empty `.mnn` graph, and tokenizer assets referenced by the config. The runtime reads the package configuration rather than assuming Qwen-specific file names or prompt formatting.

The APK will contain the MNN runtime, not invoke, bind to, or depend on the separately installed MNN Chat application. Models live in app-private storage and are installed through an explicit user-driven importer/download flow in a later milestone.

## Milestone A — native text inference probe

1. Create the Android shell and a small JNI bridge based on the official `Llm::createLLM`/`response` API.
2. Build MNN from the pinned official source with LLM, ARM64, 16 KB page-size, and vision flags; package `libMNN.so` inside our APK.
3. Add explicit load/unload/generate/cancel/error/metric APIs and a minimal Kotlin activity for device diagnostics.
4. Install without clearing existing application data, import an official MNN Qwen 3.5 4B package into this app's private storage, and prove generated output from device logs/UI.
5. Commit only after the build and on-device run succeed.

## Follow-on milestones

- **B:** create the React/Vite/Capacitor app and plugin event bridge; verify streaming, cancellation, and error propagation.
- **C:** add Android Photo Picker, camera capture, SAF document picking, local TXT/Markdown/PDF extraction with explicit failure states, Room/SQLite conversation and attachment persistence, plus MNN vision inputs.
- **D:** add an independent LiteRT-LM `EmbeddingProvider` lifecycle contract, schema tables for vectors/provenance/memory, and a diagnostic only when the official EmbeddingGemma Android artifact is verifiably usable.

## Risks and gates

- This workstation does not currently expose an Android NDK or `sdkmanager`; the required NDK r27 must be installed before compiling JNI and MNN.
- The official Qwen 3.5 4B package is approximately 2.85 GB. It is not currently available in a user-accessible model location, so real generation cannot be claimed until it is explicitly downloaded/imported.
- MNN library headers and `libMNN.so` must be built from the same pinned MNN revision. A mismatched runtime is rejected as a build/configuration error.
- Qwen 3.5 4B is marked Vision in MNN's registry. Vision input will be passed through MNN's `MultimodalPrompt` API in Milestone C; no image-processing success will be simulated before that is verified.
- LiteRT-LM remains independent from MNN and will not produce placeholder embeddings.

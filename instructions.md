## Development workflow — Device-independent testing first

**Important: My Samsung Galaxy Z Fold 6 will NOT remain connected to the Mac throughout development.**

Do not assume the physical device is available.

We will use a two-phase development workflow.

### Phase 1 — Implement and test on the Mac (no device required)

Complete as much implementation and testing as possible without Android device access.

For the current agent milestone:

1. Implement the Python agent loop, tool registry, argument validation, and structured tool responses.
2. Write Python unit tests for direct answers, tool execution, invalid arguments, unknown tools, tool failures, and maximum tool-call limits.
3. Write integration tests that exercise the complete Python orchestration flow using explicitly labeled test doubles for the inference and embedding providers.
4. Test conversation history serialization, assistant tool calls, tool responses, and persistence behavior.
5. Test the Kotlin/Python/JNI interface contracts where feasible without a physical device.
6. Run available Python static checks and test suites.
7. Compile the native Android components and build the debug APK using Gradle.
8. Resolve all locally reproducible compilation failures and test failures before requesting device access.

Test doubles are acceptable for automated tests, but must never be presented as successful real Qwen or EmbeddingGemma inference.

Do not require ADB or a connected device during this phase.

Do not repeatedly check for the device, block waiting for it, or attempt installation.

### Phase 2 — Device testing checkpoint

Once Phase 1 is complete, STOP and send me a concise report containing:

- What was implemented
- Which automated tests passed or failed
- Whether the APK compiled successfully
- The location of the debug APK
- Which parts remain unverified without real Android inference
- The exact on-device tests you intend to perform

Then explicitly tell me:

**"The implementation and available local tests are complete. Please connect your Fold 6 via USB, unlock it, and enable USB debugging so we can begin on-device testing."**

Do not begin device testing until I confirm that the phone is connected.

Once I confirm:

1. Verify ADB connectivity and device authorization.
2. Install the APK using `adb install -r`, preserving existing application data.
3. Run actual inference and agent tests on the Fold 6.
4. Verify that real Qwen-generated tool calls execute through the Python agent and EmbeddingGemma retrieval pipeline.
5. Confirm correct behavior for direct answers, successful tool calls, malformed tool calls, and tool failures.
6. Measure model-switching overhead and total request latency.
7. Inspect Logcat for native crashes or runtime errors.
8. Report the verified results and any remaining problems.

If changes are required after device testing, make the corrections and rerun all available local tests first. Request the device again only when another on-device test is genuinely required.

### Important rules

- Never report simulated inference as verified on-device inference.
- Never modify, uninstall, or clear the existing application data without explicit approval.
- Do not rely on the phone being continuously connected.
- Do not introduce unnecessary device-dependent development steps.
- Preserve the last verified working APK and source state.
- If local tests cannot cover some native functionality, document that limitation rather than inventing a successful test.
- Stop at the device-testing checkpoint and wait for my response. Do not assume you can resume testing without my confirmation.

**Our objective is to minimize physical device interaction while preserving genuine end-to-end testing on Android.**
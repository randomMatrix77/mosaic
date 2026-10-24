package io.vibeos.intelligence.mnn

import android.util.Log
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

data class GenerationMetrics(
    val promptTokens: Long,
    val generatedTokens: Long,
    val prefillMicros: Long,
    val decodeMicros: Long,
    val visionMicros: Long,
    val status: Long,
)

interface NativeTokenCallback {
    fun onToken(text: String)
}

interface GenerationListener {
    fun onToken(text: String)
    fun onComplete(metrics: GenerationMetrics)
    fun onError(error: Throwable)
}

/**
 * Process-local owner for exactly one MNN LLM. This class deliberately has no UI or Capacitor
 * dependency, so the future plugin can use the same lifecycle without changing JNI ownership.
 */
class MnnRuntime(private val worker: ExecutorService = Executors.newSingleThreadExecutor()) : AutoCloseable {
    private var nativePointer = 0L
    private var generating = false
    private var closeRequested = false

    init {
        System.loadLibrary("MNN")
        System.loadLibrary("vibeos_mnn")
    }

    @Synchronized
    fun load(modelDirectory: File, config: RuntimeConfig = RuntimeConfig()): Result<Unit> = runCatching {
        check(!generating) { "Cannot switch models while generation is active." }
        unloadLocked()
        val validation = MnnModelPackageValidator.validate(modelDirectory)
        check(validation is ModelPackageValidation.Valid) {
            (validation as? ModelPackageValidation.Invalid)?.reason ?: "Invalid MNN model package."
        }
        nativePointer = nativeLoad(File(modelDirectory, "config.json").absolutePath, config.toMnnJson())
        check(nativePointer != 0L) { "MNN returned an empty native session." }
    }

    @Synchronized
    fun unload(): Result<Unit> = runCatching {
        check(!generating) { "Generation is active; cancel it before unloading." }
        unloadLocked()
    }

    @Synchronized
    fun isLoaded(): Boolean = nativePointer != 0L

    fun generate(prompt: String, maxNewTokens: Int, listener: GenerationListener) {
        val pointer = synchronized(this) {
            check(nativePointer != 0L) { "Load an MNN model before generating." }
            check(!generating) { "Generation is already active." }
            generating = true
            nativePointer
        }
        worker.execute {
            try {
                val values = nativeGenerate(pointer, prompt, maxNewTokens, object : NativeTokenCallback {
                    override fun onToken(text: String) = listener.onToken(text)
                })
                listener.onComplete(GenerationMetrics(
                    promptTokens = values[0], generatedTokens = values[1], prefillMicros = values[2],
                    decodeMicros = values[3], visionMicros = values[4], status = values[5]
                ))
            } catch (error: Throwable) {
                Log.e(TAG, "MNN generation failed", error)
                listener.onError(error)
            } finally {
                synchronized(this) {
                    generating = false
                    if (closeRequested) unloadLocked()
                }
            }
        }
    }

    @Synchronized
    fun cancel() {
        if (nativePointer != 0L && generating) nativeCancel(nativePointer)
    }

    @Synchronized
    override fun close() {
        if (generating) {
            closeRequested = true
            nativeCancel(nativePointer)
        } else unloadLocked()
        worker.shutdown()
    }

    @Synchronized
    private fun unloadLocked() {
        if (nativePointer != 0L) {
            nativeUnload(nativePointer)
            nativePointer = 0L
        }
    }

    private external fun nativeLoad(configPath: String, runtimeConfig: String): Long
    private external fun nativeGenerate(pointer: Long, prompt: String, maxNewTokens: Int, callback: NativeTokenCallback): LongArray
    private external fun nativeCancel(pointer: Long)
    private external fun nativeUnload(pointer: Long)

    private companion object { const val TAG = "vibeOS-MNN" }
}

data class RuntimeConfig(
    val threads: Int = 4,
    val maxNewTokens: Int = 512,
    val useMmap: Boolean = true,
) {
    fun toMnnJson(): String = """{"backend_type":"cpu","thread_num":$threads,"max_new_tokens":$maxNewTokens,"use_mmap":$useMmap}"""
}

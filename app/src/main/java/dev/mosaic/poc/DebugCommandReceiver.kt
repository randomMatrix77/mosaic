package dev.mosaic.poc

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Base64
import android.util.Log
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.launch
import java.io.File

/** Short debug requests may use this receiver; model requests use the Activity. */
class DebugCommandReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != ACTION_DEBUG_COMMAND) return
        val pendingResult = goAsync()
        DebugCommandRunner.scope.launch {
            try {
                DebugCommandRunner.run(context.applicationContext, DebugCommandRunner.decodeRequest(intent))
            } finally {
                pendingResult.finish()
            }
        }
    }

    companion object {
        const val ACTION_DEBUG_COMMAND = "dev.mosaic.poc.DEBUG_COMMAND"
        const val EXTRA_REQUEST_JSON = "request_json"
        const val EXTRA_REQUEST_B64 = "request_b64"
    }
}

/** Shared, restricted bridge used by debug-only ADB entry points. */
object DebugCommandRunner {
    private const val TAG = "MosaicDebug"
    val scope = kotlinx.coroutines.CoroutineScope(
        kotlinx.coroutines.SupervisorJob() + kotlinx.coroutines.Dispatchers.Default,
    )

    fun decodeRequest(intent: Intent): String {
        intent.getStringExtra(DebugCommandReceiver.EXTRA_REQUEST_JSON)?.let { return it }
        val encoded = intent.getStringExtra(DebugCommandReceiver.EXTRA_REQUEST_B64)
            ?: throw IllegalArgumentException("Provide request_json or request_b64.")
        return String(Base64.decode(encoded, Base64.DEFAULT), Charsets.UTF_8)
    }

    fun run(context: Context, request: String) {
        try {
            NativeLlama.initialize(context.applicationInfo.nativeLibraryDir)
            copyDebugImages(context)
            configurePythonStorage(context)
            val response = Python.getInstance().getModule("mosaic.debug")
                .callAttr("run_command_json", request).toString()
            writeResult(context, response)
            Log.i(TAG, "Debug command completed; read files/mosaic-debug/last-result.json with run-as.")
        } catch (error: Exception) {
            val response = """{"ok":false,"error":{"type":"${error.javaClass.simpleName}","message":${jsonString(error.message ?: "unknown error")}}}"""
            writeResult(context, response)
            Log.e(TAG, "Debug command failed; read files/mosaic-debug/last-result.json with run-as.", error)
        }
    }

    private fun configurePythonStorage(context: Context) {
        if (!Python.isStarted()) synchronized(Python::class.java) {
            if (!Python.isStarted()) Python.start(AndroidPlatform(context.applicationContext))
        }
        val database = File(context.filesDir, "mosaic/mosaic.sqlite3").apply { parentFile?.mkdirs() }
        val qwen = File(context.filesDir, "models/selected-qwen35-4b-q4_k_m.gguf")
        val embedding = File(context.filesDir, "models/selected-embeddinggemma-2.gguf")
        val projector = File(context.filesDir, "models/selected-mmproj-embeddinggemma-2.gguf")
        val qwenProjector = File(context.filesDir, "models/selected-qwen35-mmproj-f16.gguf")
        val debugImages = File(context.filesDir, "mosaic-debug/images")
        Python.getInstance().getModule("mosaic.core").callAttr(
            "configure_storage", database.absolutePath, qwen.absolutePath, embedding.absolutePath, projector.absolutePath,
            qwenProjector.absolutePath, debugImages.absolutePath, SecureSettings(context).deepSeekApiKey(),
        )
    }

    private fun copyDebugImages(context: Context) {
        val directory = File(context.filesDir, "mosaic-debug/images").apply { mkdirs() }
        for (name in listOf("cat.jpg", "car.jpg", "banana.jpg")) {
            val target = File(directory, name)
            if (!target.isFile) context.assets.open("retrieval/$name").use { input ->
                target.outputStream().use { output -> input.copyTo(output) }
            }
        }
    }

    private fun writeResult(context: Context, response: String) {
        val directory = File(context.filesDir, "mosaic-debug").apply { mkdirs() }
        val temporary = File(directory, "last-result.tmp")
        val destination = File(directory, "last-result.json")
        temporary.writeText(response, Charsets.UTF_8)
        if (!temporary.renameTo(destination)) {
            destination.writeText(response, Charsets.UTF_8)
            temporary.delete()
        }
    }

    private fun jsonString(value: String): String = buildString {
        append('"')
        value.forEach { character ->
            when (character) {
                '\\' -> append("\\\\")
                '"' -> append("\\\"")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> append(character)
            }
        }
        append('"')
    }
}

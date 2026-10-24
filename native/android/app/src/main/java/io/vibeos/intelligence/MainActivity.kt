package io.vibeos.intelligence

import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.documentfile.provider.DocumentFile
import io.vibeos.intelligence.mnn.GenerationListener
import io.vibeos.intelligence.mnn.GenerationMetrics
import io.vibeos.intelligence.mnn.MnnRuntime
import java.io.File
import java.util.concurrent.Executors

/** A deliberately small device probe for Milestone A. Capacitor becomes the user-facing shell in B. */
class MainActivity : AppCompatActivity() {
    private val io = Executors.newSingleThreadExecutor()
    private val runtime = MnnRuntime()
    private lateinit var status: TextView
    private lateinit var response: TextView
    private lateinit var prompt: EditText
    private lateinit var generate: Button

    private val selectModel = registerForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
        if (uri == null) return@registerForActivityResult
        contentResolver.takePersistableUriPermission(uri, android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
        status.text = "Importing MNN package into app-private storage…"
        io.execute {
            runCatching {
                val tree = requireNotNull(DocumentFile.fromTreeUri(this, uri))
                val destination = File(filesDir, "models/${System.currentTimeMillis()}").also { it.mkdirs() }
                copyTree(tree, destination)
                destination
            }.onSuccess { directory ->
                runtime.load(directory).fold(
                    onSuccess = { runOnUiThread { status.text = "MNN model loaded from ${directory.name}."; generate.isEnabled = true } },
                    onFailure = { error -> runOnUiThread { status.text = "Model load failed: ${error.message}" } }
                )
            }.onFailure { error -> runOnUiThread { status.text = "Import failed: ${error.message}" } }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        title = "vibeOS Intelligence — MNN probe"
        val padding = (20 * resources.displayMetrics.density).toInt()
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(padding, padding, padding, padding)
        }
        root.addView(Button(this).apply {
            text = "Import official MNN model package"
            setOnClickListener { selectModel.launch(null) }
        })
        status = TextView(this).apply { text = "No model loaded. Select an MNN-converted package containing config.json." }
        root.addView(status)
        prompt = EditText(this).apply {
            hint = "Ask the locally loaded model…"
            setText("Explain why local-first AI is useful in two sentences.")
            minLines = 3
        }
        root.addView(prompt, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        generate = Button(this).apply {
            text = "Generate locally"
            isEnabled = false
            setOnClickListener { generate() }
        }
        root.addView(generate)
        root.addView(Button(this).apply {
            text = "Stop"
            setOnClickListener { runtime.cancel(); status.text = "Cancellation requested…" }
        })
        response = TextView(this).apply { setTextIsSelectable(true) }
        root.addView(ScrollView(this).apply {
            addView(response)
        }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        setContentView(root)
    }

    private fun generate() {
        response.text = ""
        generate.isEnabled = false
        status.text = "Generating on-device…"
        runtime.generate(prompt.text.toString(), maxNewTokens = 256, object : GenerationListener {
            override fun onToken(text: String) = runOnUiThread { response.append(text) }
            override fun onComplete(metrics: GenerationMetrics) = runOnUiThread {
                status.text = "Complete: ${metrics.generatedTokens} tokens, prefill ${metrics.prefillMicros / 1_000} ms, decode ${metrics.decodeMicros / 1_000} ms, status ${metrics.status}."
                generate.isEnabled = true
            }
            override fun onError(error: Throwable) = runOnUiThread {
                status.text = "Generation failed: ${error.message}"
                generate.isEnabled = true
            }
        })
    }

    private fun copyTree(source: DocumentFile, destination: File) {
        if (source.isDirectory) {
            source.listFiles().forEach { child ->
                val childName = requireNotNull(child.name) { "Model package contains an unnamed document." }
                copyTree(child, File(destination, childName).also { if (child.isDirectory) it.mkdirs() })
            }
        } else {
            destination.parentFile?.mkdirs()
            contentResolver.openInputStream(requireNotNull(source.uri))!!.use { input ->
                destination.outputStream().use { output -> input.copyTo(output) }
            }
        }
    }

    override fun onDestroy() {
        runtime.close()
        io.shutdown()
        super.onDestroy()
    }
}


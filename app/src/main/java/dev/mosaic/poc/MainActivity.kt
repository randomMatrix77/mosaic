package dev.mosaic.poc

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.time.measureTimedValue

class MainActivity : ComponentActivity() {
    private enum class ImportKind { QWEN, EMBEDDING, PROJECTOR }
    private data class ConversationSummary(val id: String, val title: String?, val updatedAt: String)
    private data class ChatMessage(val role: String, val content: String, val status: String)

    private var message by mutableStateOf("")
    private var response by mutableStateOf("Create or open a conversation, then send a message to the Python backend.")
    private var isWorking by mutableStateOf(false)
    private var nativeStatus by mutableStateOf("Starting llama.cpp runtime…")
    private var pendingImport = ImportKind.QWEN
    private var conversations by mutableStateOf(emptyList<ConversationSummary>())
    private var activeConversationId by mutableStateOf<String?>(null)
    private var chatMessages by mutableStateOf(emptyList<ChatMessage>())
    private var conversationTitle by mutableStateOf("")

    private val modelPicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) importSelectedModel(uri.toString(), pendingImport)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        lifecycleScope.launch {
            nativeStatus = withContext(Dispatchers.Default) {
                runCatching {
                    NativeLlama.initialize(applicationInfo.nativeLibraryDir)
                    configurePythonStorage()
                    val qwen = File(filesDir, "models/selected-qwen35-4b-q4_k_m.gguf")
                    if (qwen.isFile) {
                        val load = NativeLlama.loadModel(qwen.absolutePath)
                        check(!load.startsWith("ERROR:")) { load }
                        "SQLite initialized; restored Qwen from app-private model storage: $load"
                    } else {
                        "llama.cpp runtime loaded; SQLite initialized in app-private storage; no model imported."
                    }
                }.getOrElse { "Startup error: ${it.message}" }
            }
            refreshConversations()
        }
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    Column(
                        modifier = Modifier.padding(24.dp).verticalScroll(rememberScrollState()),
                        verticalArrangement = Arrangement.spacedBy(12.dp),
                    ) {
                        Text("Mosaic — local Python persistence", style = MaterialTheme.typography.headlineSmall)
                        Text("Chat history and embeddings are stored by Python sqlite3. Models remain app-private files.")
                        Text(nativeStatus, style = MaterialTheme.typography.bodySmall)
                        OutlinedTextField(
                            modifier = Modifier.fillMaxWidth(), value = conversationTitle,
                            onValueChange = { conversationTitle = it }, label = { Text("Conversation title") },
                            enabled = !isWorking,
                        )
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(enabled = !isWorking, onClick = { createConversation() }) { Text("New conversation") }
                            Button(enabled = activeConversationId != null && !isWorking, onClick = { renameActiveConversation() }) { Text("Rename") }
                            Button(enabled = activeConversationId != null && !isWorking, onClick = { deleteActiveConversation() }) { Text("Delete") }
                            Button(enabled = !isWorking, onClick = { refreshConversations() }) { Text("Refresh") }
                        }
                        if (conversations.isEmpty()) {
                            Text("No saved conversations yet.", style = MaterialTheme.typography.bodySmall)
                        } else {
                            Text("Saved conversations", style = MaterialTheme.typography.titleSmall)
                            conversations.take(8).forEach { item ->
                                Button(enabled = !isWorking, onClick = { openConversation(item.id) }) {
                                    Text(if (item.id == activeConversationId) "Open: ${item.title ?: "Untitled"}" else item.title ?: "Untitled")
                                }
                            }
                        }
                        Button(enabled = !isWorking, onClick = {
                            pendingImport = ImportKind.QWEN
                            modelPicker.launch(arrayOf("application/octet-stream", "application/*", "*/*"))
                        }) { Text("Import Qwen GGUF") }
                        Button(enabled = !isWorking, onClick = {
                            pendingImport = ImportKind.EMBEDDING
                            modelPicker.launch(arrayOf("application/octet-stream", "application/*", "*/*"))
                        }) { Text("Import EmbeddingGemma GGUF") }
                        Button(enabled = !isWorking, onClick = {
                            pendingImport = ImportKind.PROJECTOR
                            modelPicker.launch(arrayOf("application/octet-stream", "application/*", "*/*"))
                        }) { Text("Import EmbeddingGemma projector") }
                        OutlinedTextField(
                            modifier = Modifier.fillMaxWidth(), value = message, onValueChange = { message = it },
                            label = { Text("Message") }, enabled = !isWorking,
                        )
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(enabled = message.isNotBlank() && activeConversationId != null && !isWorking,
                                onClick = { sendToPython(message) }) { Text("Send") }
                            Button(enabled = !isWorking, onClick = { runEmbeddingTextCheckpoint() }) { Text("Text embeddings") }
                            if (isWorking) CircularProgressIndicator()
                        }
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(enabled = !isWorking, onClick = { runImageRetrievalCheckpoint() }) { Text("Image retrieval") }
                            Button(enabled = !isWorking, onClick = { storeEmbeddingPersistenceCheckpoint() }) { Text("Store image vectors") }
                            Button(enabled = !isWorking, onClick = { loadEmbeddingPersistenceCheckpoint() }) { Text("Load saved vectors") }
                        }
                        if (chatMessages.isNotEmpty()) {
                            Text("Conversation", style = MaterialTheme.typography.titleMedium)
                            chatMessages.takeLast(8).forEach { item -> Text("${item.role}: ${item.content}") }
                        }
                        Text("Response", style = MaterialTheme.typography.titleMedium)
                        Text(response)
                    }
                }
            }
        }
    }

    private fun configurePythonStorage() {
        if (!Python.isStarted()) synchronized(Python::class.java) {
            if (!Python.isStarted()) Python.start(AndroidPlatform(applicationContext))
        }
        val database = File(filesDir, "mosaic/mosaic.sqlite3").apply { parentFile?.mkdirs() }
        Python.getInstance().getModule("mosaic.core").callAttr("configure_storage", database.absolutePath)
    }

    private fun core() = Python.getInstance().getModule("mosaic.core").callAttr("MosaicCore")

    private fun parseConversations(value: String): List<ConversationSummary> {
        val array = JSONArray(value)
        return List(array.length()) { index -> array.getJSONObject(index).let {
            ConversationSummary(it.getString("id"), if (it.isNull("title")) null else it.getString("title"), it.getString("updated_at"))
        } }
    }

    private fun parseMessages(value: String): List<ChatMessage> {
        val array = JSONArray(value)
        return List(array.length()) { index -> array.getJSONObject(index).let {
            ChatMessage(it.getString("role"), it.getString("content"), it.getString("status"))
        } }
    }

    private fun refreshConversations(after: ((List<ConversationSummary>) -> Unit)? = null) {
        lifecycleScope.launch {
            val result = withContext(Dispatchers.Default) { runCatching {
                configurePythonStorage()
                parseConversations(core().callAttr("list_conversations_json").toString())
            } }
            result.onSuccess { list -> conversations = list; after?.invoke(list) }
                .onFailure { response = "Conversation list error: ${it.message}" }
        }
    }

    private fun createConversation() {
        isWorking = true
        lifecycleScope.launch {
            val result = withContext(Dispatchers.Default) { runCatching {
                configurePythonStorage()
                JSONObject(core().callAttr("create_conversation_json", conversationTitle).toString()).getString("id")
            } }
            result.onSuccess { id ->
                conversationTitle = ""; activeConversationId = id; response = "Created conversation $id"
                refreshConversations { openConversation(id) }
            }.onFailure { response = "Create conversation error: ${it.message}" }
            isWorking = false
        }
    }

    private fun renameActiveConversation() {
        val id = activeConversationId ?: return
        isWorking = true
        lifecycleScope.launch {
            val result = withContext(Dispatchers.Default) { runCatching {
                configurePythonStorage(); core().callAttr("rename_conversation", id, conversationTitle); Unit
            } }
            result.onSuccess { response = "Conversation renamed."; refreshConversations() }
                .onFailure { response = "Rename error: ${it.message}" }
            isWorking = false
        }
    }

    private fun deleteActiveConversation() {
        val id = activeConversationId ?: return
        isWorking = true
        lifecycleScope.launch {
            val result = withContext(Dispatchers.Default) { runCatching {
                configurePythonStorage(); core().callAttr("delete_conversation", id); Unit
            } }
            result.onSuccess {
                activeConversationId = null; chatMessages = emptyList(); response = "Conversation deleted."
                refreshConversations()
            }.onFailure { response = "Delete error: ${it.message}" }
            isWorking = false
        }
    }

    private fun openConversation(id: String) {
        isWorking = true
        lifecycleScope.launch {
            val result = withContext(Dispatchers.Default) { runCatching {
                configurePythonStorage(); parseMessages(core().callAttr("get_messages_json", id).toString())
            } }
            result.onSuccess {
                activeConversationId = id; chatMessages = it
                conversationTitle = conversations.firstOrNull { item -> item.id == id }?.title ?: ""
                response = "Opened saved conversation with ${it.size} messages."
            }.onFailure { response = "Open conversation error: ${it.message}" }
            isWorking = false
        }
    }

    private fun sendToPython(input: String) {
        val conversationId = activeConversationId ?: return
        isWorking = true; response = "Python is saving the user turn and running local Qwen…"
        lifecycleScope.launch {
            val result = withContext(Dispatchers.Default) { runCatching {
                val timed = measureTimedValue {
                    configurePythonStorage()
                    core().callAttr("process_message", conversationId, input).toString()
                }
                "Python response (${timed.duration.inWholeMilliseconds} ms):\n${timed.value}"
            } }
            result.onSuccess { response = it; message = ""; openConversation(conversationId); refreshConversations() }
                .onFailure { response = "Python error: ${it.message ?: it::class.java.name}\n${it.stackTraceToString()}" }
            isWorking = false
        }
    }

    private fun importSelectedModel(uriText: String, kind: ImportKind) {
        isWorking = true; nativeStatus = "Copying selected model into app-private storage…"
        lifecycleScope.launch {
            val status = withContext(Dispatchers.IO) { runCatching {
                val filename = when (kind) {
                    ImportKind.QWEN -> "selected-qwen35-4b-q4_k_m.gguf"
                    ImportKind.EMBEDDING -> "selected-embeddinggemma-2.gguf"
                    ImportKind.PROJECTOR -> "selected-mmproj-embeddinggemma-2.gguf"
                }
                val destination = File(filesDir, "models/$filename"); destination.parentFile?.mkdirs()
                contentResolver.openInputStream(android.net.Uri.parse(uriText))?.use { input ->
                    destination.outputStream().use { output -> input.copyTo(output) }
                } ?: error("Android could not open the selected model.")
                val load = measureTimedValue { when (kind) {
                    ImportKind.QWEN -> NativeLlama.loadModel(destination.absolutePath)
                    ImportKind.EMBEDDING -> NativeLlama.loadEmbeddingModel(destination.absolutePath)
                    ImportKind.PROJECTOR -> NativeLlama.loadEmbeddingProjector(destination.absolutePath)
                } }
                check(!load.value.startsWith("ERROR:")) { load.value }
                "Model loaded in ${load.duration.inWholeMilliseconds} ms from app-private storage."
            }.getOrElse { "Model import/load error: ${it.message}" } }
            nativeStatus = status; isWorking = false
        }
    }

    private fun runEmbeddingTextCheckpoint() = runDiagnostic("Calling Python EmbeddingGemma text checkpoint…") {
        Python.getInstance().getModule("mosaic.embedding_diagnostics").callAttr("run_text_checkpoint").toString()
    }

    private fun runImageRetrievalCheckpoint() = runDiagnostic("Copying test images and running retrieval…") {
        val paths = copyRetrievalAssets()
        Python.getInstance().getModule("mosaic.embedding_diagnostics")
            .callAttr("run_image_retrieval_checkpoint", paths[0], paths[1], paths[2]).toString()
    }

    private fun storeEmbeddingPersistenceCheckpoint() = runDiagnostic("Generating/storing real image vectors in SQLite…") {
        val paths = copyRetrievalAssets()
        Python.getInstance().getModule("mosaic.embedding_diagnostics")
            .callAttr("store_image_embeddings_checkpoint", paths[0], paths[1], paths[2]).toString()
    }

    private fun loadEmbeddingPersistenceCheckpoint() = runDiagnostic("Loading saved SQLite image vectors; image embeddings will not be regenerated…") {
        Python.getInstance().getModule("mosaic.embedding_diagnostics")
            .callAttr("run_persisted_image_retrieval_checkpoint").toString()
    }

    private fun runDiagnostic(start: String, work: () -> String) {
        isWorking = true; response = start
        lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) { runCatching { configurePythonStorage(); work() } }
            response = result.getOrElse { "Diagnostic error: ${it.message ?: it::class.java.name}\n${it.stackTraceToString()}" }
            isWorking = false
        }
    }

    private fun copyRetrievalAssets(): Array<String> {
        val imagesDir = File(filesDir, "retrieval-test-images").apply { mkdirs() }
        return arrayOf("cat.jpg", "car.jpg", "banana.jpg").map { name ->
            val target = File(imagesDir, name)
            if (!target.exists()) assets.open("retrieval/$name").use { input ->
                target.outputStream().use { output -> input.copyTo(output) }
            }
            target.absolutePath
        }.toTypedArray()
    }
}

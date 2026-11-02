package dev.mosaic.poc

import android.content.Context
import android.net.Uri
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.UUID

/**
 * Kotlin's narrow UI boundary for the Python-owned Mosaic application.
 *
 * This class deliberately doesn't retain a second database or inference
 * engine. Conversations, messages, model requests, tools, and persistence all
 * remain in the established Chaquopy/Python/JNI stack.
 */
class MosaicRepository(context: Context) {
    private val appContext = context.applicationContext
    private val secureSettings = SecureSettings(appContext)

    suspend fun initialize(): RuntimeStatus = withContext(Dispatchers.Default) {
        NativeLlama.initialize(appContext.applicationInfo.nativeLibraryDir)
        configurePython()
        val qwen = modelFile(QWEN_FILENAME)
        if (!qwen.isFile) {
            return@withContext RuntimeStatus.Unavailable("Qwen model not imported")
        }
        val outcome = NativeLlama.loadModel(qwen.absolutePath)
        if (outcome.startsWith("ERROR:")) RuntimeStatus.Error(outcome.removePrefix("ERROR:").trim())
        else RuntimeStatus.Ready(if (outcome.contains("GPU", ignoreCase = true)) "On-device · GPU ready" else "On-device · ready")
    }

    suspend fun listConversations(): List<ConversationSummary> = withContext(Dispatchers.Default) {
        configurePython()
        val values = JSONArray(core().callAttr("list_conversations_json").toString())
        List(values.length()) { index -> values.getJSONObject(index).toConversationSummary() }
    }

    suspend fun customCapabilities(): List<CustomCapability> = withContext(Dispatchers.Default) {
        configurePython()
        val values = JSONArray(core().callAttr("list_custom_capabilities_json").toString())
        List(values.length()) { index -> values.getJSONObject(index).toCustomCapability() }
    }

    suspend fun builtInSkills(): List<BuiltInSkill> = withContext(Dispatchers.Default) {
        configurePython()
        val values = JSONArray(core().callAttr("list_built_in_skills_json").toString())
        List(values.length()) { index -> values.getJSONObject(index).toBuiltInSkill() }
    }

    suspend fun updateBuiltInSkill(key: String, instructions: String): BuiltInSkill = withContext(Dispatchers.Default) {
        configurePython()
        JSONObject(core().callAttr("update_built_in_skill_json", key, instructions).toString()).toBuiltInSkill()
    }

    suspend fun resetBuiltInSkill(key: String) = withContext(Dispatchers.Default) {
        configurePython()
        core().callAttr("reset_built_in_skill", key)
    }

    suspend fun createCustomCapability(title: String, description: String): CustomCapability = withContext(Dispatchers.Default) {
        configurePython()
        JSONObject(core().callAttr("create_custom_capability_json", title, description).toString()).toCustomCapability()
    }

    suspend fun updateCustomCapability(id: String, title: String, description: String): CustomCapability = withContext(Dispatchers.Default) {
        configurePython()
        JSONObject(core().callAttr("update_custom_capability_json", id, title, description).toString()).toCustomCapability()
    }

    suspend fun deleteCustomCapability(capabilityId: String) = withContext(Dispatchers.Default) {
        configurePython()
        core().callAttr("delete_custom_capability", capabilityId)
    }

    suspend fun messages(conversationId: String): List<ChatMessage> = withContext(Dispatchers.Default) {
        configurePython()
        val values = JSONArray(core().callAttr("get_messages_json", conversationId).toString())
        List(values.length()) { index -> values.getJSONObject(index).toChatMessage() }
    }

    suspend fun createConversation(title: String? = null): ConversationSummary = withContext(Dispatchers.Default) {
        configurePython()
        JSONObject(core().callAttr("create_conversation_json", title).toString()).toConversationSummary()
    }

    suspend fun renameConversation(conversationId: String, title: String?): ConversationSummary = withContext(Dispatchers.Default) {
        configurePython()
        JSONObject(core().callAttr("rename_conversation", conversationId, title).toString()).toConversationSummary()
    }

    suspend fun deleteConversation(conversationId: String) = withContext(Dispatchers.Default) {
        configurePython()
        core().callAttr("delete_conversation", conversationId)
    }

    suspend fun memoryClaims(): List<MemoryClaimSummary> = withContext(Dispatchers.Default) {
        configurePython()
        val values = JSONArray(core().callAttr("list_memory_claims_json").toString())
        List(values.length()) { index -> values.getJSONObject(index).toMemoryClaimSummary() }
    }

    suspend fun memoryClaim(claimId: String): MemoryClaimDetail = withContext(Dispatchers.Default) {
        configurePython()
        JSONObject(core().callAttr("get_memory_claim_json", claimId).toString()).toMemoryClaimDetail()
    }

    suspend fun deleteMemoryClaim(claimId: String) = withContext(Dispatchers.Default) {
        configurePython()
        core().callAttr("delete_memory_claim", claimId)
    }

    suspend fun indexedImages(): List<IndexedImage> = withContext(Dispatchers.Default) {
        configurePython()
        val values = JSONArray(core().callAttr("list_indexed_images_json").toString())
        List(values.length()) { index -> values.getJSONObject(index).toIndexedImage() }
    }

    suspend fun searchIndexedImages(query: String): ImageSearchResult = withContext(Dispatchers.Default) {
        configurePython()
        val value = JSONObject(core().callAttr("search_indexed_images_json", query, 10).toString())
        val images = value.getJSONArray("results").let { values ->
            List(values.length()) { index -> values.getJSONObject(index).toIndexedImage() }
        }
        ImageSearchResult(images, value.optDouble("elapsed_ms", 0.0))
    }

    /** Import a Photo Picker image into app-private storage, then embed it in Python. */
    suspend fun indexImage(uri: Uri): IndexedImage = withContext(Dispatchers.IO) {
        val imported = importIndexedImage(uri)
        try {
            configurePython()
            JSONObject(core().callAttr(
                "index_image_json", imported.file.absolutePath, imported.mimeType,
            ).toString()).toIndexedImage()
        } catch (error: Throwable) {
            imported.file.delete()
            throw error
        }
    }

    suspend fun localStatus(): LocalStatus = withContext(Dispatchers.Default) {
        configurePython()
        JSONObject(core().callAttr("local_status_json").toString()).toLocalStatus(secureSettings.deepSeekApiKey() != null)
    }

    suspend fun saveDeepSeekApiKey(value: String) = withContext(Dispatchers.Default) {
        secureSettings.saveDeepSeekApiKey(value)
    }

    suspend fun clearDeepSeekApiKey() = withContext(Dispatchers.Default) {
        secureSettings.clearDeepSeekApiKey()
    }

    suspend fun streamMessage(conversationId: String, text: String, onEvent: (RuntimeEvent) -> Unit): RequestResult =
        withContext(Dispatchers.Default) {
            configurePython()
            val sink = PythonEventSink { serialized -> RuntimeEvent.parse(serialized)?.let(onEvent) }
            val result = JSONObject(core().callAttr("process_message_stream", conversationId, text, sink).toString())
            RequestResult(
                requestId = result.optString("request_id").ifBlank { null },
                cancelled = result.optString("status") == "cancelled",
            )
        }

    suspend fun streamImageMessage(conversationId: String, text: String, image: Uri, onEvent: (RuntimeEvent) -> Unit): RequestResult =
        withContext(Dispatchers.IO) {
            configurePython()
            val imageFile = importVisionImage(image)
            val sink = PythonEventSink { serialized -> RuntimeEvent.parse(serialized)?.let(onEvent) }
            val result = JSONObject(core().callAttr(
                "process_image_message_stream", conversationId, text, imageFile.absolutePath, sink,
            ).toString())
            RequestResult(
                requestId = result.optString("request_id").ifBlank { null },
                cancelled = result.optString("status") == "cancelled",
            )
        }

    /** Vision for a source the user explicitly selected from Mosaic's SQLite image index. */
    suspend fun streamStoredImageMessage(conversationId: String, text: String, sourceId: String, onEvent: (RuntimeEvent) -> Unit): RequestResult =
        withContext(Dispatchers.Default) {
            configurePython()
            val sink = PythonEventSink { serialized -> RuntimeEvent.parse(serialized)?.let(onEvent) }
            val result = JSONObject(core().callAttr(
                "process_indexed_image_message_stream", conversationId, text, sourceId, sink,
            ).toString())
            RequestResult(
                requestId = result.optString("request_id").ifBlank { null },
                cancelled = result.optString("status") == "cancelled",
            )
        }

    fun cancelActiveGeneration() {
        NativeLlama.cancelGeneration()
    }

    suspend fun importQwen(uri: Uri): RuntimeStatus = withContext(Dispatchers.IO) {
        val destination = modelFile(QWEN_FILENAME).apply { parentFile?.mkdirs() }
        appContext.contentResolver.openInputStream(uri)?.use { input ->
            destination.outputStream().use { output -> input.copyTo(output) }
        } ?: return@withContext RuntimeStatus.Error("Android could not open the selected model.")
        val outcome = NativeLlama.loadModel(destination.absolutePath)
        if (outcome.startsWith("ERROR:")) RuntimeStatus.Error(outcome.removePrefix("ERROR:").trim())
        else RuntimeStatus.Ready(if (outcome.contains("GPU", ignoreCase = true)) "On-device · GPU ready" else "On-device · ready")
    }

    private fun configurePython() {
        if (!Python.isStarted()) synchronized(Python::class.java) {
            if (!Python.isStarted()) Python.start(AndroidPlatform(appContext))
        }
        val database = File(appContext.filesDir, "mosaic/mosaic.sqlite3").apply { parentFile?.mkdirs() }
        coreModule().callAttr(
            "configure_storage", database.absolutePath,
            modelFile(QWEN_FILENAME).absolutePath,
            modelFile(EMBEDDING_FILENAME).absolutePath,
            modelFile(PROJECTOR_FILENAME).absolutePath,
            modelFile(QWEN_PROJECTOR_FILENAME).absolutePath,
            File(appContext.filesDir, "mosaic-debug/images").absolutePath,
            secureSettings.deepSeekApiKey(),
        )
    }

    private fun coreModule() = Python.getInstance().getModule("mosaic.core")
    private fun core() = coreModule().callAttr("MosaicCore")
    private fun modelFile(name: String) = File(appContext.filesDir, "models/$name")

    private fun importVisionImage(uri: Uri): File {
        val attachments = File(appContext.filesDir, "mosaic-debug/images").apply { mkdirs() }
        val mime = appContext.contentResolver.getType(uri)
        val extension = when (mime) {
            "image/jpeg" -> "jpg"
            "image/png" -> "png"
            "image/webp" -> "webp"
            else -> "img"
        }
        val destination = File(attachments, "vision-${UUID.randomUUID()}.$extension")
        var total = 0L
        try {
            appContext.contentResolver.openInputStream(uri)?.use { input ->
                destination.outputStream().use { output ->
                    val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                    while (true) {
                        val read = input.read(buffer)
                        if (read < 0) break
                        total += read
                        if (total > MAX_VISION_IMAGE_BYTES) throw IllegalArgumentException("Selected image is larger than 20 MB.")
                        output.write(buffer, 0, read)
                    }
                }
            } ?: throw IllegalArgumentException("Android could not open the selected image.")
        } catch (error: Throwable) {
            destination.delete()
            throw error
        }
        return destination
    }

    private data class ImportedImage(val file: File, val mimeType: String?)

    private fun importIndexedImage(uri: Uri): ImportedImage {
        val directory = File(appContext.filesDir, "mosaic/images").apply { mkdirs() }
        val mime = appContext.contentResolver.getType(uri)
        val extension = when (mime) {
            "image/jpeg" -> "jpg"
            "image/png" -> "png"
            "image/webp" -> "webp"
            else -> "img"
        }
        val destination = File(directory, "image-${UUID.randomUUID()}.$extension")
        var total = 0L
        try {
            appContext.contentResolver.openInputStream(uri)?.use { input ->
                destination.outputStream().use { output ->
                    val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                    while (true) {
                        val read = input.read(buffer)
                        if (read < 0) break
                        total += read
                        if (total > MAX_VISION_IMAGE_BYTES) throw IllegalArgumentException("Selected image is larger than 20 MB.")
                        output.write(buffer, 0, read)
                    }
                }
            } ?: throw IllegalArgumentException("Android could not open the selected image.")
        } catch (error: Throwable) {
            destination.delete()
            throw error
        }
        return ImportedImage(destination, mime)
    }

    private fun JSONObject.toConversationSummary() = ConversationSummary(
        id = getString("id"),
        title = optString("title").takeIf { !isNull("title") && it.isNotBlank() },
        updatedAt = getString("updated_at"),
    )

    private fun JSONObject.toCustomCapability() = CustomCapability(
        id = getString("id"),
        title = getString("title"),
        description = getString("description"),
        createdAt = getString("created_at"),
        updatedAt = getString("updated_at"),
    )

    private fun JSONObject.toBuiltInSkill() = BuiltInSkill(
        key = getString("key"),
        title = getString("title"),
        instructions = getString("instructions"),
        version = getString("version"),
        allowedTools = getJSONArray("allowed_tools").let { values ->
            List(values.length()) { index -> values.getString(index) }
        },
        hasLocalOverride = optBoolean("has_local_override", false),
    )

    private fun JSONObject.toChatMessage() = ChatMessage(
        id = getString("id"),
        conversationId = getString("conversation_id"),
        role = getString("role"),
        content = getString("content"),
        status = getString("status"),
        kind = getString("kind"),
        createdAt = getString("created_at"),
    )

    private fun JSONObject.toMemoryClaimSummary() = MemoryClaimSummary(
        id = getString("id"),
        fact = getString("fact"),
        subject = getString("subject"),
        subjectType = getString("subject_type"),
        predicate = getString("predicate"),
        status = getString("status"),
        assertionType = getString("assertion_type"),
        evidenceCount = getInt("evidence_count"),
        createdAt = getString("created_at"),
        updatedAt = getString("updated_at"),
    )

    private fun JSONObject.toMemoryClaimDetail() = MemoryClaimDetail(
        summary = toMemoryClaimSummary().copy(evidenceCount = getJSONArray("evidence").length()),
        evidence = getJSONArray("evidence").let { values ->
            List(values.length()) { index -> values.getJSONObject(index).toEvidenceItem() }
        },
    )

    private fun JSONObject.toEvidenceItem() = EvidenceItem(
        id = getString("id"),
        conversationId = getString("conversation_id"),
        conversationTitle = optString("conversation_title").takeIf { !isNull("conversation_title") && it.isNotBlank() },
        messageId = getString("message_id"),
        messageRole = optString("message_role").takeIf { !isNull("message_role") && it.isNotBlank() },
        messageCreatedAt = optString("message_created_at").takeIf { !isNull("message_created_at") && it.isNotBlank() },
        quote = optString("quote").takeIf { !isNull("quote") && it.isNotBlank() },
        messageContent = optString("message_content").takeIf { !isNull("message_content") && it.isNotBlank() },
        spanAvailable = optBoolean("span_available", false),
        unavailableReason = optString("unavailable_reason").takeIf { !isNull("unavailable_reason") && it.isNotBlank() },
    )

    private fun JSONObject.toIndexedImage() = IndexedImage(
        id = getString("id"),
        filename = getString("filename"),
        path = optString("path").takeIf { !isNull("path") && it.isNotBlank() },
        available = optBoolean("available", false),
        mimeType = optString("mime_type").takeIf { !isNull("mime_type") && it.isNotBlank() },
        origin = optString("origin", "Mosaic indexed image"),
        similarity = takeIf { has("similarity") && !isNull("similarity") }?.optDouble("similarity"),
    )

    private fun JSONObject.toLocalStatus(webSearchConfigured: Boolean) = LocalStatus(
        databaseBytes = optLong("database_bytes", 0L),
        conversations = optInt("conversations", 0),
        memories = optInt("memories", 0),
        indexedImages = optInt("indexed_images", 0),
        models = getJSONObject("models").let { models ->
            listOf("qwen", "embeddinggemma", "embedding_projector", "qwen_projector").associateWith { key ->
                models.getJSONObject(key).let { ModelFileStatus(it.optBoolean("present", false), it.optLong("bytes", 0L)) }
            }
        },
        webSearchConfigured = webSearchConfigured,
    )

    companion object {
        private const val QWEN_FILENAME = "selected-qwen35-4b-q4_k_m.gguf"
        private const val EMBEDDING_FILENAME = "selected-embeddinggemma-2.gguf"
        private const val PROJECTOR_FILENAME = "selected-mmproj-embeddinggemma-2.gguf"
        private const val QWEN_PROJECTOR_FILENAME = "selected-qwen35-mmproj-f16.gguf"
        private const val MAX_VISION_IMAGE_BYTES = 20L * 1024L * 1024L
    }
}

data class ConversationSummary(val id: String, val title: String?, val updatedAt: String)
data class CustomCapability(
    val id: String,
    val title: String,
    val description: String,
    val createdAt: String,
    val updatedAt: String,
)
data class BuiltInSkill(
    val key: String,
    val title: String,
    val instructions: String,
    val version: String,
    val allowedTools: List<String>,
    val hasLocalOverride: Boolean,
)

data class ChatMessage(
    val id: String,
    val conversationId: String,
    val role: String,
    val content: String,
    val status: String,
    val kind: String,
    val createdAt: String,
)

data class MemoryClaimSummary(
    val id: String,
    val fact: String,
    val subject: String,
    val subjectType: String,
    val predicate: String,
    val status: String,
    val assertionType: String,
    val evidenceCount: Int,
    val createdAt: String,
    val updatedAt: String,
)

data class MemoryClaimDetail(val summary: MemoryClaimSummary, val evidence: List<EvidenceItem>)

data class EvidenceItem(
    val id: String,
    val conversationId: String,
    val conversationTitle: String?,
    val messageId: String,
    val messageRole: String?,
    val messageCreatedAt: String?,
    val quote: String?,
    val messageContent: String?,
    val spanAvailable: Boolean,
    val unavailableReason: String?,
)

data class IndexedImage(
    val id: String,
    val filename: String,
    val path: String?,
    val available: Boolean,
    val mimeType: String?,
    val origin: String,
    val similarity: Double?,
)

data class ImageSearchResult(val images: List<IndexedImage>, val elapsedMs: Double)
data class ModelFileStatus(val present: Boolean, val bytes: Long)
data class LocalStatus(
    val databaseBytes: Long,
    val conversations: Int,
    val memories: Int,
    val indexedImages: Int,
    val models: Map<String, ModelFileStatus>,
    val webSearchConfigured: Boolean,
)

sealed interface RuntimeStatus {
    data object Starting : RuntimeStatus
    data class Ready(val label: String) : RuntimeStatus
    data class Unavailable(val message: String) : RuntimeStatus
    data class Error(val message: String) : RuntimeStatus
}

data class RequestResult(val requestId: String?, val cancelled: Boolean)

data class RuntimeEvent(
    val requestId: String,
    val sequence: Long,
    val timestampMs: Long,
    val type: String,
    val displayText: String?,
    val model: String?,
    val tool: String?,
    val resultCount: Int?,
    val sourceSummaries: List<SourceSummary>,
    val delta: String?,
    val errorMessage: String?,
) {
    companion object {
        fun parse(serialized: String): RuntimeEvent? = runCatching {
            val event = JSONObject(serialized)
            RuntimeEvent(
                requestId = event.getString("request_id"),
                sequence = event.optLong("sequence", -1),
                timestampMs = event.optLong("timestamp_ms", System.currentTimeMillis()),
                type = event.getString("type"),
                displayText = event.optString("display_text").ifBlank {
                    event.optString("display_name").ifBlank { null }
                },
                model = event.optString("model").ifBlank { null },
                tool = event.optString("tool").ifBlank { null },
                resultCount = event.takeIf { it.has("result_count") }?.optInt("result_count"),
                sourceSummaries = event.optJSONArray("source_summaries")?.let { sources ->
                    List(sources.length()) { index -> sources.getJSONObject(index).let { source ->
                        SourceSummary(
                            source.getString("kind"), source.getString("id"), source.getString("label"),
                            source.optString("url").takeIf { value -> value.isNotBlank() },
                        )
                    } }
                } ?: emptyList(),
                delta = event.optString("delta").ifBlank { null },
                errorMessage = event.optString("message").ifBlank { null },
            )
        }.getOrNull()
    }
}

data class SourceSummary(val kind: String, val id: String, val label: String, val url: String? = null)

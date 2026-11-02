package dev.mosaic.poc

import android.net.Uri
import android.os.SystemClock
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.util.UUID

/** Lifecycle owner for the single native request slot and durable chat ledger. */
class MosaicViewModel(private val repository: MosaicRepository) : ViewModel() {
    private val _state = MutableStateFlow(MosaicUiState())
    val state: StateFlow<MosaicUiState> = _state.asStateFlow()

    init {
        refreshStartup()
    }

    fun refreshStartup() = viewModelScope.launch(Dispatchers.IO) {
        val runtime = runCatching { repository.initialize() }
            .getOrElse { RuntimeStatus.Error(it.message ?: "Unable to initialize local runtime") }
        val conversations = runCatching { repository.listConversations() }.getOrElse { emptyList() }
        val localStatus = runCatching { repository.localStatus() }.getOrNull()
        _state.update { it.copy(runtimeStatus = runtime, conversations = conversations, localStatus = localStatus, startupComplete = true) }
    }

    fun selectConversation(conversationId: String?) {
        if (conversationId == null) {
            _state.update { it.copy(screen = MosaicScreen.Chat, activeConversationId = null, messages = emptyList(), highlightedMessageId = null, transientError = null) }
            return
        }
        viewModelScope.launch(Dispatchers.IO) {
            val messages = runCatching { repository.messages(conversationId) }
            _state.update {
                messages.fold(
                onSuccess = { loaded -> it.copy(screen = MosaicScreen.Chat, activeConversationId = conversationId, messages = loaded, transientError = null) },
                    onFailure = { error -> it.copy(transientError = error.message ?: "Could not open conversation") },
                )
            }
        }
    }

    fun showChat() = _state.update { it.copy(screen = MosaicScreen.Chat, selectedMemory = null) }

    fun showHome() = _state.update { it.copy(screen = MosaicScreen.Home, selectedMemory = null) }

    fun showConversations() = _state.update { it.copy(screen = MosaicScreen.Conversations, selectedMemory = null) }

    fun showMemories() {
        _state.update { it.copy(screen = MosaicScreen.Memories, selectedMemory = null, memoryError = null) }
        refreshMemories()
    }

    fun showImages() {
        _state.update { it.copy(screen = MosaicScreen.Images, imageError = null) }
        refreshIndexedImages()
    }

    fun openIndexedImage(image: IndexedImage) = _state.update {
        it.copy(screen = MosaicScreen.ImageDetail, selectedIndexedImage = image, imageError = null)
    }

    fun askAboutIndexedImage() {
        val image = _state.value.selectedIndexedImage ?: return
        _state.update { state ->
            state.copy(
                screen = MosaicScreen.Chat,
                selectedImageSource = image,
                drafts = state.drafts + (state.draftKey to "What is happening in this image?"),
            )
        }
    }

    fun refreshIndexedImages() = viewModelScope.launch(Dispatchers.IO) {
        _state.update { it.copy(imagesLoading = true, imageIndexing = false, imageError = null, imageNotice = null, imageSearchQuery = null) }
        val outcome = runCatching { repository.indexedImages() }
        _state.update { state -> outcome.fold(
            onSuccess = { images -> state.copy(indexedImages = images, imagesLoading = false) },
            onFailure = { error -> state.copy(imagesLoading = false, imageError = error.message ?: "Could not load indexed images") },
        ) }
    }

    fun searchIndexedImages(query: String) = viewModelScope.launch(Dispatchers.IO) {
        if (query.isBlank()) { refreshIndexedImages(); return@launch }
        _state.update { it.copy(imagesLoading = true, imageIndexing = false, imageError = null, imageNotice = null, imageSearchQuery = query) }
        val outcome = runCatching { repository.searchIndexedImages(query) }
        _state.update { state -> outcome.fold(
            onSuccess = { result -> state.copy(indexedImages = result.images, imagesLoading = false, imageSearchElapsedMs = result.elapsedMs) },
            onFailure = { error -> state.copy(imagesLoading = false, imageError = error.message ?: "Image search failed") },
        ) }
    }

    fun showSkills() {
        _state.update { it.copy(screen = MosaicScreen.Skills, capabilityError = null) }
        refreshCustomCapabilities()
    }

    fun updateCapabilityTitle(value: String) = _state.update { it.copy(capabilityTitleDraft = value) }

    fun updateCapabilityDescription(value: String) = _state.update { it.copy(capabilityDescriptionDraft = value) }

    fun editCustomCapability(capability: CustomCapability) = _state.update {
        it.copy(
            editingCapabilityId = capability.id,
            editingBuiltInSkillKey = null,
            capabilityTitleDraft = capability.title,
            capabilityDescriptionDraft = capability.description,
            capabilityError = null,
            capabilityNotice = null,
        )
    }

    fun editBuiltInSkill(skill: BuiltInSkill) = _state.update {
        it.copy(
            editingCapabilityId = null,
            editingBuiltInSkillKey = skill.key,
            capabilityTitleDraft = skill.title,
            capabilityDescriptionDraft = skill.instructions,
            capabilityError = null,
            capabilityNotice = null,
        )
    }

    fun cancelCapabilityEdit() = _state.update {
        it.copy(
            editingCapabilityId = null,
            editingBuiltInSkillKey = null,
            capabilityTitleDraft = "",
            capabilityDescriptionDraft = "",
            capabilityError = null,
        )
    }

    fun refreshCustomCapabilities() = viewModelScope.launch(Dispatchers.IO) {
        _state.update { it.copy(capabilitiesLoading = true, capabilityError = null) }
        val custom = runCatching { repository.customCapabilities() }
        val builtIn = runCatching { repository.builtInSkills() }
        _state.update { state ->
            val failure = custom.exceptionOrNull() ?: builtIn.exceptionOrNull()
            if (failure != null) state.copy(
                capabilitiesLoading = false,
                capabilityError = failure.message ?: "Could not load skills.",
            ) else state.copy(
                customCapabilities = custom.getOrThrow(),
                builtInSkills = builtIn.getOrThrow(),
                capabilitiesLoading = false,
            )
        }
    }

    fun addCustomCapability() {
        val snapshot = _state.value
        val title = snapshot.capabilityTitleDraft.trim()
        val description = snapshot.capabilityDescriptionDraft.trim()
        if (title.isEmpty() || description.isEmpty()) {
            _state.update { it.copy(capabilityError = "Enter both a capability name and a description.") }
            return
        }
        viewModelScope.launch(Dispatchers.IO) {
            _state.update { it.copy(capabilitySaving = true, capabilityError = null, capabilityNotice = null) }
            val builtInKey = snapshot.editingBuiltInSkillKey
            if (builtInKey != null) {
                val outcome = runCatching { repository.updateBuiltInSkill(builtInKey, description) }
                _state.update { state -> outcome.fold(
                    onSuccess = { skill -> state.copy(
                        builtInSkills = state.builtInSkills.map { if (it.key == skill.key) skill else it },
                        editingBuiltInSkillKey = null,
                        capabilityTitleDraft = "",
                        capabilityDescriptionDraft = "",
                        capabilitySaving = false,
                        capabilityNotice = "Updated ${skill.title}. The local task will use it.",
                    ) },
                    onFailure = { error -> state.copy(
                        capabilitySaving = false,
                        capabilityError = error.message ?: "Could not update the built-in skill.",
                    ) },
                ) }
                return@launch
            }
            val outcome = runCatching {
                snapshot.editingCapabilityId?.let { repository.updateCustomCapability(it, title, description) }
                    ?: repository.createCustomCapability(title, description)
            }
            _state.update { state -> outcome.fold(
                onSuccess = { capability -> state.copy(
                    customCapabilities = listOf(capability) + state.customCapabilities.filterNot { it.id == capability.id },
                    editingCapabilityId = null,
                    capabilityTitleDraft = "",
                    capabilityDescriptionDraft = "",
                    capabilitySaving = false,
                    capabilityNotice = if (snapshot.editingCapabilityId == null) {
                        "Saved ${capability.title}. It will guide future local Qwen requests."
                    } else {
                        "Updated ${capability.title}. Future local Qwen requests will use it."
                    },
                ) },
                onFailure = { error -> state.copy(
                    capabilitySaving = false,
                    capabilityError = error.message ?: "Could not save the capability.",
                ) },
            ) }
        }
    }

    fun resetBuiltInSkill(key: String) = viewModelScope.launch(Dispatchers.IO) {
        _state.update { it.copy(capabilityError = null, capabilityNotice = null) }
        val outcome = runCatching {
            repository.resetBuiltInSkill(key)
            repository.builtInSkills()
        }
        _state.update { state -> outcome.fold(
            onSuccess = { skills -> state.copy(
                builtInSkills = skills,
                editingBuiltInSkillKey = state.editingBuiltInSkillKey.takeUnless { it == key },
                capabilityTitleDraft = if (state.editingBuiltInSkillKey == key) "" else state.capabilityTitleDraft,
                capabilityDescriptionDraft = if (state.editingBuiltInSkillKey == key) "" else state.capabilityDescriptionDraft,
                capabilityNotice = "Restored the packaged ${key.replace('_', ' ')} skill.",
            ) },
            onFailure = { error -> state.copy(capabilityError = error.message ?: "Could not restore the built-in skill.") },
        ) }
    }

    fun deleteCustomCapability(id: String) = viewModelScope.launch(Dispatchers.IO) {
        _state.update { it.copy(capabilityError = null, capabilityNotice = null) }
        val outcome = runCatching { repository.deleteCustomCapability(id) }
        _state.update { state -> outcome.fold(
            onSuccess = {
                val deletedEditing = state.editingCapabilityId == id
                state.copy(
                    customCapabilities = state.customCapabilities.filterNot { it.id == id },
                    editingCapabilityId = if (deletedEditing) null else state.editingCapabilityId,
                    capabilityTitleDraft = if (deletedEditing) "" else state.capabilityTitleDraft,
                    capabilityDescriptionDraft = if (deletedEditing) "" else state.capabilityDescriptionDraft,
                    capabilityNotice = "Removed custom capability.",
                )
            },
            onFailure = { error -> state.copy(capabilityError = error.message ?: "Could not remove the capability.") },
        ) }
    }

    fun showSettings() {
        _state.update { it.copy(screen = MosaicScreen.Settings, statusLoading = true) }
        viewModelScope.launch(Dispatchers.IO) {
            val status = runCatching { repository.localStatus() }.getOrNull()
            _state.update { it.copy(localStatus = status ?: it.localStatus, statusLoading = false) }
        }
    }

    fun updateDeepSeekApiKey(value: String) = _state.update { it.copy(deepSeekApiKeyDraft = value, webSearchError = null, webSearchNotice = null) }

    fun saveDeepSeekApiKey() {
        val value = _state.value.deepSeekApiKeyDraft
        viewModelScope.launch(Dispatchers.IO) {
            _state.update { it.copy(webSearchSaving = true, webSearchError = null, webSearchNotice = null) }
            val outcome = runCatching { repository.saveDeepSeekApiKey(value) }
            _state.update { state -> outcome.fold(
                onSuccess = {
                    val local = state.localStatus
                    state.copy(
                        webSearchSaving = false,
                        deepSeekApiKeyDraft = "",
                        localStatus = local?.copy(webSearchConfigured = true),
                        webSearchNotice = "DeepSeek web search key saved securely on this device.",
                    )
                },
                onFailure = { error -> state.copy(webSearchSaving = false, webSearchError = error.message ?: "Could not save the DeepSeek API key.") },
            ) }
        }
    }

    fun clearDeepSeekApiKey() = viewModelScope.launch(Dispatchers.IO) {
        _state.update { it.copy(webSearchSaving = true, webSearchError = null, webSearchNotice = null) }
        val outcome = runCatching { repository.clearDeepSeekApiKey() }
        _state.update { state -> outcome.fold(
            onSuccess = {
                val local = state.localStatus
                state.copy(
                    webSearchSaving = false,
                    deepSeekApiKeyDraft = "",
                    localStatus = local?.copy(webSearchConfigured = false),
                    webSearchNotice = "DeepSeek web search key removed from this device.",
                )
            },
            onFailure = { error -> state.copy(webSearchSaving = false, webSearchError = error.message ?: "Could not clear the DeepSeek API key.") },
        ) }
    }

    fun refreshMemories() = viewModelScope.launch(Dispatchers.IO) {
        _state.update { it.copy(memoriesLoading = true, memoryError = null) }
        val outcome = runCatching { repository.memoryClaims() }
        _state.update {
            outcome.fold(
                onSuccess = { claims -> it.copy(memories = claims, memoriesLoading = false) },
                onFailure = { error -> it.copy(memoriesLoading = false, memoryError = error.message ?: "Could not load memories") },
            )
        }
    }

    fun openMemory(claimId: String) = viewModelScope.launch(Dispatchers.IO) {
        _state.update { it.copy(memoryDetailLoading = true, memoryError = null) }
        val outcome = runCatching { repository.memoryClaim(claimId) }
        _state.update {
            outcome.fold(
                onSuccess = { detail -> it.copy(screen = MosaicScreen.MemoryDetail, selectedMemory = detail, memoryDetailLoading = false) },
                onFailure = { error -> it.copy(memoryDetailLoading = false, memoryError = error.message ?: "Could not open memory") },
            )
        }
    }

    fun openEvidenceConversation(evidence: EvidenceItem) {
        _state.update { it.copy(highlightedMessageId = evidence.messageId) }
        showChat()
        selectConversation(evidence.conversationId)
    }

    fun deleteMemory(claimId: String) = viewModelScope.launch(Dispatchers.IO) {
        val outcome = runCatching { repository.deleteMemoryClaim(claimId) }
        _state.update { state ->
            outcome.fold(
                onSuccess = {
                    state.copy(screen = MosaicScreen.Memories, selectedMemory = null,
                        memories = state.memories.filterNot { item -> item.id == claimId }, memoryError = null)
                },
                onFailure = { error -> state.copy(memoryError = error.message ?: "Could not delete memory") },
            )
        }
    }

    fun updateDraft(value: String) {
        _state.update { state -> state.copy(drafts = state.drafts + (state.draftKey to value)) }
    }

    fun selectImage(uri: Uri) = _state.update { it.copy(screen = MosaicScreen.Chat, selectedImage = uri, selectedImageSource = null, transientError = null) }

    /** Index a user-selected photo before it appears in the Images browser. */
    fun indexSelectedImage(uri: Uri) = viewModelScope.launch(Dispatchers.IO) {
        _state.update { it.copy(imagesLoading = true, imageIndexing = true, imageError = null, imageNotice = null) }
        val outcome = runCatching { repository.indexImage(uri) }
        _state.update { state -> outcome.fold(
            onSuccess = { indexed ->
                state.copy(
                    indexedImages = listOf(indexed) + state.indexedImages.filterNot { it.id == indexed.id },
                    imagesLoading = false,
                    imageIndexing = false,
                    imageSearchQuery = null,
                    imageNotice = "Indexed ${indexed.filename} with EmbeddingGemma.",
                )
            },
            onFailure = { error -> state.copy(
                imagesLoading = false,
                imageIndexing = false,
                imageError = error.message ?: "Could not index the selected image.",
            ) },
        ) }
    }

    fun removeSelectedImage() = _state.update { it.copy(selectedImage = null, selectedImageSource = null) }

    fun sendDraft() {
        val snapshot = _state.value
        val draft = snapshot.draft.trim()
        if ((draft.isEmpty() && snapshot.selectedImage == null && snapshot.selectedImageSource == null) || snapshot.activeRequest != null) return
        if (snapshot.runtimeStatus !is RuntimeStatus.Ready) {
            _state.update { it.copy(transientError = "Import and load Qwen before sending a message.") }
            return
        }
        viewModelScope.launch(Dispatchers.IO) {
            val conversation = if (snapshot.activeConversationId == null) {
                runCatching { repository.createConversation(draft.ifBlank { "Image question" }.take(48)) }.getOrElse { error ->
                    _state.update { it.copy(transientError = error.message ?: "Could not create conversation") }
                    return@launch
                }
            } else {
                snapshot.conversations.firstOrNull { it.id == snapshot.activeConversationId }
                    ?: runCatching { repository.createConversation() }.getOrElse { error ->
                        _state.update { it.copy(transientError = error.message ?: "Could not open conversation") }
                        return@launch
                    }
            }
            val active = ActiveRequest(
                requestId = "pending-${UUID.randomUUID()}",
                conversationId = conversation.id,
                startedElapsedMs = SystemClock.elapsedRealtime(),
                phase = RequestPhase.Queued,
                activity = listOf(ActivityItem("queued", "Working on your request", ActivityStatus.Running)),
            )
            _state.update { current ->
                current.copy(
                    activeConversationId = conversation.id,
                    conversations = listOf(conversation) + current.conversations.filterNot { it.id == conversation.id },
                    drafts = current.drafts + (conversation.id to "") + (current.draftKey to ""),
                    selectedImage = null,
                    selectedImageSource = null,
                    activeRequest = active,
                    lastRequest = null,
                    transientError = null,
                )
            }
            val outcome = runCatching {
                when {
                    snapshot.selectedImageSource != null -> repository.streamStoredImageMessage(
                        conversation.id, draft.ifBlank { "Describe this image." }, snapshot.selectedImageSource.id, ::onRuntimeEvent,
                    )
                    snapshot.selectedImage != null -> repository.streamImageMessage(
                        conversation.id, draft.ifBlank { "Describe this image." }, snapshot.selectedImage, ::onRuntimeEvent,
                    )
                    else -> repository.streamMessage(conversation.id, draft, ::onRuntimeEvent)
                }
            }
            outcome.onSuccess { result -> finishRequest(conversation.id, result) }
                .onFailure { error -> failRequest(conversation.id, error.message ?: "Local request failed") }
        }
    }

    fun cancelActiveRequest() {
        val active = _state.value.activeRequest ?: return
        _state.update { state ->
            state.copy(activeRequest = active.copy(phase = RequestPhase.Cancelling, dropFurtherDeltas = true))
        }
        repository.cancelActiveGeneration()
    }

    fun renameActiveConversation(title: String) {
        val id = _state.value.activeConversationId ?: return
        renameConversation(id, title)
    }

    fun renameConversation(id: String, title: String) {
        viewModelScope.launch(Dispatchers.IO) {
            runCatching { repository.renameConversation(id, title.trim().ifBlank { null }) }
                .onSuccess { updated -> _state.update { state ->
                    state.copy(conversations = state.conversations.map { if (it.id == updated.id) updated else it })
                } }
                .onFailure { error -> _state.update { it.copy(transientError = error.message ?: "Could not rename conversation") } }
        }
    }

    fun deleteActiveConversation() {
        val id = _state.value.activeConversationId ?: return
        deleteConversation(id)
    }

    fun deleteConversation(id: String) {
        if (_state.value.activeRequest?.conversationId == id) {
            _state.update { it.copy(transientError = "Stop the active request before deleting this conversation.") }
            return
        }
        viewModelScope.launch(Dispatchers.IO) {
            runCatching { repository.deleteConversation(id) }
                .onSuccess { _state.update { state ->
                    state.copy(
                        activeConversationId = state.activeConversationId.takeUnless { it == id },
                        messages = if (state.activeConversationId == id) emptyList() else state.messages,
                        conversations = state.conversations.filterNot { it.id == id },
                        drafts = state.drafts - id,
                    )
                } }
                .onFailure { error -> _state.update { it.copy(transientError = error.message ?: "Could not delete conversation") } }
        }
    }

    fun importQwen(uri: Uri) = viewModelScope.launch(Dispatchers.IO) {
        _state.update { it.copy(runtimeStatus = RuntimeStatus.Starting, transientError = null) }
        val status = runCatching { repository.importQwen(uri) }
            .getOrElse { RuntimeStatus.Error(it.message ?: "Model import failed") }
        _state.update { it.copy(runtimeStatus = status) }
    }

    fun clearError() = _state.update { it.copy(transientError = null) }

    private fun onRuntimeEvent(event: RuntimeEvent) {
        _state.update { state ->
            val active = state.activeRequest ?: return@update state
            if (event.sequence >= 0 && event.requestId != active.requestId && !active.requestId.startsWith("pending-")) return@update state
            var revised = if (active.requestId.startsWith("pending-") && event.type == "request_started") {
                active.copy(requestId = event.requestId)
            } else active
            if (event.sequence >= 0 && event.sequence <= revised.lastSequence) return@update state
            revised = revised.copy(lastSequence = event.sequence)

            revised = when (event.type) {
                "model_loading" -> revised.withStage(RequestPhase.LoadingModel, "Loading ${event.model ?: "local model"}")
                "model_ready" -> revised.withStage(RequestPhase.ProcessingPrompt, "Preparing your request")
                "prompt_processing" -> revised.withStage(RequestPhase.ProcessingPrompt, "Processing your message")
                "tool_started" -> revised.withStage(RequestPhase.RunningTools, event.displayText ?: "Using a local tool")
                "tool_completed" -> revised.completeTool(event.displayText ?: "Local tool", event.resultCount, event.sourceSummaries)
                "tool_failed" -> revised.failTool(event.displayText ?: "Local tool")
                "answer_started" -> revised.withStage(RequestPhase.Answering, "Writing response")
                "answer_delta" -> if (revised.dropFurtherDeltas) revised else revised.copy(
                    phase = RequestPhase.Answering,
                    streamedText = revised.streamedText + (event.delta ?: ""),
                )
                "request_cancelled" -> revised.finishStage(RequestPhase.Cancelled, ActivityStatus.Failed)
                "request_failed" -> revised.finishStage(RequestPhase.Failed, ActivityStatus.Failed).copy(errorMessage = event.errorMessage)
                "answer_completed" -> revised.finishStage(RequestPhase.Completed, ActivityStatus.Completed)
                else -> revised
            }
            state.copy(activeRequest = revised)
        }
    }

    private fun finishRequest(conversationId: String, result: RequestResult) {
        viewModelScope.launch(Dispatchers.IO) {
            val messages = runCatching { repository.messages(conversationId) }.getOrElse { emptyList() }
            val conversations = runCatching { repository.listConversations() }.getOrElse { _state.value.conversations }
            _state.update { state ->
                val active = state.activeRequest?.takeIf { it.conversationId == conversationId }
                val completed = active?.finishStage(
                    if (result.cancelled) RequestPhase.Cancelled else RequestPhase.Completed,
                    if (result.cancelled) ActivityStatus.Failed else ActivityStatus.Completed,
                )
                state.copy(
                    messages = if (state.activeConversationId == conversationId) messages else state.messages,
                    conversations = conversations,
                    activeRequest = null,
                    lastRequest = completed,
                )
            }
        }
    }

    private fun failRequest(conversationId: String, message: String) {
        _state.update { state ->
            val active = state.activeRequest?.takeIf { it.conversationId == conversationId }
            state.copy(
                activeRequest = null,
                lastRequest = active?.finishStage(RequestPhase.Failed, ActivityStatus.Failed)?.copy(errorMessage = message),
                transientError = message,
            )
        }
    }
}

data class MosaicUiState(
    val startupComplete: Boolean = false,
    val runtimeStatus: RuntimeStatus = RuntimeStatus.Starting,
    val conversations: List<ConversationSummary> = emptyList(),
    val activeConversationId: String? = null,
    val messages: List<ChatMessage> = emptyList(),
    val drafts: Map<String, String> = emptyMap(),
    val activeRequest: ActiveRequest? = null,
    val lastRequest: ActiveRequest? = null,
    val transientError: String? = null,
    val screen: MosaicScreen = MosaicScreen.Home,
    val memories: List<MemoryClaimSummary> = emptyList(),
    val memoriesLoading: Boolean = false,
    val selectedMemory: MemoryClaimDetail? = null,
    val memoryDetailLoading: Boolean = false,
    val memoryError: String? = null,
    val selectedImage: Uri? = null,
    val selectedImageSource: IndexedImage? = null,
    val selectedIndexedImage: IndexedImage? = null,
    val highlightedMessageId: String? = null,
    val indexedImages: List<IndexedImage> = emptyList(),
    val imagesLoading: Boolean = false,
    val imageIndexing: Boolean = false,
    val imageSearchQuery: String? = null,
    val imageSearchElapsedMs: Double? = null,
    val imageError: String? = null,
    val imageNotice: String? = null,
    val customCapabilities: List<CustomCapability> = emptyList(),
    val builtInSkills: List<BuiltInSkill> = emptyList(),
    val capabilitiesLoading: Boolean = false,
    val capabilitySaving: Boolean = false,
    val editingCapabilityId: String? = null,
    val editingBuiltInSkillKey: String? = null,
    val capabilityTitleDraft: String = "",
    val capabilityDescriptionDraft: String = "",
    val capabilityError: String? = null,
    val capabilityNotice: String? = null,
    val localStatus: LocalStatus? = null,
    val statusLoading: Boolean = false,
    val deepSeekApiKeyDraft: String = "",
    val webSearchSaving: Boolean = false,
    val webSearchError: String? = null,
    val webSearchNotice: String? = null,
) {
    val draftKey: String get() = activeConversationId ?: NEW_CONVERSATION_DRAFT
    val draft: String get() = drafts[draftKey].orEmpty()
    val activeConversation: ConversationSummary? get() = conversations.firstOrNull { it.id == activeConversationId }
    val requestForActiveConversation: ActiveRequest? get() = activeRequest?.takeIf { it.conversationId == activeConversationId }

    companion object { private const val NEW_CONVERSATION_DRAFT = "__new__" }
}

enum class MosaicScreen { Home, Chat, Conversations, Memories, MemoryDetail, Images, ImageDetail, Skills, Settings }

enum class RequestPhase { Queued, LoadingModel, ProcessingPrompt, RunningTools, Answering, Cancelling, Completed, Cancelled, Failed }
enum class ActivityStatus { Running, Completed, Failed }

data class ActivityItem(val key: String, val label: String, val status: ActivityStatus, val detail: String? = null)

data class ActiveRequest(
    val requestId: String,
    val conversationId: String,
    val startedElapsedMs: Long,
    val finishedElapsedMs: Long? = null,
    val phase: RequestPhase,
    val activity: List<ActivityItem>,
    val streamedText: String = "",
    val lastSequence: Long = -1,
    val dropFurtherDeltas: Boolean = false,
    val errorMessage: String? = null,
    val sources: List<SourceSummary> = emptyList(),
) {
    fun withStage(phase: RequestPhase, label: String) = copy(
        phase = phase,
        activity = activity.map {
            if (it.status == ActivityStatus.Running) it.copy(status = ActivityStatus.Completed) else it
        } + ActivityItem("$phase-${activity.size}", label, ActivityStatus.Running),
    )

    fun finishStage(phase: RequestPhase, status: ActivityStatus) = copy(
        phase = phase,
        finishedElapsedMs = finishedElapsedMs ?: SystemClock.elapsedRealtime(),
        activity = activity.map { item -> if (item.status == ActivityStatus.Running) item.copy(status = status) else item },
    )

    fun completeTool(label: String, resultCount: Int?, sourceSummaries: List<SourceSummary>) = copy(
        activity = activity.map {
            if (it.status == ActivityStatus.Running) it.copy(status = ActivityStatus.Completed) else it
        } + ActivityItem("tool-complete-${activity.size}", label, ActivityStatus.Completed,
            resultCount?.let { "Found $it records" }),
        sources = sources + sourceSummaries.filterNot { candidate -> sources.any { it.kind == candidate.kind && it.id == candidate.id } },
    )

    fun failTool(label: String) = copy(
        activity = activity.map {
            if (it.status == ActivityStatus.Running) it.copy(status = ActivityStatus.Failed) else it
        } + ActivityItem("tool-failed-${activity.size}", label, ActivityStatus.Failed),
    )
}

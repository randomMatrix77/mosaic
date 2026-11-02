package dev.mosaic.poc

import android.graphics.BitmapFactory
import android.content.Intent
import android.net.Uri
import android.os.SystemClock
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay

@Composable
fun MosaicApp(
    state: MosaicUiState,
    onNewConversation: () -> Unit,
    onSelectConversation: (String) -> Unit,
    onDraftChanged: (String) -> Unit,
    onSend: () -> Unit,
    onCancel: () -> Unit,
    onRename: (String) -> Unit,
    onDelete: () -> Unit,
    onRenameConversation: (String, String) -> Unit,
    onDeleteConversation: (String) -> Unit,
    onImportQwen: () -> Unit,
    onClearError: () -> Unit,
    onShowHome: () -> Unit,
    onShowChat: () -> Unit,
    onShowConversations: () -> Unit,
    onShowMemories: () -> Unit,
    onRefreshMemories: () -> Unit,
    onOpenMemory: (String) -> Unit,
    onOpenEvidenceConversation: (EvidenceItem) -> Unit,
    onDeleteMemory: (String) -> Unit,
    onPickImage: () -> Unit,
    onAddImageToIndex: () -> Unit,
    onRemoveImage: () -> Unit,
    onShowImages: () -> Unit,
    onRefreshImages: () -> Unit,
    onSearchImages: (String) -> Unit,
    onOpenIndexedImage: (IndexedImage) -> Unit,
    onAskAboutIndexedImage: () -> Unit,
    onShowSkills: () -> Unit,
    onCapabilityTitleChanged: (String) -> Unit,
    onCapabilityDescriptionChanged: (String) -> Unit,
    onAddCapability: () -> Unit,
    onEditCapability: (CustomCapability) -> Unit,
    onCancelCapabilityEdit: () -> Unit,
    onDeleteCapability: (String) -> Unit,
    onEditBuiltInSkill: (BuiltInSkill) -> Unit,
    onResetBuiltInSkill: (String) -> Unit,
    onShowSettings: () -> Unit,
    onDeepSeekApiKeyChanged: (String) -> Unit,
    onSaveDeepSeekApiKey: () -> Unit,
    onClearDeepSeekApiKey: () -> Unit,
) {
    MaterialTheme(colorScheme = mosaicColorScheme()) {
        BoxWithConstraints(Modifier.fillMaxSize().background(MosaicTokens.Background)) {
        // Three panes need room for a 232dp navigation rail, a 360dp minimum
        // workspace, and a 286dp context rail. A Fold 6 at its default display
        // density is therefore a medium layout (navigation + workspace), while
        // this rail activates only when the actual available width is sufficient.
        val expanded = maxWidth >= 880.dp
            val persistentNavigation = maxWidth >= 700.dp
            if (persistentNavigation) {
                Row(Modifier.fillMaxSize()) {
                    MosaicNavigation(
                        state, onNewConversation, onSelectConversation, onShowHome, onShowChat, onShowConversations,
                        onShowMemories, onShowImages, onShowSkills, onShowSettings,
                        Modifier.width(232.dp).fillMaxHeight(),
                    )
                    MosaicMain(
                        state, true, null, onNewConversation, onSelectConversation, onDraftChanged, onSend, onCancel,
                        onRename, onDelete, onRenameConversation, onDeleteConversation, onImportQwen, onClearError,
                        onShowHome, onShowChat, onShowConversations, onShowMemories, onRefreshMemories, onOpenMemory,
                        onOpenEvidenceConversation, onDeleteMemory, onPickImage, onAddImageToIndex, onRemoveImage, onShowImages,
                        onRefreshImages, onSearchImages, onOpenIndexedImage, onAskAboutIndexedImage, onShowSkills,
                        onCapabilityTitleChanged, onCapabilityDescriptionChanged, onAddCapability, onEditCapability, onCancelCapabilityEdit, onDeleteCapability,
                        onEditBuiltInSkill, onResetBuiltInSkill,
                        onShowSettings, onDeepSeekApiKeyChanged, onSaveDeepSeekApiKey, onClearDeepSeekApiKey, Modifier.weight(1f),
                    )
                    if (expanded) MosaicContextRail(state, Modifier.width(286.dp).fillMaxHeight())
                }
            } else {
                var drawerOpen by remember { mutableStateOf(false) }
                var edgeDragDistance by remember { mutableFloatStateOf(0f) }
                var edgeDragActive by remember { mutableStateOf(false) }
                val density = LocalDensity.current
                // The first few physical pixels belong to Android's Back gesture on
                // gesture-navigation devices. Start just inside that zone, while still
                // keeping the target close to the left edge of the compact display.
                val edgeActivationWidth = with(density) { 64.dp.toPx() }
                val drawerRevealDistance = with(density) { 72.dp.toPx() }
                Box(
                    Modifier
                        .fillMaxSize()
                        // Compact screens have no persistent navigation rail. Only a deliberate
                        // rightward drag beginning near the left edge opens the drawer, so regular
                        // horizontal gestures inside the workspace remain unaffected.
                        .pointerInput(drawerOpen, edgeActivationWidth, drawerRevealDistance) {
                            detectHorizontalDragGestures(
                                onDragStart = { offset ->
                                    edgeDragActive = !drawerOpen && offset.x <= edgeActivationWidth
                                    edgeDragDistance = 0f
                                },
                                onHorizontalDrag = { change, dragAmount ->
                                    if (!edgeDragActive) return@detectHorizontalDragGestures
                                    if (dragAmount <= 0f) return@detectHorizontalDragGestures

                                    edgeDragDistance += dragAmount
                                    change.consume()
                                    if (edgeDragDistance >= drawerRevealDistance) {
                                        drawerOpen = true
                                        edgeDragActive = false
                                        edgeDragDistance = 0f
                                    }
                                },
                                onDragEnd = {
                                    edgeDragActive = false
                                    edgeDragDistance = 0f
                                },
                                onDragCancel = {
                                    edgeDragActive = false
                                    edgeDragDistance = 0f
                                },
                            )
                        },
                ) {
                    MosaicMain(
                        state, false, { drawerOpen = true }, onNewConversation, onSelectConversation, onDraftChanged,
                        onSend, onCancel, onRename, onDelete, onRenameConversation, onDeleteConversation, onImportQwen,
                        onClearError, onShowHome, onShowChat, onShowConversations, onShowMemories, onRefreshMemories,
                        onOpenMemory, onOpenEvidenceConversation, onDeleteMemory, onPickImage, onAddImageToIndex, onRemoveImage, onShowImages,
                        onRefreshImages, onSearchImages, onOpenIndexedImage, onAskAboutIndexedImage, onShowSkills,
                        onCapabilityTitleChanged, onCapabilityDescriptionChanged, onAddCapability, onEditCapability, onCancelCapabilityEdit, onDeleteCapability,
                        onEditBuiltInSkill, onResetBuiltInSkill,
                        onShowSettings, onDeepSeekApiKeyChanged, onSaveDeepSeekApiKey, onClearDeepSeekApiKey, Modifier.fillMaxSize(),
                    )
                    if (drawerOpen) {
                        Box(Modifier.fillMaxSize().background(Color.Black.copy(alpha = 0.48f)).clickable { drawerOpen = false })
                        MosaicNavigation(
                            state,
                            { drawerOpen = false; onNewConversation() },
                            { id -> drawerOpen = false; onSelectConversation(id) },
                            { drawerOpen = false; onShowHome() }, { drawerOpen = false; onShowChat() },
                            { drawerOpen = false; onShowConversations() }, { drawerOpen = false; onShowMemories() },
                            { drawerOpen = false; onShowImages() }, { drawerOpen = false; onShowSkills() },
                            { drawerOpen = false; onShowSettings() }, Modifier.width(320.dp).fillMaxHeight(),
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun MosaicNavigation(
    state: MosaicUiState,
    onNew: () -> Unit,
    onSelect: (String) -> Unit,
    onHome: () -> Unit,
    onChat: () -> Unit,
    onConversations: () -> Unit,
    onMemories: () -> Unit,
    onImages: () -> Unit,
    onSkills: () -> Unit,
    onSettings: () -> Unit,
    modifier: Modifier,
) {
    Surface(modifier, color = MosaicTokens.Surface, border = BorderStroke(1.dp, MosaicTokens.Border)) {
        Column(Modifier.fillMaxSize().statusBarsPadding().padding(horizontal = 13.dp, vertical = 18.dp)) {
            Image(painterResource(R.drawable.mosaic_lockup_1024), "Mosaic", Modifier.width(122.dp).height(36.dp), contentScale = ContentScale.Fit)
            Spacer(Modifier.height(22.dp))
            NavigationItem("Home", R.drawable.mosaic_ic_sparkles, state.screen == MosaicScreen.Home, onHome)
            NavigationItem("Chat", R.drawable.mosaic_ic_message_square, state.screen == MosaicScreen.Chat, onChat)
            NavigationItem("Conversations", R.drawable.mosaic_ic_layout_panel, state.screen == MosaicScreen.Conversations, onConversations)
            NavigationItem("Memories", R.drawable.mosaic_ic_brain, state.screen == MosaicScreen.Memories || state.screen == MosaicScreen.MemoryDetail, onMemories)
            NavigationItem("Images", R.drawable.mosaic_ic_images, state.screen == MosaicScreen.Images || state.screen == MosaicScreen.ImageDetail, onImages)
            NavigationItem("Skills", R.drawable.mosaic_ic_skills, state.screen == MosaicScreen.Skills, onSkills)
            Spacer(Modifier.height(10.dp))
            MosaicPrimaryButton("New conversation", onNew, icon = R.drawable.mosaic_ic_plus)
            Spacer(Modifier.height(18.dp))
            Text("RECENT CONVERSATIONS", color = MosaicTokens.Secondary, fontSize = 10.sp, fontWeight = FontWeight.Bold, modifier = Modifier.padding(horizontal = 8.dp))
            Spacer(Modifier.height(8.dp))
            if (state.conversations.isEmpty()) {
                Text("Your saved conversations will appear here.", color = MosaicTokens.Tertiary, fontSize = 11.sp, modifier = Modifier.padding(8.dp))
            } else {
                LazyColumn(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    items(state.conversations.take(12), key = { it.id }) { conversation ->
                        Surface(
                            modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).clickable { onSelect(conversation.id) },
                            color = if (conversation.id == state.activeConversationId) MosaicTokens.Active else Color.Transparent,
                            shape = RoundedCornerShape(12.dp),
                        ) {
                            Row(Modifier.padding(horizontal = 10.dp, vertical = 11.dp), verticalAlignment = Alignment.CenterVertically) {
                                MosaicIcon(R.drawable.mosaic_ic_message_square, "Conversation", tint = MosaicTokens.Secondary)
                                Spacer(Modifier.width(9.dp))
                                Text(conversation.title ?: "Untitled conversation", color = MosaicTokens.Secondary, fontSize = 12.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
                            }
                        }
                    }
                }
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                RuntimePill(state.runtimeStatus, Modifier.weight(1f))
                MosaicIconButton(R.drawable.mosaic_ic_settings, "Settings", onSettings, tint = MosaicTokens.Secondary)
            }
        }
    }
}

@Composable
private fun NavigationItem(label: String, icon: Int, selected: Boolean, onClick: () -> Unit) {
    Surface(
        modifier = Modifier.fillMaxWidth().height(48.dp).clip(RoundedCornerShape(12.dp)).clickable(onClick = onClick),
        color = if (selected) Color(0xFF513381) else Color.Transparent, shape = RoundedCornerShape(12.dp),
    ) {
        Row(Modifier.padding(horizontal = 12.dp), verticalAlignment = Alignment.CenterVertically) {
            MosaicIcon(icon, label, tint = if (selected) Color.White else MosaicTokens.Primary)
            Spacer(Modifier.width(12.dp))
            Text(label, color = if (selected) Color.White else MosaicTokens.Secondary, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
        }
    }
}

@Composable
private fun MosaicMain(
    state: MosaicUiState, wide: Boolean, openDrawer: (() -> Unit)?, onNew: () -> Unit, onSelect: (String) -> Unit,
    onDraft: (String) -> Unit, onSend: () -> Unit, onCancel: () -> Unit, onRename: (String) -> Unit, onDelete: () -> Unit,
    onRenameConversation: (String, String) -> Unit, onDeleteConversation: (String) -> Unit, onImport: () -> Unit,
    onClearError: () -> Unit, onHome: () -> Unit, onChat: () -> Unit, onConversations: () -> Unit, onMemories: () -> Unit,
    onRefreshMemories: () -> Unit, onOpenMemory: (String) -> Unit, onOpenEvidence: (EvidenceItem) -> Unit,
    onDeleteMemory: (String) -> Unit, onPickImage: () -> Unit, onAddImageToIndex: () -> Unit, onRemoveImage: () -> Unit, onImages: () -> Unit,
    onRefreshImages: () -> Unit, onSearchImages: (String) -> Unit, onOpenIndexedImage: (IndexedImage) -> Unit,
    onAskAboutIndexedImage: () -> Unit, onSkills: () -> Unit, onCapabilityTitleChanged: (String) -> Unit,
    onCapabilityDescriptionChanged: (String) -> Unit, onAddCapability: () -> Unit, onEditCapability: (CustomCapability) -> Unit,
    onCancelCapabilityEdit: () -> Unit, onDeleteCapability: (String) -> Unit,
    onEditBuiltInSkill: (BuiltInSkill) -> Unit, onResetBuiltInSkill: (String) -> Unit,
    onSettings: () -> Unit, onDeepSeekApiKeyChanged: (String) -> Unit, onSaveDeepSeekApiKey: () -> Unit,
    onClearDeepSeekApiKey: () -> Unit, modifier: Modifier,
) = when (state.screen) {
    MosaicScreen.Home -> HomeScreen(state, onNew, onImages, onMemories, onSettings, onPickImage, modifier)
    MosaicScreen.Chat -> ChatScreen(state, wide, openDrawer, onNew, onDraft, onSend, onCancel, onRename, onDelete, onImport, onClearError, onPickImage, onRemoveImage, modifier)
    MosaicScreen.Conversations -> ConversationsScreen(state, wide, openDrawer, onNew, onSelect, onRenameConversation, onDeleteConversation, modifier)
    MosaicScreen.Memories -> MemoriesScreen(state, wide, openDrawer, onRefreshMemories, onOpenMemory, modifier)
    MosaicScreen.MemoryDetail -> MemoryDetailScreen(state, wide, openDrawer, onMemories, onOpenEvidence, onDeleteMemory, modifier)
    MosaicScreen.Images -> ImagesScreen(state, wide, openDrawer, onRefreshImages, onSearchImages, onOpenIndexedImage, onPickImage, onAddImageToIndex, modifier)
    MosaicScreen.ImageDetail -> ImageDetailScreen(state, wide, openDrawer, onImages, onAskAboutIndexedImage, modifier)
    MosaicScreen.Skills -> SkillsScreen(state, wide, openDrawer, onCapabilityTitleChanged, onCapabilityDescriptionChanged, onAddCapability, onEditCapability, onCancelCapabilityEdit, onDeleteCapability, onEditBuiltInSkill, onResetBuiltInSkill, modifier)
    MosaicScreen.Settings -> SettingsScreen(state, wide, openDrawer, onImport, onDeepSeekApiKeyChanged, onSaveDeepSeekApiKey, onClearDeepSeekApiKey, modifier)
}

@Composable
private fun HomeScreen(state: MosaicUiState, onNew: () -> Unit, onImages: () -> Unit, onMemories: () -> Unit, onSettings: () -> Unit, onPickImage: () -> Unit, modifier: Modifier) {
    val models = state.localStatus?.models.orEmpty()
    val embeddings = models["embeddinggemma"]?.present == true && models["embedding_projector"]?.present == true
    val vision = models["qwen"]?.present == true && models["qwen_projector"]?.present == true
    LazyColumn(
        modifier.fillMaxSize().background(MosaicTokens.Background),
        contentPadding = PaddingValues(bottom = 28.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        item { Image(painterResource(R.drawable.mosaic_welcome_wave_1024), null, Modifier.fillMaxWidth().height(190.dp), contentScale = ContentScale.Crop) }
        item {
            Column(Modifier.padding(horizontal = 20.dp)) {
                Text("WELCOME TO MOSAIC", color = MosaicTokens.Primary, fontSize = 11.sp, fontWeight = FontWeight.Bold, letterSpacing = 1.5.sp)
                Spacer(Modifier.height(18.dp))
                Text("Your personal AI,\nright here.", color = MosaicTokens.Text, fontSize = 28.sp, lineHeight = 34.sp, fontWeight = FontWeight.Bold)
                Spacer(Modifier.height(16.dp))
                Text("Ask questions, explore your memories, and understand images — locally on your device.", color = MosaicTokens.Secondary, fontSize = 14.sp, lineHeight = 22.sp)
            }
        }
        item { Column(Modifier.padding(horizontal = 20.dp, vertical = 8.dp)) {
            HomeAction("Continue chatting", R.drawable.mosaic_ic_message_square, onNew)
            HomeAction("Ask about a photo", R.drawable.mosaic_ic_image, onPickImage)
            HomeAction("Explore your memories", R.drawable.mosaic_ic_brain, onMemories)
            HomeAction("Browse indexed images", R.drawable.mosaic_ic_images, onImages)
        } }
        item { Box(Modifier.padding(horizontal = 20.dp)) { SectionLabel("ON-DEVICE CAPABILITIES") } }
        item { Column(Modifier.padding(horizontal = 20.dp)) { CapabilityRow(R.drawable.mosaic_ic_brain, "Memory retrieval", "Find connected information from past conversations.", if (embeddings) "Available" else "Unavailable") } }
        item { Column(Modifier.padding(horizontal = 20.dp)) { CapabilityRow(R.drawable.mosaic_ic_images, "Image retrieval", "Find images from the local index by description.", if (embeddings) "Available" else "Unavailable") } }
        item { Column(Modifier.padding(horizontal = 20.dp)) { CapabilityRow(R.drawable.mosaic_ic_image, "Image understanding", "Ask Qwen questions about a selected photo.", if (vision) "Available" else "Unavailable") } }
        item { Column(Modifier.padding(horizontal = 20.dp)) { CapabilityRow(R.drawable.mosaic_ic_layout_panel, "Memory organization", "Extract evidence-backed facts from conversations.", "Limited") } }
        item {
            Row(Modifier.padding(horizontal = 20.dp, vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                RuntimePill(state.runtimeStatus, Modifier.weight(1f))
                MosaicIconButton(R.drawable.mosaic_ic_external_link, "Model status", onSettings, tint = MosaicTokens.Primary)
            }
        }
    }
}

@Composable
private fun HomeAction(label: String, icon: Int, onClick: () -> Unit) {
    MosaicCard(modifier = Modifier.padding(bottom = 10.dp), onClick = onClick) {
        Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
            MosaicIcon(icon, label, tint = MosaicTokens.Primary)
            Spacer(Modifier.width(14.dp)); Text(label, Modifier.weight(1f), color = MosaicTokens.Text, fontSize = 15.sp, fontWeight = FontWeight.Bold)
            MosaicIcon(R.drawable.mosaic_ic_chevron_right, "Open", tint = MosaicTokens.Secondary)
        }
    }
}

@Composable
private fun ChatScreen(
    state: MosaicUiState, wide: Boolean, openDrawer: (() -> Unit)?, onNew: () -> Unit, onDraft: (String) -> Unit,
    onSend: () -> Unit, onCancel: () -> Unit, onRename: (String) -> Unit, onDelete: () -> Unit, onImport: () -> Unit,
    onClearError: () -> Unit, onPickImage: () -> Unit, onRemoveImage: () -> Unit, modifier: Modifier,
) {
    var menu by remember { mutableStateOf(false) }
    var rename by remember { mutableStateOf(false) }
    var remove by remember { mutableStateOf(false) }
    Column(modifier.fillMaxSize().background(MosaicTokens.Background)) {
        MosaicTopBar(
            title = state.activeConversation?.title ?: "New conversation", subtitle = runtimeDescription(state.runtimeStatus),
            onNavigation = if (wide) null else openDrawer, action = {
                RuntimePill(state.runtimeStatus)
                MosaicIconButton(R.drawable.mosaic_ic_more_vertical, "Conversation actions", { menu = true })
            },
        )
        if (state.messages.isEmpty() && state.requestForActiveConversation == null) {
            ChatEmptyState(state.runtimeStatus, onNew, onImport, Modifier.weight(1f))
        } else {
            ConversationThread(state, onClearError, Modifier.weight(1f))
        }
        MosaicComposer(state, onDraft, onSend, onCancel, onPickImage, onRemoveImage)
    }
    if (menu) AlertDialog(
        onDismissRequest = { menu = false }, title = { Text("Conversation") },
        text = { Text(if (state.activeConversationId == null) "Send a message to create a local conversation." else "Rename or delete this stored conversation.") },
        confirmButton = { MosaicPrimaryButton(if (state.activeConversationId == null) "New conversation" else "Rename", { menu = false; if (state.activeConversationId == null) onNew() else rename = true }) },
        dismissButton = { if (state.activeConversationId != null) MosaicPrimaryButton("Delete", { menu = false; remove = true }) },
    )
    if (rename) {
        var title by remember(state.activeConversationId) { mutableStateOf(state.activeConversation?.title.orEmpty()) }
        AlertDialog(onDismissRequest = { rename = false }, title = { Text("Rename conversation") }, text = {
            MosaicSearchField(title, "Conversation title", { title = it })
        }, confirmButton = { MosaicPrimaryButton("Save", { onRename(title); rename = false }, enabled = title.isNotBlank()) }, dismissButton = { MosaicPrimaryButton("Cancel", { rename = false }) })
    }
    if (remove) AlertDialog(onDismissRequest = { remove = false }, title = { Text("Delete this conversation?") }, text = { Text("Messages and derived evidence with no other source will be removed. This cannot be undone.") }, confirmButton = { MosaicPrimaryButton("Delete", { onDelete(); remove = false }) }, dismissButton = { MosaicPrimaryButton("Cancel", { remove = false }) })
}

@Composable
private fun ChatEmptyState(status: RuntimeStatus, onNew: () -> Unit, onImport: () -> Unit, modifier: Modifier) {
    Column(modifier.fillMaxWidth().padding(28.dp), verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally) {
        Image(painterResource(R.drawable.mosaic_symbol_256), "Mosaic", Modifier.size(64.dp))
        Spacer(Modifier.height(20.dp)); Text("Start a local conversation", color = MosaicTokens.Text, fontSize = 21.sp, fontWeight = FontWeight.Bold)
        Spacer(Modifier.height(8.dp)); Text("Messages, memories, and model inference stay on this device.", color = MosaicTokens.Secondary, textAlign = TextAlign.Center, fontSize = 13.sp, lineHeight = 20.sp)
        Spacer(Modifier.height(20.dp)); RuntimePill(status)
        Spacer(Modifier.height(22.dp))
        if (status is RuntimeStatus.Unavailable || status is RuntimeStatus.Error) MosaicPrimaryButton("Import Qwen model", onImport) else MosaicPrimaryButton("Start a conversation", onNew, icon = R.drawable.mosaic_ic_message_circle_plus)
    }
}

@Composable
private fun ConversationThread(state: MosaicUiState, onClearError: () -> Unit, modifier: Modifier) {
    val listState = rememberLazyListState()
    val request = state.requestForActiveConversation
    val messages = state.messages.filter { it.kind == "chat" }
    LaunchedEffect(messages.size, request?.streamedText?.length, state.highlightedMessageId) {
        val anchor = state.highlightedMessageId?.let { id -> messages.indexOfFirst { it.id == id } }
        if (anchor != null && anchor >= 0) listState.scrollToItem(anchor)
        else if (messages.isNotEmpty() && listState.firstVisibleItemIndex >= (messages.size - 3).coerceAtLeast(0)) listState.scrollToItem(messages.size - 1 + if (request != null) 1 else 0)
    }
    LazyColumn(modifier = modifier.fillMaxWidth(), state = listState, contentPadding = PaddingValues(horizontal = 18.dp, vertical = 16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        state.transientError?.let { error -> item("error") { MosaicErrorInline(error, onClearError) } }
        items(messages, key = { it.id }) { message -> MosaicMessageBubble(message, message.id == state.highlightedMessageId) }
        request?.let { active ->
            item("activity-${active.requestId}") { MosaicAgentTimeline(active) }
            if (active.streamedText.isNotBlank()) item("stream-${active.requestId}") { MosaicStreamingMessage(active) }
        }
        state.lastRequest?.takeIf { it.conversationId == state.activeConversationId }?.let { item("last-${it.requestId}") { MosaicAgentTimeline(it) } }
    }
}

@Composable
private fun MosaicMessageBubble(message: ChatMessage, highlighted: Boolean) {
    val user = message.role == "user"
    Row(Modifier.fillMaxWidth(), horizontalArrangement = if (user) Arrangement.End else Arrangement.Start) {
        Surface(
            modifier = Modifier.fillMaxWidth(if (user) 0.84f else 0.98f),
            color = if (user) MosaicTokens.UserBubble else MosaicTokens.Elevated,
            shape = if (user) RoundedCornerShape(18.dp, 18.dp, 5.dp, 18.dp) else RoundedCornerShape(16.dp),
            border = if (highlighted) BorderStroke(2.dp, MosaicTokens.Primary) else if (user) null else BorderStroke(1.dp, MosaicTokens.Border),
        ) {
            Column(Modifier.padding(14.dp)) {
                if (!user) Text("Mosaic", color = MosaicTokens.Text, fontSize = 12.sp, fontWeight = FontWeight.Bold, modifier = Modifier.padding(bottom = 7.dp))
                Text(message.content, color = MosaicTokens.Text, fontSize = 14.sp, lineHeight = 22.sp)
            }
        }
    }
}

@Composable
private fun MosaicStreamingMessage(request: ActiveRequest) {
    Surface(color = MosaicTokens.Elevated, shape = RoundedCornerShape(16.dp), border = BorderStroke(1.dp, MosaicTokens.Border), modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(14.dp)) {
            Row { Text("Mosaic", color = MosaicTokens.Text, fontWeight = FontWeight.Bold, fontSize = 12.sp); Spacer(Modifier.width(8.dp)); MosaicChip("Streaming") }
            Spacer(Modifier.height(8.dp)); Text(request.streamedText, color = MosaicTokens.Text, fontSize = 14.sp, lineHeight = 22.sp)
            if (request.phase == RequestPhase.Answering) Text("▋", color = MosaicTokens.Primary, fontSize = 15.sp)
        }
    }
}

@Composable
private fun MosaicAgentTimeline(request: ActiveRequest, modifier: Modifier = Modifier) {
    var expanded by remember(request.requestId) { mutableStateOf(request.phase !in setOf(RequestPhase.Completed, RequestPhase.Cancelled)) }
    MosaicCard(modifier, active = request.phase in setOf(RequestPhase.ProcessingPrompt, RequestPhase.RunningTools, RequestPhase.Answering), onClick = { expanded = !expanded }) {
        Column(Modifier.padding(14.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.size(30.dp).clip(RoundedCornerShape(12.dp)).background(Color(0xFF573989)), contentAlignment = Alignment.Center) { MosaicIcon(R.drawable.mosaic_ic_sparkles, "Request activity", tint = Color(0xFFD9B6FF)) }
                Spacer(Modifier.width(10.dp)); Column(Modifier.weight(1f)) { Text(activityHeadline(request), color = MosaicTokens.Text, fontSize = 13.sp, fontWeight = FontWeight.Bold); Text(elapsedText(request.startedElapsedMs, request.finishedElapsedMs), color = MosaicTokens.Secondary, fontSize = 10.sp) }
                MosaicIcon(R.drawable.mosaic_ic_chevron_down, if (expanded) "Collapse activity" else "Expand activity", tint = MosaicTokens.Text)
            }
            AnimatedVisibility(expanded) {
                Column(Modifier.padding(start = 8.dp, top = 14.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    request.activity.takeLast(6).forEach { item -> ActivityRow(item) }
                    request.errorMessage?.let { Text(it, color = MosaicTokens.Error, fontSize = 11.sp) }
                    request.sources.takeIf { it.isNotEmpty() }?.let { sources ->
                        Text("SOURCES FROM THIS REQUEST", color = MosaicTokens.Tertiary, fontSize = 10.sp, fontWeight = FontWeight.Bold, modifier = Modifier.padding(top = 4.dp))
                        sources.forEach { source -> MosaicSourceReference(source, Modifier.padding(top = 5.dp)) }
                    }
                }
            }
        }
    }
}

@Composable
private fun MosaicSourceReference(source: SourceSummary, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val canOpen = source.url != null
    Text(
        source.label,
        color = if (canOpen) MosaicTokens.AccentBlue else MosaicTokens.Secondary,
        fontSize = 11.sp,
        maxLines = 2,
        overflow = TextOverflow.Ellipsis,
        modifier = modifier.then(
            if (canOpen) Modifier.clickable {
                runCatching { context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(source.url))) }
            }.semantics { contentDescription = "Open source ${source.label}" } else Modifier,
        ),
    )
}

@Composable
private fun ActivityRow(item: ActivityItem) {
    val tint = when (item.status) { ActivityStatus.Running -> MosaicTokens.Primary; ActivityStatus.Completed -> MosaicTokens.Success; ActivityStatus.Failed -> MosaicTokens.Error }
    Row(verticalAlignment = Alignment.Top) {
        Box(Modifier.size(16.dp).clip(RoundedCornerShape(12.dp)).background(tint.copy(alpha = 0.18f)), contentAlignment = Alignment.Center) { MosaicIcon(if (item.status == ActivityStatus.Completed) R.drawable.mosaic_ic_check else R.drawable.mosaic_ic_sparkles, item.label, modifier = Modifier.size(11.dp), tint = tint) }
        Spacer(Modifier.width(10.dp)); Column { Text(item.label, color = MosaicTokens.Text, fontSize = 12.sp, fontWeight = FontWeight.SemiBold); item.detail?.let { Text(it, color = MosaicTokens.Secondary, fontSize = 10.sp) } }
    }
}

@Composable
private fun MosaicComposer(state: MosaicUiState, onDraft: (String) -> Unit, onSend: () -> Unit, onCancel: () -> Unit, onPick: () -> Unit, onRemove: () -> Unit) {
    val request = state.requestForActiveConversation
    val attached = state.selectedImage != null || state.selectedImageSource != null
    Surface(Modifier.fillMaxWidth().imePadding(), color = MosaicTokens.Background) {
        Column(Modifier.padding(horizontal = 13.dp, vertical = 10.dp)) {
            if (attached) Row(Modifier.padding(start = 12.dp, bottom = 6.dp), verticalAlignment = Alignment.CenterVertically) {
                MosaicIcon(R.drawable.mosaic_ic_image, "Attached image", tint = MosaicTokens.Primary); Spacer(Modifier.width(7.dp)); Text(state.selectedImageSource?.filename ?: "Selected image", color = MosaicTokens.Secondary, fontSize = 11.sp, modifier = Modifier.weight(1f)); MosaicIconButton(R.drawable.mosaic_ic_x, "Remove attachment", onRemove, tint = MosaicTokens.Secondary)
            }
            Surface(color = MosaicTokens.Composer, shape = RoundedCornerShape(26.dp), border = BorderStroke(1.dp, MosaicTokens.BorderEmphasis)) {
                Row(Modifier.padding(6.dp), verticalAlignment = Alignment.Bottom) {
                    MosaicIconButton(if (attached) R.drawable.mosaic_ic_x else R.drawable.mosaic_ic_plus, if (attached) "Remove selected image" else "Attach an image", if (attached) onRemove else onPick, tint = MosaicTokens.Primary, enabled = request == null)
                    Box(Modifier.weight(1f).padding(vertical = 10.dp)) {
                        if (state.draft.isBlank()) Text(if (attached) "Ask about this image…" else "Message Mosaic…", color = MosaicTokens.Secondary, fontSize = 14.sp)
                        BasicTextField(value = state.draft, onValueChange = onDraft, enabled = request == null, textStyle = androidx.compose.ui.text.TextStyle(color = MosaicTokens.Text, fontSize = 14.sp, lineHeight = 20.sp), minLines = 1, maxLines = 5, modifier = Modifier.fillMaxWidth())
                    }
                    if (request != null) ComposerAction(R.drawable.mosaic_ic_stop, "Stop generation", onCancel) else ComposerAction(R.drawable.mosaic_ic_send, "Send message", onSend, enabled = state.draft.isNotBlank() || attached)
                }
            }
        }
    }
}

@Composable
private fun ComposerAction(icon: Int, description: String, onClick: () -> Unit, enabled: Boolean = true) {
    Box(Modifier.size(43.dp).clip(RoundedCornerShape(24.dp)).background(if (enabled) MosaicTokens.Gradient else androidx.compose.ui.graphics.SolidColor(MosaicTokens.Border)).clickable(enabled = enabled, onClick = onClick).semantics { contentDescription = description }, contentAlignment = Alignment.Center) { MosaicIcon(icon, description, tint = Color.White) }
}

@Composable
private fun ConversationsScreen(state: MosaicUiState, wide: Boolean, openDrawer: (() -> Unit)?, onNew: () -> Unit, onSelect: (String) -> Unit, onRename: (String, String) -> Unit, onDelete: (String) -> Unit, modifier: Modifier) {
    var query by remember { mutableStateOf("") }
    var actionConversation by remember { mutableStateOf<ConversationSummary?>(null) }
    var renaming by remember { mutableStateOf(false) }
    var deleting by remember { mutableStateOf(false) }
    val filtered = state.conversations.filter { it.title.orEmpty().contains(query, ignoreCase = true) }
    Column(modifier.fillMaxSize().background(MosaicTokens.Background)) {
        MosaicTopBar("Conversations", if (wide) null else openDrawer, action = { MosaicIconButton(R.drawable.mosaic_ic_plus, "New conversation", onNew, tint = MosaicTokens.Primary) })
        Column(Modifier.padding(horizontal = 20.dp).weight(1f)) {
            MosaicSearchField(query, "Search conversations", { query = it })
            Spacer(Modifier.height(20.dp))
            if (filtered.isEmpty()) EmptyPane("No conversations match this search.", R.drawable.empty_conversations) else LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp), contentPadding = PaddingValues(bottom = 18.dp)) {
                items(filtered, key = { it.id }) { item ->
                    MosaicCard(active = item.id == state.activeConversationId, onClick = { onSelect(item.id) }) {
                        Row(Modifier.padding(14.dp), verticalAlignment = Alignment.CenterVertically) {
                            Box(Modifier.size(42.dp).clip(RoundedCornerShape(14.dp)).background(Color(0xFF342154)), contentAlignment = Alignment.Center) { MosaicIcon(R.drawable.mosaic_ic_message_square, "Conversation", tint = MosaicTokens.Primary) }
                            Spacer(Modifier.width(13.dp)); Column(Modifier.weight(1f)) { Text(item.title ?: "Untitled conversation", color = MosaicTokens.Text, fontSize = 14.sp, fontWeight = FontWeight.Bold, maxLines = 1, overflow = TextOverflow.Ellipsis); Text(relativeDate(item.updatedAt), color = MosaicTokens.Secondary, fontSize = 11.sp, modifier = Modifier.padding(top = 4.dp)) }
                            MosaicIconButton(R.drawable.mosaic_ic_more_vertical, "Conversation actions", { actionConversation = item }, tint = MosaicTokens.Secondary)
                        }
                    }
                }
            }
        }
        MosaicPrimaryButton("New conversation", onNew, Modifier.padding(20.dp), R.drawable.mosaic_ic_plus)
    }
    actionConversation?.let { item -> AlertDialog(onDismissRequest = { actionConversation = null }, title = { Text(item.title ?: "Conversation") }, text = { Text("Manage this stored conversation.") }, confirmButton = { MosaicPrimaryButton("Rename", { renaming = true }) }, dismissButton = { MosaicPrimaryButton("Delete", { deleting = true }) }) }
    if (renaming) { var title by remember(actionConversation?.id) { mutableStateOf(actionConversation?.title.orEmpty()) }; AlertDialog(onDismissRequest = { renaming = false }, title = { Text("Rename conversation") }, text = { MosaicSearchField(title, "Conversation title", { title = it }) }, confirmButton = { MosaicPrimaryButton("Save", { actionConversation?.let { onRename(it.id, title) }; renaming = false; actionConversation = null }, enabled = title.isNotBlank()) }, dismissButton = { MosaicPrimaryButton("Cancel", { renaming = false }) }) }
    if (deleting) AlertDialog(onDismissRequest = { deleting = false }, title = { Text("Delete this conversation?") }, text = { Text("This cannot be undone.") }, confirmButton = { MosaicPrimaryButton("Delete", { actionConversation?.let { onDelete(it.id) }; deleting = false; actionConversation = null }) }, dismissButton = { MosaicPrimaryButton("Cancel", { deleting = false }) })
}

@Composable
private fun MemoriesScreen(state: MosaicUiState, wide: Boolean, openDrawer: (() -> Unit)?, onRefresh: () -> Unit, onOpen: (String) -> Unit, modifier: Modifier) {
    var query by remember { mutableStateOf("") }; var category by remember { mutableStateOf("All") }
    val claims = state.memories.filter { claim ->
        (query.isBlank() || claim.fact.contains(query, true) || claim.subject.contains(query, true)) &&
            (category == "All" || claim.subjectType.contains(category.dropLastWhile { it == 's' }, true) || claim.assertionType.contains(category.dropLastWhile { it == 's' }, true))
    }
    Column(modifier.fillMaxSize().background(MosaicTokens.Background)) {
        MosaicTopBar("Memories", if (wide) null else openDrawer, action = { MosaicIconButton(R.drawable.mosaic_ic_refresh, "Refresh memories", onRefresh, tint = MosaicTokens.Primary, enabled = !state.memoriesLoading) })
        Column(Modifier.padding(horizontal = 20.dp).weight(1f)) {
            Text("What Mosaic remembers", color = MosaicTokens.Text, fontSize = 21.sp, fontWeight = FontWeight.Bold); Spacer(Modifier.height(8.dp)); Text("Connected facts from your conversations, with sources you can inspect.", color = MosaicTokens.Secondary, fontSize = 13.sp, lineHeight = 20.sp); Spacer(Modifier.height(16.dp))
            MosaicSearchField(query, "Search your knowledge", { query = it }); Spacer(Modifier.height(12.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(7.dp)) { listOf("All", "Current", "Tentative", "Historical").forEach { filter -> MosaicChip(filter, category == filter) { category = filter } } }
            Spacer(Modifier.height(16.dp)); state.memoryError?.let { MosaicErrorInline(it, onRefresh); Spacer(Modifier.height(12.dp)) }
            when { state.memoriesLoading -> LoadingPane("Loading saved memories…") ; claims.isEmpty() -> EmptyPane(if (query.isBlank()) "No extracted memories are available yet." else "No memories match this search.", R.drawable.empty_memories); else -> LazyColumn(verticalArrangement = Arrangement.spacedBy(11.dp), contentPadding = PaddingValues(bottom = 18.dp)) { items(claims, key = { it.id }) { MemoryRow(it, { onOpen(it.id) }) } } }
        }
    }
}

@Composable
private fun MemoryRow(claim: MemoryClaimSummary, onClick: () -> Unit) {
    MosaicCard(onClick = onClick) { Row(Modifier.padding(14.dp), verticalAlignment = Alignment.Top) {
        Box(Modifier.size(42.dp).clip(RoundedCornerShape(14.dp)).background(Color(0xFF342154)), contentAlignment = Alignment.Center) { MosaicIcon(R.drawable.mosaic_ic_brain, "Memory", tint = MosaicTokens.Primary) }
        Spacer(Modifier.width(13.dp)); Column(Modifier.weight(1f)) { Text(claim.fact, color = MosaicTokens.Text, fontSize = 14.sp, lineHeight = 20.sp, fontWeight = FontWeight.Bold); Spacer(Modifier.height(9.dp)); Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) { ClaimStatus(claim.status); MosaicChip(claim.subjectType) }; Text("${claim.assertionType.replaceFirstChar { it.uppercase() }} · ${claim.evidenceCount} source${if (claim.evidenceCount == 1) "" else "s"}", color = MosaicTokens.Secondary, fontSize = 11.sp, modifier = Modifier.padding(top = 8.dp)) }
        MosaicIcon(R.drawable.mosaic_ic_chevron_right, "Open memory", tint = MosaicTokens.Secondary)
    } }
}

@Composable
private fun ClaimStatus(status: String) {
    val color = when (status.lowercase()) { "current" -> MosaicTokens.Success; "tentative" -> MosaicTokens.Warning; else -> MosaicTokens.Tertiary }
    MosaicStatusPill(status.replaceFirstChar { it.uppercase() }, color)
}

@Composable
private fun MemoryDetailScreen(state: MosaicUiState, wide: Boolean, openDrawer: (() -> Unit)?, onBack: () -> Unit, onOpenEvidence: (EvidenceItem) -> Unit, onDelete: (String) -> Unit, modifier: Modifier) {
    val detail = state.selectedMemory; var evidenceTab by remember { mutableStateOf(true) }; var delete by remember { mutableStateOf(false) }
    Column(modifier.fillMaxSize().background(MosaicTokens.Background)) {
        MosaicTopBar("Memory detail", if (wide) onBack else onBack, R.drawable.mosaic_ic_arrow_left, action = { if (detail != null) MosaicIconButton(R.drawable.mosaic_ic_trash, "Delete memory", { delete = true }, tint = MosaicTokens.Error) })
        when { state.memoryDetailLoading -> LoadingPane("Loading provenance…", Modifier.weight(1f)); detail == null -> EmptyPane("This memory is no longer available.", R.drawable.empty_memories, Modifier.weight(1f)); else -> LazyColumn(Modifier.weight(1f), contentPadding = PaddingValues(20.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
            item { Text(detail.summary.subjectType.uppercase(), color = MosaicTokens.Primary, fontSize = 11.sp, fontWeight = FontWeight.Bold, letterSpacing = 1.2.sp); Spacer(Modifier.height(12.dp)); Text(detail.summary.fact, color = MosaicTokens.Text, fontSize = 23.sp, lineHeight = 29.sp, fontWeight = FontWeight.Bold); Spacer(Modifier.height(14.dp)); Row(horizontalArrangement = Arrangement.spacedBy(7.dp)) { ClaimStatus(detail.summary.status); MosaicChip(detail.summary.assertionType); MosaicChip("${detail.evidence.size} sources") } }
            item { Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(15.dp)).background(MosaicTokens.Elevated).padding(4.dp)) { Segment("Details", !evidenceTab) { evidenceTab = false }; Segment("Evidence", evidenceTab) { evidenceTab = true } } }
            if (!evidenceTab) { item { MosaicCard { Column(Modifier.padding(16.dp)) { Text("RELATIONSHIPS", color = MosaicTokens.Secondary, fontSize = 11.sp, fontWeight = FontWeight.Bold); Spacer(Modifier.height(10.dp)); DetailRow("Subject", detail.summary.subject); DetailRow("Predicate", detail.summary.predicate); DetailRow("Type", detail.summary.subjectType); DetailRow("Assertion", detail.summary.assertionType) } } } }
            else { item { Text("ORIGINAL CONVERSATION EVIDENCE", color = MosaicTokens.Secondary, fontSize = 11.sp, fontWeight = FontWeight.Bold) }; items(detail.evidence, key = { it.id }) { EvidenceCard(it, { onOpenEvidence(it) }) } }
        } }
    }
    if (delete && detail != null) AlertDialog(onDismissRequest = { delete = false }, title = { Text("Delete this memory?") }, text = { Text("This removes only the derived memory. Its original conversation evidence remains unchanged.") }, confirmButton = { MosaicPrimaryButton("Delete memory", { onDelete(detail.summary.id); delete = false }) }, dismissButton = { MosaicPrimaryButton("Cancel", { delete = false }) })
}

@Composable
private fun androidx.compose.foundation.layout.RowScope.Segment(label: String, selected: Boolean, onClick: () -> Unit) = Surface(Modifier.weight(1f).height(44.dp).clip(RoundedCornerShape(11.dp)).clickable(onClick = onClick), color = if (selected) Color(0xFF65409D) else Color.Transparent, shape = RoundedCornerShape(11.dp)) { Box(contentAlignment = Alignment.Center) { Text(label, color = if (selected) Color.White else MosaicTokens.Secondary, fontSize = 12.sp, fontWeight = FontWeight.Bold) } }

@Composable
private fun DetailRow(label: String, value: String) = Row(Modifier.fillMaxWidth().padding(vertical = 6.dp)) { Text(label, color = MosaicTokens.Tertiary, fontSize = 12.sp, modifier = Modifier.width(100.dp)); Text(value, color = MosaicTokens.Text, fontSize = 12.sp, modifier = Modifier.weight(1f)) }

@Composable
private fun EvidenceCard(evidence: EvidenceItem, onOpen: () -> Unit) {
    MosaicCard { Column(Modifier.padding(16.dp)) {
        Row { Column(Modifier.weight(1f)) { Text(evidence.conversationTitle ?: "Conversation source", color = MosaicTokens.Text, fontSize = 13.sp, fontWeight = FontWeight.Bold); Text(evidence.messageCreatedAt ?: "Saved evidence", color = MosaicTokens.Secondary, fontSize = 10.sp, modifier = Modifier.padding(top = 3.dp)) }; ClaimStatus(if (evidence.spanAvailable) "Exact span" else "Message") }
        Spacer(Modifier.height(12.dp)); Surface(color = MosaicTokens.Active, shape = RoundedCornerShape(12.dp), border = BorderStroke(1.dp, MosaicTokens.Border)) { Text("“${evidence.quote ?: evidence.messageContent ?: evidence.unavailableReason ?: "Source no longer available"}”", color = MosaicTokens.Text, fontSize = 13.sp, lineHeight = 20.sp, fontStyle = androidx.compose.ui.text.font.FontStyle.Italic, modifier = Modifier.padding(14.dp)) }
        Spacer(Modifier.height(12.dp)); MosaicPrimaryButton("View original conversation", onOpen, icon = R.drawable.mosaic_ic_external_link, enabled = evidence.unavailableReason == null)
    } }
}

@Composable
private fun ImagesScreen(state: MosaicUiState, wide: Boolean, openDrawer: (() -> Unit)?, onRefresh: () -> Unit, onSearch: (String) -> Unit, onOpen: (IndexedImage) -> Unit, onPick: () -> Unit, onAddToIndex: () -> Unit, modifier: Modifier) {
    var query by remember { mutableStateOf("") }; var askTab by remember { mutableStateOf(false) }
    Column(modifier.fillMaxSize().background(MosaicTokens.Background)) {
        MosaicTopBar("Images", if (wide) null else openDrawer, action = {
            Row(horizontalArrangement = Arrangement.spacedBy(2.dp)) {
                MosaicIconButton(R.drawable.mosaic_ic_plus, "Add image to index", onAddToIndex, tint = MosaicTokens.Primary, enabled = !state.imagesLoading)
                MosaicIconButton(R.drawable.mosaic_ic_refresh, "Refresh images", onRefresh, tint = MosaicTokens.Primary, enabled = !state.imagesLoading)
            }
        })
        Column(Modifier.padding(horizontal = 20.dp).weight(1f)) {
            Text("See more with Mosaic", color = MosaicTokens.Text, fontSize = 21.sp, fontWeight = FontWeight.Bold); Spacer(Modifier.height(8.dp)); Text("Ask Qwen about an image or find images you have explicitly indexed.", color = MosaicTokens.Secondary, fontSize = 13.sp, lineHeight = 20.sp); Spacer(Modifier.height(16.dp))
            Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(15.dp)).background(MosaicTokens.Elevated).padding(4.dp)) { Segment("Indexed", !askTab) { askTab = false }; Segment("Ask about a photo", askTab) { askTab = true } }
            Spacer(Modifier.height(16.dp))
            if (askTab) AskPhotoPane(onPick) else {
                Text("SEARCH INDEXED IMAGES", color = MosaicTokens.Secondary, fontSize = 11.sp, fontWeight = FontWeight.Bold); Spacer(Modifier.height(8.dp)); MosaicSearchField(query, "e.g. photo of a cat", { query = it }); Spacer(Modifier.height(8.dp)); MosaicPrimaryButton("Search images", { onSearch(query) }, icon = R.drawable.mosaic_ic_search, enabled = query.isNotBlank() && !state.imagesLoading)
                MosaicPrimaryButton("Add image to index", onAddToIndex, Modifier.padding(top = 8.dp), R.drawable.mosaic_ic_plus, enabled = !state.imagesLoading)
                state.imageSearchQuery?.let { Text("Visual similarity search for “$it”${state.imageSearchElapsedMs?.let { ms -> " · ${ms.toInt()} ms" }.orEmpty()}", color = MosaicTokens.Secondary, fontSize = 11.sp, modifier = Modifier.padding(vertical = 12.dp)) }
                state.imageNotice?.let { Text(it, color = MosaicTokens.Success, fontSize = 11.sp, modifier = Modifier.padding(vertical = 12.dp)) }
                state.imageError?.let { MosaicErrorInline(it, onRefresh) }
                when { state.imagesLoading -> LoadingPane(if (state.imageIndexing) "Indexing the selected image with EmbeddingGemma…" else "Searching indexed images with EmbeddingGemma…", Modifier.weight(1f)); state.indexedImages.isEmpty() -> EmptyPane(if (state.imageSearchQuery == null) "No indexed images yet." else "No indexed images match this query.", R.drawable.empty_images, Modifier.weight(1f)); else -> ImageGrid(state.indexedImages, onOpen, Modifier.weight(1f)) }
            }
        }
    }
}

@Composable
private fun AskPhotoPane(onPick: () -> Unit) = Column(Modifier.fillMaxWidth(), horizontalAlignment = Alignment.CenterHorizontally) {
    MosaicCard { Column(Modifier.padding(24.dp), horizontalAlignment = Alignment.CenterHorizontally) { Box(Modifier.size(54.dp).clip(RoundedCornerShape(18.dp)).background(Color(0xFF342154)), contentAlignment = Alignment.Center) { MosaicIcon(R.drawable.mosaic_ic_image, "Select a photo", tint = MosaicTokens.Primary) }; Spacer(Modifier.height(14.dp)); Text("Choose a photo", color = MosaicTokens.Text, fontWeight = FontWeight.Bold, fontSize = 15.sp); Spacer(Modifier.height(6.dp)); Text("Mosaic analyzes only images you explicitly select. It never scans your gallery.", color = MosaicTokens.Secondary, fontSize = 12.sp, lineHeight = 18.sp, textAlign = TextAlign.Center); Spacer(Modifier.height(18.dp)); MosaicPrimaryButton("Choose image", onPick, icon = R.drawable.mosaic_ic_image) } }
}

@Composable
private fun ImageGrid(images: List<IndexedImage>, onOpen: (IndexedImage) -> Unit, modifier: Modifier) = LazyColumn(modifier, contentPadding = PaddingValues(vertical = 12.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) { items(images.chunked(2), key = { row -> row.joinToString { it.id } }) { row -> Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) { row.forEach { image -> MosaicImageTile(image, { onOpen(image) }, Modifier.weight(1f)) }; if (row.size == 1) Spacer(Modifier.weight(1f)) } } }

@Composable
private fun MosaicImageTile(image: IndexedImage, onClick: () -> Unit, modifier: Modifier) {
    val bitmap = remember(image.path) { image.path?.let { runCatching { BitmapFactory.decodeFile(it)?.asImageBitmap() }.getOrNull() } }
    MosaicCard(modifier, onClick = onClick) { Column { Box(Modifier.fillMaxWidth().height(142.dp).clip(RoundedCornerShape(topStart = 15.dp, topEnd = 15.dp)).background(MosaicTokens.Active)) { if (bitmap != null) Image(bitmap, image.filename, Modifier.fillMaxSize(), contentScale = ContentScale.Crop) else Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { MosaicIcon(R.drawable.mosaic_ic_image, "Image unavailable", tint = MosaicTokens.Tertiary) } }; Column(Modifier.padding(10.dp)) { Text(image.filename, color = MosaicTokens.Text, fontSize = 12.sp, fontWeight = FontWeight.Bold, maxLines = 1, overflow = TextOverflow.Ellipsis); image.similarity?.let { Text("Similarity %.2f".format(it), color = MosaicTokens.Secondary, fontSize = 10.sp, modifier = Modifier.padding(top = 4.dp)) } } } }
}

@Composable
private fun ImageDetailScreen(state: MosaicUiState, wide: Boolean, openDrawer: (() -> Unit)?, onBack: () -> Unit, onAsk: () -> Unit, modifier: Modifier) {
    val image = state.selectedIndexedImage
    Column(modifier.fillMaxSize().background(MosaicTokens.Background)) {
        MosaicTopBar("Photo details", if (wide) onBack else onBack, R.drawable.mosaic_ic_arrow_left)
        if (image == null) EmptyPane("This image is no longer selected.", R.drawable.empty_images, Modifier.weight(1f)) else {
            val bitmap = remember(image.path) { image.path?.let { runCatching { BitmapFactory.decodeFile(it)?.asImageBitmap() }.getOrNull() } }
            LazyColumn(Modifier.weight(1f), contentPadding = PaddingValues(20.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                item { Surface(color = MosaicTokens.Elevated, shape = RoundedCornerShape(18.dp), border = BorderStroke(1.dp, MosaicTokens.Border)) { if (bitmap != null) Image(bitmap, image.filename, Modifier.fillMaxWidth().height(300.dp), contentScale = ContentScale.Fit) else Box(Modifier.fillMaxWidth().height(220.dp), contentAlignment = Alignment.Center) { Text("Preview unavailable", color = MosaicTokens.Secondary) } } }
                item { Row { MosaicStatusPill("Indexed image", MosaicTokens.Success); Spacer(Modifier.weight(1f)); MosaicStatusPill(if (image.available) "Source available" else "Source unavailable", if (image.available) MosaicTokens.Secondary else MosaicTokens.Warning) } }
                item { Text("ASK ABOUT THIS IMAGE", color = MosaicTokens.Secondary, fontSize = 11.sp, fontWeight = FontWeight.Bold) }
                item { MosaicCard { Column(Modifier.padding(16.dp)) { Text("What is happening in this image?", color = MosaicTokens.Text, fontSize = 16.sp, fontWeight = FontWeight.Bold); Spacer(Modifier.height(8.dp)); Text("Qwen vision examines the actual stored pixels. Image retrieval scores alone are not image understanding.", color = MosaicTokens.Secondary, fontSize = 12.sp, lineHeight = 18.sp); Spacer(Modifier.height(16.dp)); MosaicPrimaryButton("Ask in conversation", onAsk, icon = R.drawable.mosaic_ic_arrow_right) } } }
                item { Text("SOURCE", color = MosaicTokens.Secondary, fontSize = 11.sp, fontWeight = FontWeight.Bold) }
                item { MosaicCard { Column(Modifier.padding(16.dp)) { Text(image.filename, color = MosaicTokens.Text, fontSize = 14.sp, fontWeight = FontWeight.Bold); Text("${image.origin} · ${if (image.available) "Source available" else "Source no longer available"}", color = MosaicTokens.Secondary, fontSize = 12.sp, modifier = Modifier.padding(top = 6.dp)) } } }
            }
        }
    }
}

@Composable
private fun SkillsScreen(
    state: MosaicUiState,
    wide: Boolean,
    openDrawer: (() -> Unit)?,
    onTitleChanged: (String) -> Unit,
    onDescriptionChanged: (String) -> Unit,
    onAdd: () -> Unit,
    onEdit: (CustomCapability) -> Unit,
    onCancelEdit: () -> Unit,
    onDelete: (String) -> Unit,
    onEditBuiltIn: (BuiltInSkill) -> Unit,
    onResetBuiltIn: (String) -> Unit,
    modifier: Modifier,
) {
    val editingBuiltIn = state.editingBuiltInSkillKey != null
    val editing = state.editingCapabilityId != null || editingBuiltIn
    Column(modifier.fillMaxSize().background(MosaicTokens.Background)) {
        MosaicTopBar("Skills", if (wide) null else openDrawer)
        LazyColumn(
            Modifier.padding(horizontal = 20.dp),
            contentPadding = PaddingValues(bottom = 24.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            item {
                Text("Local agent skills", color = MosaicTokens.Text, fontSize = 21.sp, fontWeight = FontWeight.Bold, modifier = Modifier.padding(top = 14.dp))
                Text("Review the packaged tasks, create a custom responder skill, or save a local edit. Python keeps tools and permissions fixed.", color = MosaicTokens.Secondary, fontSize = 13.sp, lineHeight = 20.sp, modifier = Modifier.padding(top = 8.dp))
            }
            item { SectionLabel("BUILT-IN SKILLS") }
            if (state.capabilitiesLoading) {
                item { Text("Loading local skills…", color = MosaicTokens.Secondary, fontSize = 12.sp) }
            } else {
                items(state.builtInSkills, key = { it.key }) { skill ->
                    MosaicCard {
                        Row(Modifier.padding(16.dp), verticalAlignment = Alignment.Top) {
                            Column(Modifier.weight(1f)) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Text(skill.title, color = MosaicTokens.Text, fontSize = 14.sp, fontWeight = FontWeight.Bold)
                                    Spacer(Modifier.width(8.dp))
                                    MosaicStatusPill(if (skill.hasLocalOverride) "Edited locally" else "Packaged", if (skill.hasLocalOverride) MosaicTokens.Warning else MosaicTokens.Success)
                                }
                                Text(skill.version, color = MosaicTokens.Tertiary, fontSize = 10.sp, modifier = Modifier.padding(top = 4.dp))
                                Text(skill.instructions, color = MosaicTokens.Secondary, fontSize = 12.sp, lineHeight = 18.sp, maxLines = 3, overflow = TextOverflow.Ellipsis, modifier = Modifier.padding(top = 7.dp))
                                Text(if (skill.allowedTools.isEmpty()) "No model tools" else "Authorized tools: ${skill.allowedTools.joinToString()}", color = MosaicTokens.Tertiary, fontSize = 10.sp, modifier = Modifier.padding(top = 9.dp))
                            }
                            Row {
                                MosaicIconButton(R.drawable.mosaic_ic_pencil, "Edit ${skill.title}", { onEditBuiltIn(skill) }, tint = MosaicTokens.Primary)
                                if (skill.hasLocalOverride) MosaicIconButton(R.drawable.mosaic_ic_refresh, "Restore packaged ${skill.title}", { onResetBuiltIn(skill.key) }, tint = MosaicTokens.Tertiary)
                            }
                        }
                    }
                }
            }
            item { SectionLabel(if (editingBuiltIn) "EDIT BUILT-IN SKILL" else "CREATE OR EDIT CUSTOM SKILL") }
            item {
                MosaicCard {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        Text(if (editingBuiltIn) "Edit ${state.capabilityTitleDraft}" else if (editing) "Edit custom skill" else "Add a custom skill", color = MosaicTokens.Text, fontSize = 15.sp, fontWeight = FontWeight.Bold)
                        Text(if (editingBuiltIn) "Built-in skill names and tool permissions are fixed. You can edit its local instructions or restore the packaged version." else "Custom skills are local responder instructions. They guide future answers but cannot create new tools or grant device access.", color = MosaicTokens.Secondary, fontSize = 12.sp, lineHeight = 18.sp)
                        MosaicFormField(state.capabilityTitleDraft, "Skill name", onTitleChanged, enabled = !editingBuiltIn)
                        MosaicFormField(state.capabilityDescriptionDraft, "Skill instructions", onDescriptionChanged, Modifier.height(112.dp), singleLine = false)
                        MosaicPrimaryButton(
                            if (state.capabilitySaving) "Saving…" else if (editing) "Save changes" else "Save custom skill",
                            onAdd,
                            icon = R.drawable.mosaic_ic_plus,
                            enabled = !state.capabilitySaving,
                        )
                        if (editing) MosaicChip("Cancel editing", onClick = onCancelEdit)
                    }
                }
            }
            state.capabilityError?.let { error -> item { MosaicErrorInline(error) } }
            state.capabilityNotice?.let { notice -> item { Text(notice, color = MosaicTokens.Success, fontSize = 12.sp, lineHeight = 18.sp) } }
            item { SectionLabel("CUSTOM SKILLS") }
            if (state.capabilitiesLoading) {
                item { Text("Loading custom skills…", color = MosaicTokens.Secondary, fontSize = 12.sp) }
            } else if (state.customCapabilities.isEmpty()) {
                item { Text("No custom skills saved yet.", color = MosaicTokens.Tertiary, fontSize = 12.sp) }
            } else {
                items(state.customCapabilities, key = { it.id }) { capability ->
                    MosaicCard {
                        Row(Modifier.padding(16.dp), verticalAlignment = Alignment.Top) {
                            Column(Modifier.weight(1f)) {
                                Text(capability.title, color = MosaicTokens.Text, fontSize = 14.sp, fontWeight = FontWeight.Bold)
                                Text(capability.description, color = MosaicTokens.Secondary, fontSize = 12.sp, lineHeight = 18.sp, modifier = Modifier.padding(top = 6.dp))
                                Text("Included in future local Qwen responder requests", color = MosaicTokens.Tertiary, fontSize = 10.sp, modifier = Modifier.padding(top = 10.dp))
                            }
                            Row {
                                MosaicIconButton(R.drawable.mosaic_ic_pencil, "Edit ${capability.title}", { onEdit(capability) }, tint = MosaicTokens.Primary)
                                MosaicIconButton(R.drawable.mosaic_ic_trash, "Remove ${capability.title}", { onDelete(capability.id) }, tint = MosaicTokens.Tertiary)
                            }
                        }
                    }
                }
            }
            item { Text("Capabilities describe installed features. Skills are task instructions. Python keeps model tools fixed and read-only even when a skill is edited.", color = MosaicTokens.Tertiary, fontSize = 11.sp, lineHeight = 16.sp) }
        }
    }
}

@Composable
private fun CapabilityRow(icon: Int, title: String, detail: String, availability: String) = Row(Modifier.fillMaxWidth().padding(vertical = 16.dp), verticalAlignment = Alignment.Top) { Box(Modifier.size(48.dp).clip(RoundedCornerShape(16.dp)).background(Color(0xFF342154)), contentAlignment = Alignment.Center) { MosaicIcon(icon, title, tint = MosaicTokens.Primary) }; Spacer(Modifier.width(14.dp)); Column(Modifier.weight(1f)) { Text(title, color = MosaicTokens.Text, fontSize = 15.sp, fontWeight = FontWeight.Bold); Text(detail, color = MosaicTokens.Secondary, fontSize = 12.sp, lineHeight = 18.sp, modifier = Modifier.padding(top = 5.dp)); Spacer(Modifier.height(8.dp)); MosaicStatusPill(availability, when (availability) { "Available" -> MosaicTokens.Success; "Limited" -> MosaicTokens.Warning; else -> MosaicTokens.Error }) } }

@Composable
private fun SettingsScreen(
    state: MosaicUiState,
    wide: Boolean,
    openDrawer: (() -> Unit)?,
    onImport: () -> Unit,
    onDeepSeekApiKeyChanged: (String) -> Unit,
    onSaveDeepSeekApiKey: () -> Unit,
    onClearDeepSeekApiKey: () -> Unit,
    modifier: Modifier,
) {
    val local = state.localStatus
    Column(modifier.fillMaxSize().background(MosaicTokens.Background)) { MosaicTopBar("Settings", if (wide) null else openDrawer); LazyColumn(Modifier.padding(horizontal = 20.dp), contentPadding = PaddingValues(bottom = 24.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
        item { Text("Your device. Your data.", color = MosaicTokens.Text, fontSize = 21.sp, fontWeight = FontWeight.Bold, modifier = Modifier.padding(top = 14.dp)); Text("Mosaic stores conversation history and indexed knowledge locally.", color = MosaicTokens.Secondary, fontSize = 13.sp, lineHeight = 20.sp, modifier = Modifier.padding(top = 8.dp)) }
        item { SectionLabel("INFERENCE") }; item { MosaicCard { Column(Modifier.padding(16.dp)) { SettingRow(R.drawable.mosaic_ic_cpu, "Qwen 3.5 4B", runtimeDescription(state.runtimeStatus), runtimeColor(state.runtimeStatus)); Spacer(Modifier.height(14.dp)); SettingDivider(); Spacer(Modifier.height(14.dp)); local?.models?.get("embeddinggemma")?.let { status -> SettingRow(R.drawable.mosaic_ic_brain, "EmbeddingGemma 2", bytesLabel(status.bytes), if (status.present) MosaicTokens.Success else MosaicTokens.Warning) } ?: Text("Reading embedding model status…", color = MosaicTokens.Secondary, fontSize = 12.sp) } } }
        if (state.runtimeStatus !is RuntimeStatus.Ready) item { MosaicPrimaryButton("Import Qwen model", onImport, icon = R.drawable.mosaic_ic_plus) }
        item { SectionLabel("WEB SEARCH") }
        item {
            MosaicCard {
                Column(Modifier.padding(16.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        MosaicIcon(R.drawable.mosaic_ic_external_link, "DeepSeek web search", tint = MosaicTokens.Primary)
                        Spacer(Modifier.width(12.dp))
                        Column(Modifier.weight(1f)) {
                            Text("DeepSeek Search & Fetch", color = MosaicTokens.Text, fontSize = 14.sp, fontWeight = FontWeight.Bold)
                            Text("Optional public-web tool for local Qwen.", color = MosaicTokens.Secondary, fontSize = 11.sp, modifier = Modifier.padding(top = 4.dp))
                        }
                        MosaicStatusPill(if (local?.webSearchConfigured == true) "Configured" else "Not configured", if (local?.webSearchConfigured == true) MosaicTokens.Success else MosaicTokens.Warning)
                    }
                    Spacer(Modifier.height(12.dp))
                    Text("Only a Qwen-selected search query or a selected result URL is sent to DeepSeek. Local memories are never sent automatically. DeepSeek is not used for model inference.", color = MosaicTokens.Secondary, fontSize = 12.sp, lineHeight = 18.sp)
                    Spacer(Modifier.height(14.dp))
                    MosaicSecretField(
                        value = state.deepSeekApiKeyDraft,
                        placeholder = if (local?.webSearchConfigured == true) "Replace DeepSeek API key" else "DeepSeek API key",
                        onValueChange = onDeepSeekApiKeyChanged,
                        enabled = !state.webSearchSaving,
                    )
                    Spacer(Modifier.height(10.dp))
                    MosaicPrimaryButton(
                        if (local?.webSearchConfigured == true) "Replace key" else "Save key securely",
                        onSaveDeepSeekApiKey,
                        enabled = state.deepSeekApiKeyDraft.isNotBlank() && !state.webSearchSaving,
                    )
                    if (local?.webSearchConfigured == true) {
                        Spacer(Modifier.height(9.dp))
                        MosaicPrimaryButton("Remove saved key", onClearDeepSeekApiKey, enabled = !state.webSearchSaving)
                    }
                    state.webSearchError?.let { error -> Spacer(Modifier.height(10.dp)); MosaicErrorInline(error) }
                    state.webSearchNotice?.let { notice -> Spacer(Modifier.height(10.dp)); Text(notice, color = MosaicTokens.Success, fontSize = 11.sp, lineHeight = 16.sp) }
                }
            }
        }
        item { SectionLabel("PREFERENCES") }; item { MosaicCard { Column(Modifier.padding(16.dp)) { SettingText("Appearance", "Dark theme") ; SettingDivider(); SettingText("Memory privacy", "Stored in app-private SQLite storage") ; SettingDivider(); SettingText("Selected photos", "Only images you explicitly provide can be analyzed") } } }
        item { SectionLabel("ABOUT") }; item { MosaicCard { Column(Modifier.padding(16.dp)) { Text("Mosaic v0.1", color = MosaicTokens.Text, fontSize = 14.sp, fontWeight = FontWeight.Bold); Text("Local personal AI — Android proof of product", color = MosaicTokens.Secondary, fontSize = 12.sp, modifier = Modifier.padding(top = 6.dp)); local?.let { Text("${it.conversations} conversations · ${it.memories} memories · ${it.indexedImages} indexed images · ${bytesLabel(it.databaseBytes)}", color = MosaicTokens.Tertiary, fontSize = 11.sp, modifier = Modifier.padding(top = 12.dp)) } } } }
    } }
}

@Composable
private fun SectionLabel(value: String) = Text(value, color = MosaicTokens.Secondary, fontSize = 11.sp, fontWeight = FontWeight.Bold, modifier = Modifier.padding(top = 6.dp))
@Composable
private fun SettingDivider() = Box(Modifier.fillMaxWidth().height(1.dp).background(MosaicTokens.Border).padding(vertical = 14.dp))
@Composable
private fun SettingText(title: String, detail: String) = Column(Modifier.padding(vertical = 8.dp)) { Text(title, color = MosaicTokens.Text, fontSize = 13.sp, fontWeight = FontWeight.Bold); Text(detail, color = MosaicTokens.Secondary, fontSize = 12.sp, modifier = Modifier.padding(top = 5.dp)) }
@Composable
private fun SettingRow(icon: Int, title: String, detail: String, tint: Color) = Row(verticalAlignment = Alignment.CenterVertically) { MosaicIcon(icon, title, tint = MosaicTokens.Primary); Spacer(Modifier.width(12.dp)); Column(Modifier.weight(1f)) { Text(title, color = MosaicTokens.Text, fontSize = 14.sp, fontWeight = FontWeight.Bold); Text(detail, color = MosaicTokens.Secondary, fontSize = 11.sp, modifier = Modifier.padding(top = 4.dp)) }; MosaicStatusPill(if (tint == MosaicTokens.Success) "On-device" else "Unavailable", tint) }

@Composable
private fun MosaicContextRail(state: MosaicUiState, modifier: Modifier) {
    Surface(modifier, color = MosaicTokens.Surface, border = BorderStroke(1.dp, MosaicTokens.Border)) { Column(Modifier.fillMaxSize().statusBarsPadding().padding(16.dp)) { Text("CONTEXT", color = MosaicTokens.Secondary, fontSize = 11.sp, fontWeight = FontWeight.Bold); Spacer(Modifier.height(12.dp)); state.requestForActiveConversation?.let { MosaicAgentTimeline(it) } ?: state.lastRequest?.takeIf { it.conversationId == state.activeConversationId }?.let { MosaicAgentTimeline(it) } ?: Text("Request activity and source references appear here while you work.", color = MosaicTokens.Tertiary, fontSize = 12.sp, lineHeight = 18.sp); val sources = state.requestForActiveConversation?.sources ?: state.lastRequest?.sources.orEmpty(); if (sources.isNotEmpty()) { Spacer(Modifier.height(20.dp)); Text("SOURCES", color = MosaicTokens.Secondary, fontSize = 11.sp, fontWeight = FontWeight.Bold); sources.forEach { source -> MosaicCard(modifier = Modifier.padding(top = 9.dp)) { Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) { MosaicIcon(if (source.kind == "image") R.drawable.mosaic_ic_image else R.drawable.mosaic_ic_external_link, source.label, tint = MosaicTokens.Primary); Spacer(Modifier.width(8.dp)); MosaicSourceReference(source, Modifier.weight(1f)) } } } } } }
}

@Composable
private fun RuntimePill(status: RuntimeStatus, modifier: Modifier = Modifier) = MosaicStatusPill(runtimeDescription(status), runtimeColor(status), modifier)
private fun runtimeDescription(status: RuntimeStatus): String = when (status) { RuntimeStatus.Starting -> "Starting local runtime"; is RuntimeStatus.Ready -> status.label; is RuntimeStatus.Unavailable -> status.message; is RuntimeStatus.Error -> status.message }
private fun runtimeColor(status: RuntimeStatus): Color = when (status) { RuntimeStatus.Starting -> MosaicTokens.Warning; is RuntimeStatus.Ready -> MosaicTokens.Success; is RuntimeStatus.Unavailable -> MosaicTokens.Warning; is RuntimeStatus.Error -> MosaicTokens.Error }
private fun activityHeadline(request: ActiveRequest): String = when (request.phase) { RequestPhase.Queued -> "Working on your request"; RequestPhase.LoadingModel -> "Loading local model"; RequestPhase.ProcessingPrompt -> "Processing request locally"; RequestPhase.RunningTools -> "Using local tools"; RequestPhase.Answering -> "Generating answer locally"; RequestPhase.Cancelling -> "Stopping generation"; RequestPhase.Completed -> "Completed locally"; RequestPhase.Cancelled -> "Interrupted — not saved as complete"; RequestPhase.Failed -> "Request failed" }
@Composable
private fun elapsedText(start: Long, finished: Long?): String {
    val now by produceState(finished ?: SystemClock.elapsedRealtime(), start, finished) {
        if (finished == null) while (true) {
            value = SystemClock.elapsedRealtime()
            delay(500)
        }
    }
    val seconds = (((finished ?: now) - start).coerceAtLeast(0) / 1000)
    return if (seconds < 60) "${seconds}s elapsed" else "${seconds / 60}m ${seconds % 60}s elapsed"
}
private fun bytesLabel(bytes: Long): String = if (bytes >= 1024L * 1024L * 1024L) "%.2f GB".format(bytes / (1024.0 * 1024 * 1024)) else "%.1f MB".format(bytes / (1024.0 * 1024 * 1024))
private fun relativeDate(value: String): String = value.replace('T', ' ').substringBefore('.')
@Composable
private fun LoadingPane(text: String, modifier: Modifier = Modifier) = Box(modifier.fillMaxSize(), contentAlignment = Alignment.Center) { Text(text, color = MosaicTokens.Secondary, fontSize = 13.sp, textAlign = TextAlign.Center) }
@Composable
private fun EmptyPane(text: String, illustration: Int, modifier: Modifier = Modifier) = Column(modifier.fillMaxSize().padding(30.dp), verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally) { Image(painterResource(illustration), null, Modifier.size(100.dp)); Spacer(Modifier.height(15.dp)); Text(text, color = MosaicTokens.Secondary, fontSize = 13.sp, textAlign = TextAlign.Center, lineHeight = 20.sp) }

@Composable
private fun mosaicColorScheme() = MaterialTheme.colorScheme.copy(primary = MosaicTokens.Primary, onPrimary = Color.White, background = MosaicTokens.Background, onBackground = MosaicTokens.Text, surface = MosaicTokens.Surface, onSurface = MosaicTokens.Text, surfaceVariant = MosaicTokens.Elevated, onSurfaceVariant = MosaicTokens.Secondary, outline = MosaicTokens.Border, error = MosaicTokens.Error)

package dev.mosaic.poc

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.compose.collectAsStateWithLifecycle

/** Hosts Compose only; all Mosaic state and Python/JNI calls live below it. */
class MainActivity : ComponentActivity() {
    private val viewModel: MosaicViewModel by viewModels {
        object : ViewModelProvider.Factory {
            @Suppress("UNCHECKED_CAST")
            override fun <T : ViewModel> create(modelClass: Class<T>): T =
                MosaicViewModel(MosaicRepository(applicationContext)) as T
        }
    }

    private val qwenPicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) viewModel.importQwen(uri)
    }

    private val imagePicker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) viewModel.selectImage(uri)
    }

    private val imageIndexPicker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) viewModel.indexSelectedImage(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            val state = viewModel.state.collectAsStateWithLifecycle().value
            MosaicApp(
                state = state,
                onNewConversation = { viewModel.selectConversation(null) },
                onSelectConversation = viewModel::selectConversation,
                onDraftChanged = viewModel::updateDraft,
                onSend = viewModel::sendDraft,
                onCancel = viewModel::cancelActiveRequest,
                onRename = viewModel::renameActiveConversation,
                onDelete = viewModel::deleteActiveConversation,
                onRenameConversation = viewModel::renameConversation,
                onDeleteConversation = viewModel::deleteConversation,
                onImportQwen = { qwenPicker.launch(arrayOf("application/octet-stream", "application/*", "*/*")) },
                onClearError = viewModel::clearError,
                onShowHome = viewModel::showHome,
                onShowChat = viewModel::showChat,
                onShowConversations = viewModel::showConversations,
                onShowMemories = viewModel::showMemories,
                onRefreshMemories = viewModel::refreshMemories,
                onOpenMemory = viewModel::openMemory,
                onOpenEvidenceConversation = viewModel::openEvidenceConversation,
                onDeleteMemory = viewModel::deleteMemory,
                onPickImage = { imagePicker.launch("image/*") },
                onAddImageToIndex = { imageIndexPicker.launch("image/*") },
                onRemoveImage = viewModel::removeSelectedImage,
                onShowImages = viewModel::showImages,
                onRefreshImages = viewModel::refreshIndexedImages,
                onSearchImages = viewModel::searchIndexedImages,
                onOpenIndexedImage = viewModel::openIndexedImage,
                onAskAboutIndexedImage = viewModel::askAboutIndexedImage,
                onShowSkills = viewModel::showSkills,
                onCapabilityTitleChanged = viewModel::updateCapabilityTitle,
                onCapabilityDescriptionChanged = viewModel::updateCapabilityDescription,
                onAddCapability = viewModel::addCustomCapability,
                onEditCapability = viewModel::editCustomCapability,
                onCancelCapabilityEdit = viewModel::cancelCapabilityEdit,
                onDeleteCapability = viewModel::deleteCustomCapability,
                onEditBuiltInSkill = viewModel::editBuiltInSkill,
                onResetBuiltInSkill = viewModel::resetBuiltInSkill,
                onShowSettings = viewModel::showSettings,
                onDeepSeekApiKeyChanged = viewModel::updateDeepSeekApiKey,
                onSaveDeepSeekApiKey = viewModel::saveDeepSeekApiKey,
                onClearDeepSeekApiKey = viewModel::clearDeepSeekApiKey,
            )
        }
    }
}

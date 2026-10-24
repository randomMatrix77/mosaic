package io.vibeos.intelligence.mnn

import java.io.File

sealed interface ModelPackageValidation {
    data class Valid(val config: File, val graph: File, val tokenizerAssetCount: Int) : ModelPackageValidation
    data class Invalid(val reason: String) : ModelPackageValidation
}

/**
 * Validates the on-disk MNN package shape before JNI sees it. This intentionally validates format,
 * not a Hugging Face repository name: a raw Transformers checkout must never be treated as MNN-ready.
 */
object MnnModelPackageValidator {
    fun validate(directory: File): ModelPackageValidation {
        if (!directory.isDirectory) return ModelPackageValidation.Invalid("Model directory does not exist.")
        val config = File(directory, "config.json")
        if (!config.isFile || config.length() == 0L) {
            return ModelPackageValidation.Invalid("Missing non-empty config.json; this is not an MNN LLM package.")
        }
        val graph = directory.walkTopDown().firstOrNull {
            it.isFile && it.extension.equals("mnn", ignoreCase = true) && it.length() > 0L
        } ?: return ModelPackageValidation.Invalid("No non-empty .mnn graph was found in the model package.")
        val tokenizerAssets = directory.walkTopDown().count {
            it.isFile && (it.name.contains("tokenizer", ignoreCase = true) ||
                it.name.equals("merges.txt", ignoreCase = true) || it.name.equals("vocab.json", ignoreCase = true))
        }
        if (tokenizerAssets == 0) {
            return ModelPackageValidation.Invalid("No tokenizer assets were found next to config.json.")
        }
        return ModelPackageValidation.Valid(config, graph, tokenizerAssets)
    }
}


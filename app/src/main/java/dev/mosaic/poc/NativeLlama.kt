package dev.mosaic.poc

/** Thin Kotlin/JNI boundary. Python reaches it through Chaquopy's java.jclass. */
object NativeLlama {
    @JvmStatic external fun initialize(nativeLibraryDir: String): String
    @JvmStatic external fun loadModel(modelPath: String): String
    @JvmStatic external fun loadEmbeddingModel(modelPath: String): String
    @JvmStatic external fun loadEmbeddingProjector(projectorPath: String): String
    @JvmStatic external fun generate(message: String): String
    @JvmStatic external fun generateChat(serializedMessages: String): String
    @JvmStatic external fun embedText(text: String): FloatArray?
    @JvmStatic external fun embedImage(imagePath: String): FloatArray?
    @JvmStatic external fun unloadModel()
    @JvmStatic external fun isModelLoaded(): Boolean
    @JvmStatic external fun isEmbeddingModelLoaded(): Boolean
    @JvmStatic external fun isEmbeddingProjectorLoaded(): Boolean

    init {
        System.loadLibrary("mosaic_jni")
    }
}

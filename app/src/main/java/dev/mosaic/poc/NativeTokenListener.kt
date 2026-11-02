package dev.mosaic.poc

/**
 * Narrow synchronous callback used only while llama.cpp is generating text.
 *
 * Python creates a Chaquopy dynamic proxy for this interface, so native token
 * pieces travel through the Python orchestration layer before Kotlin observes
 * the corresponding request event.
 */
interface NativeTokenListener {
    fun onDelta(piece: String)
}

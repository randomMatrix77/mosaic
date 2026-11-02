package dev.mosaic.poc

import android.util.Log
import org.json.JSONObject

/**
 * Receives Python lifecycle events and native token deltas for one request.
 *
 * Calling a Python dynamic proxy from inside the synchronous JNI generation
 * call can deadlock Chaquopy's bridge. Native deltas therefore enter Kotlin
 * directly. Python still owns the request, agent loop, tools, and persistence;
 * this class only relays safe presentation events to the Compose activity.
 */
class PythonEventSink(private val handler: (String) -> Unit) : NativeTokenListener {
    private enum class OutputMode { DECIDING, VISIBLE, THINKING, TOOL_CALL }

    private var requestId: String? = null
    private var mode = OutputMode.DECIDING
    private val undecided = StringBuilder()
    private val thinking = StringBuilder()
    private var answerStarted = false
    private var sequence = 0L
    private var terminalForwarded = false

    fun onEvent(serializedEvent: String) {
        synchronized(this) {
            val event = runCatching { JSONObject(serializedEvent) }.getOrNull() ?: return
            if (event.optString("type") == "request_started") {
                requestId = event.optString("request_id").takeIf { it.isNotBlank() }
                sequence = 0L
                terminalForwarded = false
            }
            forward(event)
        }
    }

    override fun onDelta(piece: String) = synchronized(this) {
        if (terminalForwarded) return@synchronized
        // Length only: never write generated user/model content to Logcat.
        Log.d("MosaicStream", "Received native token delta (${piece.length} chars)")
        accept(piece)
    }

    private fun accept(piece: String) {
        when (mode) {
            OutputMode.VISIBLE -> emitAnswer(piece)
            OutputMode.TOOL_CALL -> Unit // Tool JSON/XML is never user-visible.
            OutputMode.THINKING -> {
                thinking.append(piece)
                val closing = thinking.indexOf("</think>")
                if (closing >= 0) {
                    val remainder = thinking.substring(closing + "</think>".length)
                    thinking.clear()
                    mode = OutputMode.DECIDING
                    if (remainder.isNotEmpty()) accept(remainder)
                }
            }
            OutputMode.DECIDING -> {
                undecided.append(piece)
                val candidate = undecided.toString()
                val trimmed = candidate.trimStart()
                when {
                    "<tool_call>".startsWith(trimmed) -> {
                        if (trimmed == "<tool_call>") {
                            undecided.clear()
                            mode = OutputMode.TOOL_CALL
                        }
                    }
                    "<think>".startsWith(trimmed) -> {
                        if (trimmed == "<think>") {
                            undecided.clear()
                            mode = OutputMode.THINKING
                        }
                    }
                    else -> {
                        undecided.clear()
                        mode = OutputMode.VISIBLE
                        emitAnswer(candidate)
                    }
                }
            }
        }
    }

    private fun emitAnswer(piece: String) {
        if (piece.isEmpty()) return
        val id = requestId ?: return
        if (!answerStarted) {
            answerStarted = true
            Log.d("MosaicStream", "Forwarding first visible answer delta")
            forward(JSONObject().put("type", "answer_started").put("request_id", id))
        }
        forward(JSONObject().put("type", "answer_delta").put("request_id", id).put("delta", piece))
    }

    private fun forward(event: JSONObject) {
        val type = event.optString("type")
        val terminal = type == "answer_completed" || type == "request_cancelled" || type == "request_failed"
        if (terminal && terminalForwarded) return
        if (terminal) terminalForwarded = true
        event.put("sequence", ++sequence)
        event.put("timestamp_ms", System.currentTimeMillis())
        handler(event.toString())
    }
}

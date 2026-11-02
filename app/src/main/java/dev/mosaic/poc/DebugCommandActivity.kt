package dev.mosaic.poc

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.util.Log

/** Debug-only long-running ADB entry point without BroadcastReceiver ANR limits. */
class DebugCommandActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        dispatch(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        dispatch(intent)
    }

    private fun dispatch(commandIntent: Intent) {
        val request = runCatching { DebugCommandRunner.decodeRequest(commandIntent) }.getOrElse {
            Log.e("MosaicDebug", "Debug Activity request could not be decoded.", it)
            finishAndRemoveTask()
            return
        }
        Thread {
            DebugCommandRunner.run(applicationContext, request)
            runOnUiThread { finishAndRemoveTask() }
        }.apply { name = "MosaicDebugCommand" }.start()
    }
}

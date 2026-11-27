package com.arta.mobile

internal object NativeChat {
    init {
        System.loadLibrary("arta_android")
    }

    external fun loadModel(path: String)
    external fun hasModel(): Boolean
    external fun reset()
    external fun send(prompt: String, sink: ReplySink): Boolean
}

// The native bridge calls these methods on the same worker thread that called send.
internal interface ReplySink {
    fun onChunk(chunk: String)
    fun isCancelled(): Boolean
}

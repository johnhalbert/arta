package com.arta.mobile

internal object NativeChat {
    init {
        System.loadLibrary("arta_android")
    }

    external fun loadModel(path: String, gpuLayers: Int)
    external fun vulkanDevices(): Array<String>
    external fun setGpuLayers(layers: Int)
    external fun hasModel(): Boolean
    external fun reset()
    external fun restoreTurn(prompt: String, reply: String)
    external fun send(prompt: String, sink: ReplySink): Boolean
}

// The native bridge calls these methods on the same worker thread that called send.
internal interface ReplySink {
    fun onChunk(chunk: String)
    fun isCancelled(): Boolean
}

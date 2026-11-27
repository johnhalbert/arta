package com.arta.mobile

import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.os.Process

/** Hosts the Rust harness in a process that can exit independently of the UI. */
class InferenceService : Service() {
    private val api = object : IInference.Stub() {
        override fun workerPid() = Process.myPid()
        override fun vulkanDevices() = nativeCall { NativeChat.vulkanDevices() }
        override fun loadModel(path: String, gpuLayers: Int) = nativeCall { NativeChat.loadModel(path, gpuLayers) }
        override fun setGpuLayers(layers: Int) = nativeCall { NativeChat.setGpuLayers(layers) }
        override fun hasModel() = nativeCall { NativeChat.hasModel() }
        override fun reset() = nativeCall { NativeChat.reset() }
        override fun restoreTurn(prompt: String, reply: String) = nativeCall { NativeChat.restoreTurn(prompt, reply) }

        override fun send(prompt: String, sink: IReplySink) = nativeCall {
            NativeChat.send(prompt, object : ReplySink {
                override fun onChunk(chunk: String) = sink.onChunk(chunk)
                override fun isCancelled() = sink.isCancelled
            })
        }
    }

    override fun onBind(intent: Intent): IBinder = api

    // JNI reports recoverable failures as RuntimeException. Binder can marshal this error type.
    private inline fun <T> nativeCall(action: () -> T): T = try {
        action()
    } catch (error: RuntimeException) {
        throw IllegalStateException(error.message, error)
    }
}

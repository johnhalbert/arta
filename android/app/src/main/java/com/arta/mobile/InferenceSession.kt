package com.arta.mobile

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.DeadObjectException
import android.os.IBinder
import android.util.Log
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeUnit

/** Called by the chat worker. Keeps enough state to replace a crashed native process. */
internal class InferenceSession(private val context: Context) {
    private var binding: Connection? = null
    private var worker: InferenceWorker? = null
    private var path: String? = null
    private val history = mutableListOf<Turn>()

    val hasModel: Boolean get() = path != null
    var gpuLayers: Int = 0
        private set

    fun vulkanDevices(): List<String> = call { it.vulkanDevices().toList() }

    fun loadModel(path: String, layers: Int) = replace(layers) { service ->
        service.loadModel(path, layers)
        this.path = path
        gpuLayers = layers
        history.clear()
    }

    fun setGpuLayers(layers: Int) = replace(layers) { service ->
        service.setGpuLayers(layers)
        gpuLayers = layers
    }

    fun reset() = call { service ->
        service.reset()
        history.clear()
    }

    fun send(prompt: String, sink: ReplySink): Boolean = call { service ->
        val reply = StringBuffer()
        val completed = service.send(prompt, object : IReplySink.Stub() {
            override fun isCancelled() = sink.isCancelled()

            override fun onChunk(chunk: String) {
                reply.append(chunk)
                sink.onChunk(chunk)
            }
        })
        if (completed) history.add(Turn(prompt, reply.toString()))
        completed
    }

    private fun replace(layers: Int, action: (IInference) -> Unit) = call(gpuLayers != 0 || layers != 0) { service ->
        try {
            action(service)
        } catch (error: IllegalStateException) {
            if (!service.hasModel()) clearModel()
            throw error
        }
    }

    private fun <T> call(usesVulkan: Boolean = gpuLayers != 0, action: (IInference) -> T): T = try {
        action(connect())
    } catch (error: DeadObjectException) {
        val stoppedWorker = worker
        disconnect()
        val available = restoreOnCpu()
        val title = if (usesVulkan) R.string.vulkan_inference_stopped else R.string.inference_stopped
        val recovery = if (available) R.string.inference_recovered else R.string.inference_exited
        val details = stoppedWorker?.exitDetails(context) ?: context.getString(R.string.inference_exit_unavailable)
        throw IllegalStateException("${context.getString(title)} ${context.getString(recovery)}\n$details", error)
    }

    private fun restoreOnCpu(): Boolean {
        val model = path ?: return false
        return try {
            val service = connect()
            service.loadModel(model, 0)
            for ((prompt, reply) in history) service.restoreTurn(prompt, reply)
            gpuLayers = 0
            true
        } catch (error: Exception) {
            Log.e("ArtaInference", "Could not restore the chat on CPU", error)
            disconnect()
            clearModel()
            false
        }
    }

    private fun clearModel() {
        path = null
        gpuLayers = 0
        history.clear()
    }

    private fun connect(): IInference {
        binding?.let { return it.awaitService() }
        val connection = Connection()
        val bound = context.bindService(
            Intent(context, InferenceService::class.java), connection, Context.BIND_AUTO_CREATE,
        )
        binding = connection
        try {
            check(bound) { "Couldn’t start the inference service." }
            val service = connection.awaitService()
            val connectedAt = System.currentTimeMillis()
            worker = InferenceWorker(service.workerPid(), connectedAt)
            return service
        } catch (error: Exception) {
            disconnect()
            throw error
        }
    }

    private fun disconnect() {
        val connection = binding ?: return
        binding = null
        worker = null
        context.unbindService(connection)
    }

    private data class Turn(val prompt: String, val reply: String)

    private class Connection : ServiceConnection {
        private val service = CompletableFuture<IInference?>()

        fun awaitService(): IInference = checkNotNull(service.get(10, TimeUnit.SECONDS)) {
            "Couldn’t connect to the inference service."
        }

        override fun onServiceConnected(name: ComponentName, binder: IBinder) {
            service.complete(IInference.Stub.asInterface(binder))
        }

        // Keep the original binder so the worker observes its death before issuing any new calls.
        override fun onServiceDisconnected(name: ComponentName) = Unit

        override fun onBindingDied(name: ComponentName) {
            service.complete(null)
        }

        override fun onNullBinding(name: ComponentName) {
            service.complete(null)
        }
    }
}

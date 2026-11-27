package com.arta.mobile

import android.app.Application
import android.content.Context
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.provider.OpenableColumns
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class ArtaApplication : Application() {
    internal val chat by lazy { ChatController(this) }
}

internal enum class ActivityState { IDLE, LOADING, IMPORTING, GENERATING, STOPPING, RESETTING }

internal const val ALL_GPU_LAYERS = -1

internal data class Exchange(val prompt: String, val reply: String = "", val note: String? = null)

internal data class ChatScreen(
    val activity: ActivityState = ActivityState.IDLE,
    val modelName: String? = null,
    val exchanges: List<Exchange> = emptyList(),
    val error: String? = null,
    val vulkanDevices: List<String>? = null,
    val gpuLayers: Int = 0,
)

/** Keeps chat state and its connection to the inference process across Activity recreation. */
internal class ChatController(private val context: Context) {
    private val main = Handler(Looper.getMainLooper())
    private val worker = Executors.newSingleThreadExecutor()
    private val inference = InferenceSession(context)
    private val preferences = context.getSharedPreferences("model", Context.MODE_PRIVATE)
    private val directory = File(context.noBackupFilesDir, "models")
    private var modelFile: File? = null
    private var cancellation: AtomicBoolean? = null
    private var observer: ((ChatScreen) -> Unit)? = null
    private var screen = ChatScreen()

    init {
        val name = preferences.getString("file", null)
        val file = name?.let { File(directory, it) }?.takeIf { it.isFile }
        if (file != null) screen = screen.copy(activity = ActivityState.LOADING)
        worker.execute {
            val vulkanDevices = runCatching { inference.vulkanDevices() }.getOrNull()
            val result = file?.let { runCatching { inference.loadModel(it.absolutePath, 0) } }
            main.post {
                if (file != null && result != null) {
                    if (result.isSuccess) modelFile = file
                    update(screen.copy(
                        activity = ActivityState.IDLE,
                        modelName = if (result.isSuccess) preferences.getString("name", file.name) else null,
                        error = result.exceptionOrNull()?.let(::errorMessage),
                        vulkanDevices = vulkanDevices,
                    ))
                } else {
                    update(screen.copy(vulkanDevices = vulkanDevices))
                }
            }
        }
    }

    fun observe(listener: ((ChatScreen) -> Unit)?) {
        observer = listener
        listener?.invoke(screen)
    }

    fun readLastCrashReport(onResult: (String?) -> Unit) {
        worker.execute {
            val report = lastNativeCrashReport(context)
            main.post { onResult(report) }
        }
    }

    fun importModel(uri: Uri) {
        if (screen.activity != ActivityState.IDLE) return
        val previousFile = modelFile
        val gpuLayers = screen.gpuLayers
        update(screen.copy(activity = ActivityState.IMPORTING, error = null))
        worker.execute {
            val result = runCatching {
                val name = displayName(uri)
                check(directory.isDirectory || directory.mkdirs()) { "Couldn’t open model storage." }
                val file = File.createTempFile("model-", ".gguf", directory)
                try {
                    val input = checkNotNull(context.contentResolver.openInputStream(uri)) {
                        "Couldn’t read the selected file."
                    }
                    input.use { source -> file.outputStream().use { source.copyTo(it) } }
                    inference.loadModel(file.absolutePath, gpuLayers)
                } catch (error: Throwable) {
                    file.delete()
                    throw error
                }
                val saved = preferences.edit().putString("file", file.name).putString("name", name).commit()
                if (saved) previousFile?.delete()
                ImportedModel(file, name, saved)
            }
            val previousStillLoaded = result.isFailure && inference.hasModel
            val activeLayers = inference.gpuLayers
            main.post {
                result.fold(
                    onSuccess = { model ->
                        modelFile = model.file
                        update(ChatScreen(
                            modelName = model.name,
                            error = if (model.saved) null else context.getString(R.string.save_failed),
                            vulkanDevices = screen.vulkanDevices,
                            gpuLayers = activeLayers,
                        ))
                    },
                    onFailure = { error ->
                        if (!previousStillLoaded) modelFile = null
                        val previous = if (previousStillLoaded) screen else ChatScreen(vulkanDevices = screen.vulkanDevices)
                        update(previous.copy(activity = ActivityState.IDLE, gpuLayers = activeLayers, error = errorMessage(error)))
                    },
                )
            }
        }
    }

    fun setGpuLayers(layers: Int) {
        if (screen.activity != ActivityState.IDLE || modelFile == null || screen.vulkanDevices.isNullOrEmpty() ||
            layers == screen.gpuLayers
        ) return
        update(screen.copy(activity = ActivityState.LOADING, error = null))
        worker.execute {
            val result = runCatching { inference.setGpuLayers(layers) }
            val modelStillLoaded = inference.hasModel
            val activeLayers = inference.gpuLayers
            main.post {
                if (!modelStillLoaded) modelFile = null
                val next = if (modelStillLoaded) screen else ChatScreen(vulkanDevices = screen.vulkanDevices)
                update(next.copy(
                    activity = ActivityState.IDLE,
                    gpuLayers = activeLayers,
                    error = result.exceptionOrNull()?.let(::errorMessage),
                ))
            }
        }
    }

    fun send(prompt: String) {
        if (screen.activity != ActivityState.IDLE || screen.modelName == null || prompt.isBlank()) return
        val cancelled = AtomicBoolean()
        cancellation = cancelled
        update(screen.copy(
            activity = ActivityState.GENERATING,
            exchanges = screen.exchanges + Exchange(prompt),
            error = null,
        ))
        worker.execute {
            val result = runCatching {
                inference.send(prompt, object : ReplySink {
                    override fun isCancelled() = cancelled.get()

                    override fun onChunk(chunk: String) {
                        main.post { updateLastExchange { it.copy(reply = it.reply + chunk) } }
                    }
                })
            }
            val modelStillLoaded = inference.hasModel
            val activeLayers = inference.gpuLayers
            main.post {
                cancellation = null
                val note = result.fold(
                    onSuccess = { completed -> if (completed) null else context.getString(R.string.stopped) },
                    onFailure = ::errorMessage,
                )
                if (!modelStillLoaded) modelFile = null
                update(screen.copy(
                    activity = ActivityState.IDLE,
                    modelName = screen.modelName.takeIf { modelStillLoaded },
                    exchanges = screen.exchanges.dropLast(1) + screen.exchanges.last().copy(note = note),
                    gpuLayers = activeLayers,
                ))
            }
        }
    }

    fun stop() {
        if (screen.activity != ActivityState.GENERATING) return
        cancellation?.set(true)
        update(screen.copy(activity = ActivityState.STOPPING))
    }

    fun reset() {
        if (screen.activity != ActivityState.IDLE || screen.modelName == null) return
        update(screen.copy(activity = ActivityState.RESETTING, error = null))
        worker.execute {
            val result = runCatching { inference.reset() }
            val modelStillLoaded = inference.hasModel
            val activeLayers = inference.gpuLayers
            main.post {
                if (!modelStillLoaded) modelFile = null
                update(screen.copy(
                    activity = ActivityState.IDLE,
                    modelName = screen.modelName.takeIf { modelStillLoaded },
                    exchanges = if (result.isSuccess) emptyList() else screen.exchanges,
                    error = result.exceptionOrNull()?.let(::errorMessage),
                    gpuLayers = activeLayers,
                ))
            }
        }
    }

    private fun updateLastExchange(transform: (Exchange) -> Exchange) {
        val exchanges = screen.exchanges
        update(screen.copy(exchanges = exchanges.dropLast(1) + transform(exchanges.last())))
    }

    private fun update(state: ChatScreen) {
        screen = state
        observer?.invoke(state)
    }

    private fun displayName(uri: Uri): String {
        return context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { if (it.moveToFirst()) it.getString(0) else null }
            ?.takeIf { it.isNotBlank() } ?: "model.gguf"
    }

    private fun errorMessage(error: Throwable): String =
        error.message?.takeIf { it.isNotBlank() } ?: context.getString(R.string.operation_failed)

    private data class ImportedModel(val file: File, val name: String, val saved: Boolean)
}

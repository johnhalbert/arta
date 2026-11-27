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
    internal lateinit var chat: ChatController
        private set

    override fun onCreate() {
        super.onCreate()
        chat = ChatController(this)
    }
}

internal enum class ActivityState { IDLE, LOADING, IMPORTING, GENERATING, STOPPING, RESETTING }

internal data class Exchange(val prompt: String, val reply: String = "", val note: String? = null)

internal data class ChatScreen(
    val activity: ActivityState = ActivityState.IDLE,
    val modelName: String? = null,
    val exchanges: List<Exchange> = emptyList(),
    val error: String? = null,
)

/** Keeps the chat and its single native worker alive across Activity recreation. */
internal class ChatController(private val context: Context) {
    private val main = Handler(Looper.getMainLooper())
    private val worker = Executors.newSingleThreadExecutor()
    private val preferences = context.getSharedPreferences("model", Context.MODE_PRIVATE)
    private val directory = File(context.noBackupFilesDir, "models")
    private var modelFile: File? = null
    private var cancellation: AtomicBoolean? = null
    private var observer: ((ChatScreen) -> Unit)? = null
    private var screen = ChatScreen()

    init {
        val name = preferences.getString("file", null)
        val file = name?.let { File(directory, it) }?.takeIf { it.isFile }
        if (file != null) {
            screen = screen.copy(activity = ActivityState.LOADING)
            worker.execute {
                val result = runCatching { NativeChat.loadModel(file.absolutePath) }
                main.post {
                    if (result.isSuccess) modelFile = file
                    update(screen.copy(
                        activity = ActivityState.IDLE,
                        modelName = if (result.isSuccess) preferences.getString("name", file.name) else null,
                        error = result.exceptionOrNull()?.let(::errorMessage),
                    ))
                }
            }
        }
    }

    fun observe(listener: ((ChatScreen) -> Unit)?) {
        observer = listener
        listener?.invoke(screen)
    }

    fun importModel(uri: Uri) {
        if (screen.activity != ActivityState.IDLE) return
        val previousFile = modelFile
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
                    NativeChat.loadModel(file.absolutePath)
                } catch (error: Throwable) {
                    file.delete()
                    throw error
                }
                val saved = preferences.edit().putString("file", file.name).putString("name", name).commit()
                if (saved) previousFile?.delete()
                ImportedModel(file, name, saved)
            }
            val previousStillLoaded = result.isFailure && previousFile != null &&
                runCatching { NativeChat.hasModel() }.getOrDefault(false)
            main.post {
                result.fold(
                    onSuccess = { model ->
                        modelFile = model.file
                        update(ChatScreen(
                            modelName = model.name,
                            error = if (model.saved) null else context.getString(R.string.save_failed),
                        ))
                    },
                    onFailure = { error ->
                        if (!previousStillLoaded) modelFile = null
                        val previous = if (previousStillLoaded) screen else ChatScreen()
                        update(previous.copy(activity = ActivityState.IDLE, error = errorMessage(error)))
                    },
                )
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
                NativeChat.send(prompt, object : ReplySink {
                    override fun isCancelled() = cancelled.get()

                    override fun onChunk(chunk: String) {
                        main.post { updateLastExchange { it.copy(reply = it.reply + chunk) } }
                    }
                })
            }
            main.post {
                cancellation = null
                val note = result.fold(
                    onSuccess = { completed -> if (completed) null else context.getString(R.string.stopped) },
                    onFailure = ::errorMessage,
                )
                updateLastExchange(ActivityState.IDLE) { it.copy(note = note) }
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
            val result = runCatching { NativeChat.reset() }
            main.post {
                update(screen.copy(
                    activity = ActivityState.IDLE,
                    exchanges = if (result.isSuccess) emptyList() else screen.exchanges,
                    error = result.exceptionOrNull()?.let(::errorMessage),
                ))
            }
        }
    }

    private fun updateLastExchange(
        activity: ActivityState = screen.activity,
        transform: (Exchange) -> Exchange,
    ) {
        val exchanges = screen.exchanges
        update(screen.copy(activity = activity, exchanges = exchanges.dropLast(1) + transform(exchanges.last())))
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

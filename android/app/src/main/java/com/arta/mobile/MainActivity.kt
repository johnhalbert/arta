package com.arta.mobile

import android.app.Activity
import android.app.AlertDialog
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.graphics.Typeface
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.text.Editable
import android.text.InputType
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.TextWatcher
import android.text.method.LinkMovementMethod
import android.text.style.StyleSpan
import android.view.Gravity
import android.view.View
import android.view.WindowInsets
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import io.noties.markwon.AbstractMarkwonPlugin
import io.noties.markwon.LinkResolverDef
import io.noties.markwon.Markwon
import io.noties.markwon.MarkwonConfiguration
import io.noties.markwon.ext.strikethrough.StrikethroughPlugin
import io.noties.markwon.ext.tables.TablePlugin

class MainActivity : Activity() {
    private val chat get() = (application as ArtaApplication).chat
    private lateinit var status: TextView
    private lateinit var transcript: TextView
    private lateinit var scroll: ScrollView
    private lateinit var draft: EditText
    private lateinit var send: Button
    private lateinit var importModel: Button
    private lateinit var reset: Button
    private lateinit var gpuOffload: Button
    private var screen = ChatScreen()
    private var formattedReplies = emptyList<Spanned>()
    private val markdown by lazy {
        Markwon.builder(this)
            .usePlugin(StrikethroughPlugin.create())
            .usePlugin(TablePlugin.create(this))
            .usePlugin(object : AbstractMarkwonPlugin() {
                override fun configureConfiguration(builder: MarkwonConfiguration.Builder) {
                    val resolver = LinkResolverDef()
                    builder.linkResolver { view, link ->
                        val uri = Uri.parse(link)
                        if ((uri.scheme.equals("https", ignoreCase = true) ||
                                uri.scheme.equals("http", ignoreCase = true)) && !uri.host.isNullOrBlank()
                        ) {
                            resolver.resolve(view, link)
                        }
                    }
                }
            })
            .build()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(8), dp(16), dp(8))
        }
        if (Build.VERSION.SDK_INT >= 30) {
            if (Build.VERSION.SDK_INT < 35) {
                @Suppress("DEPRECATION")
                window.setDecorFitsSystemWindows(false)
            }
            root.setOnApplyWindowInsetsListener { view, insets ->
                val edges = insets.getInsets(
                    WindowInsets.Type.systemBars() or WindowInsets.Type.displayCutout() or WindowInsets.Type.ime(),
                )
                view.setPadding(edges.left + dp(16), edges.top + dp(8), edges.right + dp(16), edges.bottom + dp(8))
                insets
            }
        }

        val heading = TextView(this).apply {
            setText(R.string.app_name)
            textSize = 25f
            setTypeface(typeface, Typeface.BOLD)
            isAccessibilityHeading = true
        }
        importModel = Button(this).apply {
            setText(R.string.import_model)
            setOnClickListener { chooseModel() }
        }
        reset = Button(this).apply {
            setText(R.string.new_chat)
            setOnClickListener { chat.reset() }
        }
        val toolbar = LinearLayout(this).apply {
            gravity = Gravity.CENTER_VERTICAL
            addView(heading, LinearLayout.LayoutParams(0, -2, 1f))
            addView(importModel)
            addView(reset)
        }
        gpuOffload = Button(this).apply {
            isAllCaps = false
            setOnClickListener { chooseGpuLayers() }
        }
        status = TextView(this).apply {
            accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_POLITE
            setTextIsSelectable(true)
        }
        transcript = TextView(this).apply {
            textSize = 16f
            setTextIsSelectable(true)
            movementMethod = LinkMovementMethod.getInstance()
            setPadding(0, dp(16), 0, dp(16))
        }
        scroll = ScrollView(this).apply { addView(transcript) }

        draft = EditText(this).apply {
            id = R.id.draft
            setHint(R.string.prompt_hint)
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE or
                InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
            maxLines = 5
            addTextChangedListener(object : TextWatcher {
                override fun beforeTextChanged(text: CharSequence?, start: Int, count: Int, after: Int) = Unit
                override fun onTextChanged(text: CharSequence?, start: Int, before: Int, count: Int) = updateControls()
                override fun afterTextChanged(text: Editable?) = Unit
            })
        }
        send = Button(this).apply {
            setText(R.string.send)
            setOnClickListener {
                if (screen.activity == ActivityState.GENERATING) {
                    chat.stop()
                } else if (screen.modelName != null && screen.activity == ActivityState.IDLE) {
                    val prompt = draft.text.toString()
                    if (prompt.isNotBlank()) {
                        chat.send(prompt)
                        draft.text.clear()
                    }
                }
            }
        }
        val composer = LinearLayout(this).apply {
            gravity = Gravity.BOTTOM
            addView(draft, LinearLayout.LayoutParams(0, -2, 1f))
            addView(send)
        }
        root.addView(toolbar)
        root.addView(gpuOffload, LinearLayout.LayoutParams(-2, -2))
        root.addView(status)
        root.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))
        root.addView(composer)
        setContentView(root)
    }

    override fun onStart() {
        super.onStart()
        chat.observe(::render)
    }

    override fun onStop() {
        chat.observe(null)
        super.onStop()
    }

    @Suppress("DEPRECATION")
    private fun chooseModel() {
        startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "*/*"
        }, PICK_MODEL)
    }

    @Deprecated("Uses the framework document picker result callback.")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        @Suppress("DEPRECATION")
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == PICK_MODEL && resultCode == RESULT_OK) data?.data?.let(chat::importModel)
    }

    private fun chooseGpuLayers() {
        val devices = screen.vulkanDevices
        val gpuInfo = when {
            devices == null -> getString(R.string.gpu_query_unavailable)
            devices.isEmpty() -> getString(R.string.gpu_none_detected)
            else -> getString(R.string.gpu_detected, devices.joinToString("\n"))
        }
        val deviceInfo = "$gpuInfo\n\n${getString(R.string.npu_offload_unavailable)}"
        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(24), dp(12), dp(24), 0)
            addView(TextView(this@MainActivity).apply { text = deviceInfo })
            addView(Button(this@MainActivity).apply {
                isAllCaps = false
                setText(R.string.inference_last_crash)
                setOnClickListener { showLastCrashReport() }
            })
        }
        val builder = AlertDialog.Builder(this).setTitle(R.string.gpu_offload).setView(content)
        if (screen.modelName == null || devices.isNullOrEmpty()) {
            if (!devices.isNullOrEmpty()) {
                content.addView(TextView(this).apply { setText(R.string.gpu_model_required) })
            }
            builder.setPositiveButton(android.R.string.ok, null).show()
            return
        }

        val layers = EditText(this).apply {
            setSingleLine()
            inputType = InputType.TYPE_CLASS_NUMBER
            setHint(R.string.gpu_layer_count)
            setText(screen.gpuLayers.takeIf { it >= 0 }?.toString())
            selectAll()
        }
        content.addView(TextView(this).apply { setText(R.string.gpu_layer_help) })
        content.addView(layers)
        val dialog = builder
            .setNegativeButton(android.R.string.cancel, null)
            .setNeutralButton(R.string.gpu_all_layers) { _, _ -> chat.setGpuLayers(ALL_GPU_LAYERS) }
            .setPositiveButton(android.R.string.ok, null)
            .show()
        dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
            val count = layers.text.toString().trim().toIntOrNull()
            if (count == null || count < 0) {
                layers.error = getString(R.string.gpu_invalid_count)
            } else {
                chat.setGpuLayers(count)
                dialog.dismiss()
            }
        }
    }

    private fun showLastCrashReport() {
        chat.readLastCrashReport { report ->
            if (isFinishing || isDestroyed) return@readLastCrashReport
            val title = getString(R.string.inference_last_crash)
            val builder = AlertDialog.Builder(this).setTitle(title)
                .setPositiveButton(android.R.string.ok, null)
            if (report == null) {
                builder.setMessage(R.string.inference_no_saved_crash).show()
                return@readLastCrashReport
            }
            val devices = screen.vulkanDevices.orEmpty()
            val text = if (devices.isEmpty()) report else {
                "$report\n\n${getString(R.string.gpu_detected, devices.joinToString("\n"))}"
            }
            val details = ScrollView(this).apply {
                setPadding(dp(24), dp(12), dp(24), dp(12))
                addView(TextView(this@MainActivity).apply {
                    this.text = text
                    setTextIsSelectable(true)
                })
            }
            builder.setView(details)
                .setNeutralButton(R.string.inference_copy_crash) { _, _ ->
                    getSystemService(ClipboardManager::class.java)
                        ?.setPrimaryClip(ClipData.newPlainText(title, text))
                }
                .show()
        }
    }

    private fun render(state: ChatScreen) {
        val previousExchanges = screen.exchanges
        screen = state
        status.text = when (state.activity) {
            ActivityState.IMPORTING -> getString(R.string.importing_model)
            ActivityState.LOADING -> getString(R.string.loading_model)
            ActivityState.GENERATING -> getString(R.string.generating)
            ActivityState.STOPPING -> getString(R.string.stopping)
            else -> state.error ?: state.modelName ?: getString(R.string.model_required)
        }
        updateControls()
        if (previousExchanges != state.exchanges) {
            val follow = !scroll.canScrollVertically(1)
            formattedReplies = state.exchanges.mapIndexed { index, exchange ->
                if (previousExchanges.getOrNull(index)?.reply == exchange.reply) {
                    formattedReplies[index]
                } else {
                    markdown.toMarkdown(exchange.reply)
                }
            }
            val text = SpannableStringBuilder().apply {
                state.exchanges.forEachIndexed { index, exchange ->
                    if (isNotEmpty()) append("\n\n")
                    appendSpeaker(getString(R.string.you))
                    append(exchange.prompt)
                    append("\n\n")
                    appendSpeaker(getString(R.string.app_name))
                    append(formattedReplies[index])
                    exchange.note?.let { append("\n").append(it) }
                }
            }
            markdown.setParsedMarkdown(transcript, text)
            if (follow) scroll.post { scroll.smoothScrollTo(0, transcript.height) }
        }
    }

    private fun SpannableStringBuilder.appendSpeaker(name: String) {
        val start = length
        append(name)
        setSpan(StyleSpan(Typeface.BOLD), start, length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        append('\n')
    }

    private fun updateControls() {
        val idle = screen.activity == ActivityState.IDLE
        val ready = idle && screen.modelName != null
        val generating = screen.activity == ActivityState.GENERATING
        importModel.isEnabled = idle
        reset.isEnabled = ready && screen.exchanges.isNotEmpty()
        gpuOffload.isEnabled = idle
        gpuOffload.text = when {
            screen.vulkanDevices.isNullOrEmpty() -> getString(R.string.gpu_unavailable)
            screen.modelName == null -> getString(R.string.gpu_offload)
            screen.gpuLayers == 0 -> getString(R.string.gpu_cpu)
            screen.gpuLayers == ALL_GPU_LAYERS -> getString(R.string.gpu_all)
            else -> resources.getQuantityString(R.plurals.gpu_layers, screen.gpuLayers, screen.gpuLayers)
        }
        draft.isEnabled = ready
        send.setText(if (generating || screen.activity == ActivityState.STOPPING) R.string.stop else R.string.send)
        send.isEnabled = generating || (ready && draft.text.isNotBlank())
    }

    private fun dp(value: Int) = (value * resources.displayMetrics.density + 0.5f).toInt()

    private companion object {
        const val PICK_MODEL = 100
    }
}

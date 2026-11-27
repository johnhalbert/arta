package com.arta.mobile

import android.app.Activity
import android.content.Intent
import android.graphics.Typeface
import android.os.Build
import android.os.Bundle
import android.text.Editable
import android.text.InputType
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.TextWatcher
import android.text.style.StyleSpan
import android.view.Gravity
import android.view.View
import android.view.WindowInsets
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

class MainActivity : Activity() {
    private val chat get() = (application as ArtaApplication).chat
    private lateinit var status: TextView
    private lateinit var transcript: TextView
    private lateinit var scroll: ScrollView
    private lateinit var draft: EditText
    private lateinit var send: Button
    private lateinit var importModel: Button
    private lateinit var reset: Button
    private var screen = ChatScreen()

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
        status = TextView(this).apply { accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_POLITE }
        transcript = TextView(this).apply {
            textSize = 16f
            setTextIsSelectable(true)
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

    private fun render(state: ChatScreen) {
        val changed = screen.exchanges != state.exchanges
        screen = state
        status.text = when (state.activity) {
            ActivityState.IMPORTING -> getString(R.string.importing_model)
            ActivityState.LOADING -> getString(R.string.loading_model)
            ActivityState.GENERATING -> getString(R.string.generating)
            ActivityState.STOPPING -> getString(R.string.stopping)
            else -> state.error ?: state.modelName ?: getString(R.string.model_required)
        }
        updateControls()
        if (changed) {
            val follow = !scroll.canScrollVertically(1)
            transcript.text = SpannableStringBuilder().apply {
                state.exchanges.forEach { exchange ->
                    if (isNotEmpty()) append("\n\n")
                    appendSpeaker(getString(R.string.you))
                    append(exchange.prompt)
                    append("\n\n")
                    appendSpeaker(getString(R.string.app_name))
                    append(exchange.reply)
                    exchange.note?.let { append("\n").append(it) }
                }
            }
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
        draft.isEnabled = ready
        send.setText(if (generating || screen.activity == ActivityState.STOPPING) R.string.stop else R.string.send)
        send.isEnabled = generating || (ready && draft.text.isNotBlank())
    }

    private fun dp(value: Int) = (value * resources.displayMetrics.density + 0.5f).toInt()

    private companion object {
        const val PICK_MODEL = 100
    }
}

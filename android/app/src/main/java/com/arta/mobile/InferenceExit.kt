package com.arta.mobile

import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.content.Context
import android.os.Build
import android.os.SystemClock
import android.system.OsConstants
import com.google.protobuf.CodedInputStream
import java.io.IOException
import java.io.InputStream
import java.text.DateFormat
import java.util.Date

internal data class InferenceWorker(val pid: Int, val connectedAt: Long) {
    fun exitDetails(context: Context): String? {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) return null
        val manager = context.getSystemService(ActivityManager::class.java) ?: return null
        // Binder can report the death before Android finishes recording the exit.
        repeat(5) { attempt ->
            val exit = runCatching {
                manager.getHistoricalProcessExitReasons(context.packageName, pid, 5).firstOrNull {
                    it.isInferenceProcess(context) && it.timestamp >= connectedAt
                }
            }.getOrElse { return null }
            if (exit != null) return describeExit(context, exit)
            if (attempt < 4) SystemClock.sleep(100)
        }
        return null
    }
}

internal fun lastNativeCrashReport(context: Context): String? {
    if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) return null
    val manager = context.getSystemService(ActivityManager::class.java) ?: return null
    return runCatching {
        val exit = manager.getHistoricalProcessExitReasons(context.packageName, 0, 0).firstOrNull {
            it.isInferenceProcess(context) && it.reason == ApplicationExitInfo.REASON_CRASH_NATIVE
        } ?: return null
        val date = DateFormat.getDateTimeInstance().format(Date(exit.timestamp))
        "$date\n${describeExit(context, exit, includeStack = true)}"
    }.getOrNull()
}

@android.annotation.TargetApi(Build.VERSION_CODES.R)
private fun ApplicationExitInfo.isInferenceProcess(context: Context): Boolean =
    processName == "${context.packageName}:inference"

@android.annotation.TargetApi(Build.VERSION_CODES.R)
private fun describeExit(context: Context, exit: ApplicationExitInfo, includeStack: Boolean = false): String {
    val tombstone = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S &&
        exit.reason == ApplicationExitInfo.REASON_CRASH_NATIVE
    ) {
        runCatching { exit.traceInputStream?.use { readTombstone(it, includeStack) } }.getOrNull()
    } else {
        null
    }
    val signalNumber = tombstone?.signal?.number?.takeIf { it > 0 } ?: exit.status
    val summary = when (exit.reason) {
        ApplicationExitInfo.REASON_CRASH_NATIVE -> if (signalNumber > 0) {
            context.getString(R.string.inference_native_crash_with_signal, signalDetails(context, signalNumber, tombstone?.signal))
        } else {
            context.getString(R.string.inference_native_crash)
        }
        ApplicationExitInfo.REASON_SIGNALED ->
            context.getString(R.string.inference_signal_exit, signalDetails(context, signalNumber))
        ApplicationExitInfo.REASON_LOW_MEMORY -> context.getString(R.string.inference_low_memory)
        ApplicationExitInfo.REASON_CRASH -> context.getString(R.string.inference_java_crash)
        ApplicationExitInfo.REASON_EXIT_SELF -> context.getString(R.string.inference_exit_code, exit.status)
        else -> context.getString(R.string.inference_exit_other, exit.reason, exit.status)
    }
    val detail = tombstone?.abortMessage?.trim()?.takeIf { it.isNotEmpty() }?.let {
        context.getString(R.string.inference_abort_message, it.take(500))
    } ?: exit.description?.trim()?.takeIf { it.isNotEmpty() && !it.equals("crash", ignoreCase = true) }?.take(500)
    val stack = tombstone?.backtrace.orEmpty().takeIf { it.isNotEmpty() }?.let {
        context.getString(R.string.inference_native_stack, it.joinToString("\n"))
    }
    return listOfNotNull(summary, detail, stack).joinToString("\n")
}

private fun signalDetails(context: Context, number: Int, signal: CrashSignal? = null): String {
    val name = signal?.name?.takeIf { it.isNotBlank() } ?: when (number) {
        OsConstants.SIGABRT -> "SIGABRT"
        OsConstants.SIGBUS -> "SIGBUS"
        OsConstants.SIGILL -> "SIGILL"
        OsConstants.SIGKILL -> "SIGKILL"
        OsConstants.SIGSEGV -> "SIGSEGV"
        else -> null
    }
    return listOfNotNull(
        name,
        context.getString(R.string.inference_signal_number, number),
        signal?.code?.takeIf { it.isNotBlank() },
    ).joinToString(", ")
}

private data class NativeTombstone(val signal: CrashSignal?, val abortMessage: String?, val backtrace: List<String>)
private data class CrashSignal(val number: Int, val name: String?, val code: String?)

private const val MAX_STACK_FRAMES = 12

// Fields follow Android's debuggerd/proto/tombstone.proto. Skip memory dumps and other unrelated data.
private fun readTombstone(stream: InputStream, includeStack: Boolean): NativeTombstone {
    val input = CodedInputStream.newInstance(stream)
    input.setSizeLimit(4 * 1024 * 1024)
    var crashedTid = 0
    var signal: CrashSignal? = null
    var abortMessage: String? = null
    val threads = mutableMapOf<Int, List<String>>()
    try {
        while (includeStack || signal == null || abortMessage == null) {
            when (val tag = input.readTag()) {
                0 -> break
                6 * 8 -> crashedTid = input.readUInt32()
                10 * 8 + 2 -> signal = readSignal(input.readMessage())
                14 * 8 + 2 -> abortMessage = input.readString()
                16 * 8 + 2 -> if (includeStack) {
                    val (tid, frames) = readThreadEntry(input.readMessage())
                    threads[tid] = frames
                } else if (!input.skipField(tag)) break
                else -> if (!input.skipField(tag)) break
            }
        }
    } catch (_: IOException) {
        // A truncated or oversized record can still contain the signal and crashing thread.
    }
    return NativeTombstone(signal, abortMessage, threads[crashedTid].orEmpty())
}

private fun CodedInputStream.readMessage(): CodedInputStream = CodedInputStream.newInstance(readByteArray())

private fun readSignal(input: CodedInputStream): CrashSignal {
    var number = 0
    var name: String? = null
    var code: String? = null
    while (true) {
        when (val tag = input.readTag()) {
            0 -> break
            1 * 8 -> number = input.readInt32()
            2 * 8 + 2 -> name = input.readString()
            4 * 8 + 2 -> code = input.readString()
            else -> if (!input.skipField(tag)) break
        }
    }
    return CrashSignal(number, name, code)
}

private fun readThreadEntry(input: CodedInputStream): Pair<Int, List<String>> {
    var tid = 0
    var frames = emptyList<String>()
    while (true) {
        when (val tag = input.readTag()) {
            0 -> break
            1 * 8 -> tid = input.readUInt32()
            2 * 8 + 2 -> frames = readBacktrace(input.readMessage())
            else -> if (!input.skipField(tag)) break
        }
    }
    return tid to frames
}

private fun readBacktrace(input: CodedInputStream): List<String> = buildList {
    while (size < MAX_STACK_FRAMES) {
        when (val tag = input.readTag()) {
            0 -> break
            4 * 8 + 2 -> add("#${size.toString().padStart(2, '0')} ${readFrame(input.readMessage())}")
            else -> if (!input.skipField(tag)) break
        }
    }
}

private fun readFrame(input: CodedInputStream): String {
    var pc = 0L
    var function = ""
    var offset = 0L
    var library = ""
    var buildId = ""
    while (true) {
        when (val tag = input.readTag()) {
            0 -> break
            1 * 8 -> pc = input.readUInt64()
            4 * 8 + 2 -> function = input.readString()
            5 * 8 -> offset = input.readUInt64()
            6 * 8 + 2 -> library = input.readString().substringAfterLast('/')
            8 * 8 + 2 -> buildId = input.readString()
            else -> if (!input.skipField(tag)) break
        }
    }
    return buildString {
        append("pc ${pc.toULong().toString(16).padStart(8, '0')} ${library.ifEmpty { "?" }}")
        if (function.isNotEmpty()) append(" (${function.take(180)}+$offset)")
        if (buildId.isNotEmpty()) append(" (BuildId: ${buildId.take(80)})")
    }
}

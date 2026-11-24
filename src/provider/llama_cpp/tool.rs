//! The llama.cpp implementation of structured inference. It renders schemas and call history
//! through the chat template, buffers an entire generation on a blocking worker, then parses
//! it into a reply or callable batch.

use super::{
    LlamaCPP, Sampling, generate, logging, model::chat_messages_with, text::text_chunks_with_stop,
};
use crate::provider::{
    PromptObserver,
    tool::{ToolAdapter, ToolCall, ToolExchange, ToolRequest, ToolResponse},
};
use anyhow::{Context, Result, ensure};
use futures_util::future::BoxFuture;
use llama_cpp_4::prelude::AddBos;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{cell::Cell, sync::Arc};
use tokio::{runtime::Handle, sync::oneshot};

impl ToolAdapter for LlamaCPP {
    fn infer<'a>(&'a self, request: ToolRequest<'a>) -> BoxFuture<'a, Result<ToolResponse>> {
        self.infer_observed(request, None)
    }

    fn infer_observed<'a>(
        &'a self,
        request: ToolRequest<'a>,
        observer: Option<PromptObserver>,
    ) -> BoxFuture<'a, Result<ToolResponse>> {
        self.infer_recorded(request, &[], observer)
    }

    /// Replay exchanges by turn, capture the rendered prompt, and generate with template
    /// grammar and raw token bytes. Dropping the awaiting future closes a cancellation
    /// channel checked by the worker.
    fn infer_recorded<'a>(
        &'a self,
        request: ToolRequest<'a>,
        past_exchanges: &'a [Vec<ToolExchange>],
        observer: Option<PromptObserver>,
    ) -> BoxFuture<'a, Result<ToolResponse>> {
        Box::pin(async move {
            ensure!(
                !request.prompt.contains('\0'),
                "prompt contains a NUL character"
            );
            ensure!(
                !request
                    .system_prompt
                    .is_some_and(|prompt| prompt.contains('\0')),
                "system prompt contains a NUL character"
            );
            let model = self.load_model(request.model_id, None)?;
            let (messages, tools) = tool_prompt(&request, past_exchanges);
            let inference = Arc::clone(&self.inference);
            let verbose = self.verbose;
            let previous_calls = past_exchanges
                .iter()
                .flatten()
                .chain(request.exchanges)
                .map(|exchange| exchange.calls.len())
                .sum();
            let runtime =
                Handle::try_current().context("llama.cpp inference requires a Tokio runtime")?;
            let (worker_cancel, cancel_on_drop) = oneshot::channel::<()>();
            let result = runtime
                .spawn_blocking(move || {
                    let _logs = logging::Scope::new(verbose);
                    let cancelled = || worker_cancel.is_closed();
                    ensure!(!cancelled(), "llama.cpp inference cancelled");
                    let params = model.chat.params(messages, Some(tools))?;
                    let chat = model
                        .tool_template()?
                        .apply(&params)
                        .context("failed to format tool prompt")?;
                    if let Some(observe) = &observer {
                        observe(&chat.prompt);
                    }
                    let add_bos = if model.inner.get_vocab().get_add_bos() {
                        AddBos::Always
                    } else {
                        AddBos::Never
                    };
                    let tokens = chat
                        .tokenize_prompt(&model.inner, add_bos)
                        .context("failed to tokenize tool prompt")?;
                    let mut stops: Vec<String> = serde_json::from_str(&chat.additional_stops_json)
                        .context("failed to read chat template stop strings")?;
                    stops.extend(model.chat.stop.iter().cloned());
                    let exhausted = Cell::new(false);
                    let pieces = generate(
                        &model,
                        tokens,
                        &inference,
                        Sampling::Tools {
                            chat: &chat,
                            exhausted: &exhausted,
                        },
                        cancelled,
                    )?;
                    let text: String =
                        text_chunks_with_stop(pieces, stops, || exhausted.set(false))
                            .collect::<Result<_>>()?;
                    ensure!(!cancelled(), "llama.cpp inference cancelled");
                    let parsed = chat
                        .parse(&text, false)
                        .context("failed to parse model output")?;
                    parse_response(&parsed, previous_calls, exhausted.get())
                })
                .await;
            drop(cancel_on_drop);
            result.context("llama.cpp tool inference worker failed")?
        })
    }
}

/// Serialize role messages and function schemas separately. Replay prior exchanges at their
/// original turns and append current exchanges after this user prompt.
fn tool_prompt(
    request: &ToolRequest<'_>,
    past_exchanges: &[Vec<ToolExchange>],
) -> (String, String) {
    let mut messages = chat_messages_with(
        request.system_prompt,
        request.history,
        request.prompt,
        |index, messages| {
            if let Some(exchanges) = past_exchanges.get(index) {
                append_exchanges(messages, exchanges);
            }
        },
    );
    append_exchanges(&mut messages, request.exchanges);
    let tools: Vec<_> = request
        .tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                }
            })
        })
        .collect();
    (
        Value::Array(messages).to_string(),
        Value::Array(tools).to_string(),
    )
}

/// Write each assistant call batch followed by tool-role results referencing the original
/// call IDs. Prefix recoverable tool errors in the text sent to the model.
fn append_exchanges(messages: &mut Vec<Value>, exchanges: &[ToolExchange]) {
    for exchange in exchanges {
        let calls: Vec<_> = exchange
            .calls
            .iter()
            .map(|(call, _)| {
                json!({
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": Value::Object(call.arguments.clone()).to_string()
                    }
                })
            })
            .collect();
        let mut assistant = json!({
            "role": "assistant",
            "content": exchange.content,
            "tool_calls": calls,
        });
        if let Some(reasoning) = &exchange.reasoning {
            assistant["reasoning_content"] = json!(reasoning);
        }
        messages.push(assistant);
        for (call, output) in &exchange.calls {
            let content = match &output.content {
                Value::String(text) => text.clone(),
                value => value.to_string(),
            };
            let content = if output.is_error {
                format!("Tool error: {content}")
            } else {
                content
            };
            messages.push(json!({
                "role": "tool",
                "name": call.name,
                "tool_call_id": call.id,
                "content": content
            }));
        }
    }
}

/// Shape returned by the native chat parser. Missing calls default to an ordinary text reply.
#[derive(Deserialize)]
struct AssistantResponse {
    /// Final assistant text or text accompanying a batch of calls.
    content: Option<String>,
    /// Optional template-parsed reasoning retained when replaying a call batch.
    reasoning_content: Option<String>,
    /// Parsed calls in model order. Absence is deserialized as an empty list.
    #[serde(default)]
    tool_calls: Vec<AssistantCall>,
}

/// Native parser's function invocation and optional template-supplied correlation ID.
#[derive(Deserialize)]
struct AssistantCall {
    /// Optional correlation ID. The adapter synthesizes one when absent or empty.
    id: Option<String>,
    /// The function descriptor to convert into an executable Arta call.
    function: AssistantFunction,
}

/// Function name and JSON-encoded argument string as emitted by the parser.
#[derive(Deserialize)]
struct AssistantFunction {
    /// Name matched against the turn's visible tool registry.
    name: String,
    /// Serialized JSON decoded into a string-keyed object before any tool executes.
    arguments: String,
}

/// Parse arguments as objects, synthesize missing IDs from the cumulative call count, and
/// reject a call batch cut off at the token limit. A text-only result may use the full
/// budget.
fn parse_response(parsed: &str, previous_calls: usize, exhausted: bool) -> Result<ToolResponse> {
    let response: AssistantResponse =
        serde_json::from_str(parsed).context("invalid llama.cpp assistant message")?;
    if response.tool_calls.is_empty() {
        return Ok(ToolResponse::Reply(response.content.unwrap_or_default()));
    }
    ensure!(
        !exhausted,
        "generation reached its token limit while producing a tool call"
    );
    let calls = response
        .tool_calls
        .into_iter()
        .enumerate()
        .map(|(index, call)| {
            let arguments = serde_json::from_str(&call.function.arguments)
                .with_context(|| format!("invalid arguments for tool '{}'", call.function.name))?;
            Ok(ToolCall {
                id: call
                    .id
                    .filter(|id| !id.is_empty())
                    .unwrap_or_else(|| format!("{:09}", previous_calls + index + 1)),
                name: call.function.name,
                arguments,
            })
        })
        .collect::<Result<_>>()?;
    Ok(ToolResponse::Calls {
        content: response.content,
        reasoning: response.reasoning_content,
        calls,
    })
}

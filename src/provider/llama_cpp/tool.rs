use super::{LlamaCPP, Sampling, generate, model::chat_messages, text::text_chunks_with_stop};
use crate::provider::tool::{ToolAdapter, ToolCall, ToolRequest, ToolResponse};
use anyhow::{Context, Result, ensure};
use futures_util::future::BoxFuture;
use llama_cpp_4::prelude::{AddBos, ChatApplyParams};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{cell::Cell, sync::Arc};
use tokio::{runtime::Handle, sync::oneshot};

impl ToolAdapter for LlamaCPP {
    fn infer<'a>(&'a self, request: ToolRequest<'a>) -> BoxFuture<'a, Result<ToolResponse>> {
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
            let model = Arc::clone(self.loaded_model(request.model_id)?);
            let params = tool_prompt(&request);
            let context_length = self.context_length;
            let max_tokens = self.max_tokens;
            let previous_calls: usize = request.exchanges.iter().map(|turn| turn.calls.len()).sum();
            let runtime =
                Handle::try_current().context("llama.cpp inference requires a Tokio runtime")?;
            let (worker_cancel, cancel_on_drop) = oneshot::channel::<()>();
            let result = runtime
                .spawn_blocking(move || {
                    let cancelled = || worker_cancel.is_closed();
                    ensure!(!cancelled(), "llama.cpp inference cancelled");
                    let chat = model
                        .tool_template()?
                        .apply(&params)
                        .context("failed to format tool prompt")?;
                    let add_bos = if model.inner.get_vocab().get_add_bos() {
                        AddBos::Always
                    } else {
                        AddBos::Never
                    };
                    let tokens = chat
                        .tokenize_prompt(&model.inner, add_bos)
                        .context("failed to tokenize tool prompt")?;
                    let stops = serde_json::from_str(&chat.additional_stops_json)
                        .context("failed to read chat template stop strings")?;
                    let exhausted = Cell::new(false);
                    let pieces = generate(
                        &model,
                        tokens,
                        context_length,
                        max_tokens,
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

fn tool_prompt(request: &ToolRequest<'_>) -> ChatApplyParams {
    let mut messages = chat_messages(request.system_prompt, request.history, request.prompt);
    for exchange in request.exchanges {
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
    ChatApplyParams::new(Value::Array(messages).to_string())
        .with_tools(Value::Array(tools).to_string())
        .with_enable_thinking(false)
}

#[derive(Deserialize)]
struct AssistantResponse {
    content: Option<String>,
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<AssistantCall>,
}

#[derive(Deserialize)]
struct AssistantCall {
    id: Option<String>,
    function: AssistantFunction,
}

#[derive(Deserialize)]
struct AssistantFunction {
    name: String,
    arguments: String,
}

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

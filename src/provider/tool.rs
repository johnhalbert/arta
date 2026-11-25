//! Orchestrates structured inference for providers implementing [`ToolAdapter`]. The adapter
//! renders and parses one step. This module enforces a per-prompt budget, dispatches calls,
//! and decides whether another inference step is needed.

use super::{PromptObserver, Provider, Turn, TurnObservers};
use crate::{
    stream::Message,
    tool::{ToolArguments, ToolOutput, ToolSpec, Tools},
};
use anyhow::{Context, Result, ensure};
use futures_util::{future::BoxFuture, stream};

pub(crate) const MAX_CALLS_PER_PROMPT: u32 = 8;

/// The current inference turn, including tool exchanges that have already completed.
pub struct ToolRequest<'a> {
    /// Selection within the adapter's provider.
    pub model_id: &'a str,
    /// Combined profile and caller-supplied instructions for this turn.
    pub system_prompt: Option<&'a str>,
    /// Completed prompt/reply pairs in chronological order.
    pub history: &'a [Turn],
    /// New user input, repeated on every inference step for this turn.
    pub prompt: &'a str,
    /// Schemas visible to the model for this turn.
    pub tools: &'a [&'a ToolSpec],
    /// Completed call batches for the current prompt, in execution order.
    pub exchanges: &'a [ToolExchange],
}

/// One parsed invocation whose ID links tool-role messages to the assistant's request.
pub struct ToolCall {
    /// Correlation ID from the provider or its adapter.
    pub id: String,
    /// Exact key used to look up a visible tool.
    pub name: String,
    /// Parsed argument object delivered to the implementation.
    pub arguments: ToolArguments,
}

/// One assistant call message paired with the outputs of the calls that actually ran.
pub struct ToolExchange {
    /// Optional assistant text accompanying the batch.
    pub content: Option<String>,
    /// Optional reasoning content preserved for replay by compatible templates.
    pub reasoning: Option<String>,
    /// Invocations and results paired in execution order.
    pub calls: Vec<(ToolCall, ToolOutput)>,
}

/// Outcome of a single inference step.
pub enum ToolResponse {
    /// Text completing this turn.
    Reply(String),
    /// Request tool execution with the assistant's accompanying text and reasoning.
    Calls {
        /// Assistant text accompanying the proposed calls.
        content: Option<String>,
        /// Reasoning supplied by the template parser for this message.
        reasoning: Option<String>,
        /// Ordered calls requested by the model. An empty batch is rejected.
        calls: Vec<ToolCall>,
    },
}

/// Renders one inference step for a provider and interprets its model's response.
pub trait ToolAdapter: Send + Sync {
    /// Create inference state owned by a single tool turn, when the engine needs it.
    fn begin_turn(&self, _model_id: &str) -> Result<Option<Box<dyn ToolTurn>>> {
        Ok(None)
    }

    /// Evaluate one request, returning final text or parsed tool calls.
    fn infer<'a>(&'a self, request: ToolRequest<'a>) -> BoxFuture<'a, Result<ToolResponse>>;

    /// Run one inference step and report the rendered input if supported.
    fn infer_observed<'a>(
        &'a self,
        request: ToolRequest<'a>,
        _observer: Option<PromptObserver>,
    ) -> BoxFuture<'a, Result<ToolResponse>> {
        self.infer(request)
    }

    /// Run inference with the tool exchanges belonging to each completed turn.
    fn infer_recorded<'a>(
        &'a self,
        request: ToolRequest<'a>,
        _past_exchanges: &'a [Vec<ToolExchange>],
        observer: Option<PromptObserver>,
    ) -> BoxFuture<'a, Result<ToolResponse>> {
        self.infer_observed(request, observer)
    }

    /// Run a step and optionally report provisional assistant text to the frontend.
    fn infer_with_observers<'a>(
        &'a self,
        request: ToolRequest<'a>,
        past_exchanges: &'a [Vec<ToolExchange>],
        observers: TurnObservers,
    ) -> BoxFuture<'a, Result<ToolResponse>> {
        self.infer_recorded(request, past_exchanges, observers.prompt)
    }
}

/// Inference state that survives tool exchanges within a turn and is dropped on cancellation.
pub trait ToolTurn: Send {
    /// Evaluate the next step using the same engine conversation.
    fn infer<'a>(
        &'a mut self,
        request: ToolRequest<'a>,
        past_exchanges: &'a [Vec<ToolExchange>],
        observer: Option<PromptObserver>,
    ) -> BoxFuture<'a, Result<ToolResponse>>;
}

/// Use the provider's single-prompt entry point for an empty registry. Otherwise start tool
/// inference with empty history.
pub(crate) async fn send(
    provider: &dyn Provider,
    model_id: &str,
    prompt: &str,
    tools: &Tools,
) -> Result<Message> {
    if tools.is_empty() {
        return provider.send(model_id, prompt).await;
    }
    chat(
        provider,
        model_id,
        None,
        &[],
        prompt,
        tools,
        TurnObservers::default(),
    )
    .await
}

/// Run a turn with caller-owned history and discard its intermediate tool exchanges.
pub(crate) async fn chat(
    provider: &dyn Provider,
    model_id: &str,
    system_prompt: Option<&str>,
    history: &[Turn],
    prompt: &str,
    tools: &Tools,
    observers: TurnObservers,
) -> Result<Message> {
    let request = ToolRequest {
        model_id,
        system_prompt,
        history,
        prompt,
        tools: &[],
        exchanges: &[],
    };
    Ok(chat_recorded(provider, request, tools, &[], observers)
        .await?
        .0)
}

/// Use observed chat for an empty tool set. Otherwise budget eight calls across all steps,
/// checking each batch before dispatch. Unknown names produce recoverable error outputs.
/// Execution errors abort the turn. A successful finishing tool skips later calls in its
/// batch and supplies the final reply.
pub(crate) async fn chat_recorded(
    provider: &dyn Provider,
    request: ToolRequest<'_>,
    tools: &Tools,
    past_exchanges: &[Vec<ToolExchange>],
    observers: TurnObservers,
) -> Result<(Message, Vec<ToolExchange>)> {
    if tools.is_empty() {
        let reply = provider
            .chat_observed(
                request.model_id,
                request.system_prompt,
                request.history,
                request.prompt,
                observers.prompt,
            )
            .await?;
        return Ok((reply, Vec::new()));
    }
    let specs = tools.specs();
    let request = ToolRequest {
        tools: &specs,
        ..request
    };
    let adapter = provider
        .tool_adapter()
        .context("this provider does not support tool use")?;
    let mut turn = adapter.begin_turn(request.model_id)?;
    let mut exchanges = Vec::new();
    let mut remaining_calls = MAX_CALLS_PER_PROMPT as usize;

    loop {
        let request = ToolRequest {
            exchanges: &exchanges,
            ..request
        };
        let response = match &mut turn {
            Some(turn) => {
                turn.infer(request, past_exchanges, observers.prompt.clone())
                    .await?
            }
            None => {
                adapter
                    .infer_with_observers(request, past_exchanges, observers.clone())
                    .await?
            }
        };
        let (content, reasoning, calls) = match response {
            ToolResponse::Reply(reply) => {
                return Ok((Message::new(stream::iter([Ok(reply)])), exchanges));
            }
            ToolResponse::Calls {
                content,
                reasoning,
                calls,
            } => (content, reasoning, calls),
        };
        if let Some(preview) = &observers.reply_preview {
            preview("");
        }
        ensure!(
            !calls.is_empty(),
            "provider returned an empty tool call list"
        );
        ensure!(
            calls.len() <= remaining_calls,
            "the model exceeded the limit of {MAX_CALLS_PER_PROMPT} tool calls per prompt"
        );
        remaining_calls -= calls.len();

        let mut completed = Vec::with_capacity(calls.len());
        let mut final_reply = None;
        for call in calls {
            let tool = tools.get(&call.name);
            let output = match tool {
                Some(tool) => tool
                    .call(call.arguments.clone())
                    .await
                    .with_context(|| format!("failed to call tool '{}'", call.name))?,
                None => ToolOutput::error(format!("Unknown tool '{}'", call.name)),
            };
            if !output.is_error && tool.is_some_and(|tool| tool.finishes_turn()) {
                final_reply = Some(match &output.content {
                    serde_json::Value::String(text) if text.is_empty() => {
                        content.clone().unwrap_or_default()
                    }
                    serde_json::Value::String(text) => text.clone(),
                    value => value.to_string(),
                });
            }
            completed.push((call, output));
            if final_reply.is_some() {
                break;
            }
        }
        exchanges.push(ToolExchange {
            content,
            reasoning,
            calls: completed,
        });
        if let Some(reply) = final_reply {
            return Ok((Message::new(stream::iter([Ok(reply)])), exchanges));
        }
    }
}

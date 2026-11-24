use super::{PromptObserver, Provider, Turn};
use crate::{
    stream::Message,
    tool::{ToolArguments, ToolOutput, ToolSpec, Tools},
};
use anyhow::{Context, Result, ensure};
use futures_util::{future::BoxFuture, stream};

/// The current inference turn, including tool exchanges that have already completed.
pub struct ToolRequest<'a> {
    pub model_id: &'a str,
    pub system_prompt: Option<&'a str>,
    pub history: &'a [Turn],
    pub prompt: &'a str,
    pub tools: &'a [&'a ToolSpec],
    pub exchanges: &'a [ToolExchange],
}

pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: ToolArguments,
}

pub struct ToolExchange {
    pub content: Option<String>,
    pub reasoning: Option<String>,
    pub calls: Vec<(ToolCall, ToolOutput)>,
}

pub enum ToolResponse {
    Reply(String),
    Calls {
        content: Option<String>,
        reasoning: Option<String>,
        calls: Vec<ToolCall>,
    },
}

/// Renders one inference step for a provider and interprets its model's response.
pub trait ToolAdapter: Send + Sync {
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
}

pub(crate) async fn send(
    provider: &dyn Provider,
    model_id: &str,
    prompt: &str,
    tools: &Tools,
) -> Result<Message> {
    if tools.is_empty() {
        return provider.send(model_id, prompt).await;
    }
    chat(provider, model_id, None, &[], prompt, tools, None).await
}

pub(crate) async fn chat(
    provider: &dyn Provider,
    model_id: &str,
    system_prompt: Option<&str>,
    history: &[Turn],
    prompt: &str,
    tools: &Tools,
    observer: Option<PromptObserver>,
) -> Result<Message> {
    let request = ToolRequest {
        model_id,
        system_prompt,
        history,
        prompt,
        tools: &[],
        exchanges: &[],
    };
    Ok(chat_recorded(provider, request, tools, &[], observer)
        .await?
        .0)
}

pub(crate) async fn chat_recorded(
    provider: &dyn Provider,
    request: ToolRequest<'_>,
    tools: &Tools,
    past_exchanges: &[Vec<ToolExchange>],
    observer: Option<PromptObserver>,
) -> Result<(Message, Vec<ToolExchange>)> {
    if tools.is_empty() {
        let reply = provider
            .chat_observed(
                request.model_id,
                request.system_prompt,
                request.history,
                request.prompt,
                observer,
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
    let mut exchanges = Vec::new();
    const MAX_CALLS: usize = 8;
    let mut remaining_calls = MAX_CALLS;

    loop {
        let response = adapter
            .infer_recorded(
                ToolRequest {
                    exchanges: &exchanges,
                    ..request
                },
                past_exchanges,
                observer.clone(),
            )
            .await?;
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
        ensure!(
            !calls.is_empty(),
            "provider returned an empty tool call list"
        );
        ensure!(
            calls.len() <= remaining_calls,
            "the model exceeded the limit of {MAX_CALLS} tool calls per prompt"
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

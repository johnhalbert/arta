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
    if tools.is_empty() {
        return provider
            .chat_observed(model_id, system_prompt, history, prompt, observer)
            .await;
    }
    let specs = tools.specs();
    let adapter = provider
        .tool_adapter()
        .context("this provider does not support tool use")?;
    let mut exchanges = Vec::new();
    const MAX_CALLS: usize = 8;
    let mut remaining_calls = MAX_CALLS;

    loop {
        let response = adapter
            .infer_observed(
                ToolRequest {
                    model_id,
                    system_prompt,
                    history,
                    prompt,
                    tools: &specs,
                    exchanges: &exchanges,
                },
                observer.clone(),
            )
            .await?;
        let (content, reasoning, calls) = match response {
            ToolResponse::Reply(reply) => return Ok(Message::new(stream::iter([Ok(reply)]))),
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
        for call in calls {
            let output = match tools.get(&call.name) {
                Some(tool) => tool
                    .call(call.arguments.clone())
                    .await
                    .with_context(|| format!("failed to call tool '{}'", call.name))?,
                None => ToolOutput::error(format!("Unknown tool '{}'", call.name)),
            };
            completed.push((call, output));
        }
        exchanges.push(ToolExchange {
            content,
            reasoning,
            calls: completed,
        });
    }
}

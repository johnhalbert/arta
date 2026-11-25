//! Translates Arta tool requests into Needle's stateful completion protocol.

use super::{Needle, NeedleConfig};
use crate::{
    provider::{
        PromptObserver,
        cactus::{RuntimeConfig, runtime::Worker},
        tool::{ToolCall, ToolExchange, ToolRequest, ToolResponse, ToolTurn},
    },
    tool::{ToolArguments, ToolSpec},
};
use anyhow::{Context, Result, bail, ensure};
use futures_util::future::BoxFuture;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{num::NonZeroU32, sync::Arc};

pub(super) struct Inference {
    worker: Worker,
    max_tokens: NonZeroU32,
    input: Value,
}

#[derive(Deserialize)]
pub(super) struct Completion {
    #[serde(rename = "type")]
    kind: Option<String>,
    success: bool,
    error: Option<String>,
    #[serde(default)]
    function_calls: Vec<Call>,
    reasoning: Option<String>,
    confidence: Option<f64>,
    validation: Option<Validation>,
}

#[derive(Deserialize)]
pub(super) struct Call {
    name: String,
    pub(super) arguments: ToolArguments,
}

#[derive(Deserialize)]
struct Validation {
    #[serde(default)]
    ungrounded: Vec<String>,
    #[serde(default)]
    negation: bool,
}

impl Completion {
    pub(super) fn actionable_calls(&self, settings: &NeedleConfig) -> Result<&[Call]> {
        if self.kind.as_deref() != Some("call") || self.function_calls.is_empty() {
            return Ok(&[]);
        }
        if let Some(minimum) = settings.min_confidence
            && !self.confidence.is_some_and(|score| score >= minimum)
        {
            return Ok(&[]);
        }
        if let Some(validation) = self.validation.as_ref().filter(|_| settings.strict) {
            ensure!(
                !validation.negation,
                "Needle reported a negated tool request"
            );
            ensure!(
                validation.ungrounded.is_empty(),
                "Needle reported ungrounded tool arguments: {}",
                validation.ungrounded.join(", ")
            );
        }
        Ok(&self.function_calls)
    }

    fn matches(&self, exchange: &ToolExchange) -> bool {
        self.kind.as_deref() == Some("call")
            && self.function_calls.len() == exchange.calls.len()
            && self
                .function_calls
                .iter()
                .zip(&exchange.calls)
                .all(|(prediction, (call, _))| {
                    prediction.name == call.name && prediction.arguments == call.arguments
                })
    }
}

impl Inference {
    pub(super) async fn start(
        settings: &NeedleConfig,
        runtime: &RuntimeConfig,
        system: Option<&str>,
        tools: Vec<Value>,
    ) -> Result<Self> {
        let system = match (settings.system.as_deref(), system) {
            (Some(facts), Some(prompt)) => Some(format!("{facts}\n\n{prompt}")),
            (facts, prompt) => facts.or(prompt).map(str::to_owned),
        };
        ensure!(
            !system
                .as_deref()
                .is_some_and(|system| system.contains('\0')),
            "Needle system prompt contains a NUL character"
        );
        let mut worker = Worker::start(runtime, "needle")?;
        worker
            .request::<()>(json!({
                "operation": "start",
                "threads": settings.threads,
                "options": {
                    "tools": tools,
                    "system": system,
                    "weights": settings.weights,
                    "tool_index_path": settings.tool_index_path,
                    "buffer_size": settings.buffer_size,
                    "auto_date": settings.auto_date,
                }
            }))
            .await?;
        Ok(Self {
            worker,
            max_tokens: settings.max_tokens,
            input: json!({"system": system, "tools": tools}),
        })
    }

    pub(super) async fn complete(
        &mut self,
        text: &str,
        observer: Option<&PromptObserver>,
    ) -> Result<Completion> {
        ensure!(
            !text.contains('\0'),
            "Needle input contains a NUL character"
        );
        if let Some(observe) = observer {
            self.input["text"] = json!(text);
            observe(&serde_json::to_string_pretty(&self.input)?);
        }
        let response: Completion = self
            .worker
            .request(json!({
                "operation": "complete", "text": text, "max_tokens": self.max_tokens,
            }))
            .await?;
        if !response.success {
            bail!(
                "Needle inference failed: {}",
                response.error.as_deref().unwrap_or("unknown error")
            );
        }
        Ok(response)
    }
}

pub(in crate::provider::cactus) struct NeedleTurn {
    model: Arc<Needle>,
    runtime: Arc<RuntimeConfig>,
    inference: Option<Inference>,
}

impl NeedleTurn {
    pub(super) fn new(model: Arc<Needle>, runtime: Arc<RuntimeConfig>) -> Self {
        Self {
            model,
            runtime,
            inference: None,
        }
    }
}

impl ToolTurn for NeedleTurn {
    fn infer<'a>(
        &'a mut self,
        request: ToolRequest<'a>,
        past_exchanges: &'a [Vec<ToolExchange>],
        observer: Option<PromptObserver>,
    ) -> BoxFuture<'a, Result<ToolResponse>> {
        Box::pin(async move {
            let settings = &self.model.settings;
            let observer = observer.as_ref();
            if request.exchanges.len() >= settings.max_steps.get() as usize {
                return Ok(ToolResponse::Reply(tool_reply(request.exchanges)));
            }
            let response = match self.inference.take() {
                Some(mut inference) => {
                    let exchange = request
                        .exchanges
                        .last()
                        .context("Needle expected a tool result batch")?;
                    let response = inference
                        .complete(&tool_results(exchange), observer)
                        .await?;
                    self.inference = Some(inference);
                    response
                }
                None => {
                    let tools = request
                        .tools
                        .iter()
                        .map(|spec| schema(spec, settings))
                        .collect();
                    let mut inference =
                        Inference::start(settings, &self.runtime, request.system_prompt, tools)
                            .await?;
                    if !settings.stateless {
                        for (index, turn) in request.history.iter().enumerate() {
                            let mut prior = inference.complete(&turn.prompt, None).await?;
                            for exchange in past_exchanges.get(index).into_iter().flatten() {
                                if !prior.matches(exchange) {
                                    break;
                                }
                                prior = inference.complete(&tool_results(exchange), None).await?;
                            }
                        }
                    }
                    let mut response = inference.complete(request.prompt, observer).await?;
                    // Direct ToolAdapter callers can supply previously completed steps.
                    for exchange in request.exchanges {
                        ensure!(
                            response.matches(exchange),
                            "Needle could not replay the previous tool calls"
                        );
                        response = inference
                            .complete(&tool_results(exchange), observer)
                            .await?;
                    }
                    self.inference = Some(inference);
                    response
                }
            };

            let calls = response.actionable_calls(settings)?;
            if calls.is_empty() {
                return Ok(ToolResponse::Reply(tool_reply(request.exchanges)));
            }
            let calls = calls
                .iter()
                .enumerate()
                .map(|(index, call)| ToolCall {
                    id: format!("needle-{}-{index}", request.exchanges.len()),
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                })
                .collect();
            Ok(ToolResponse::Calls {
                content: None,
                reasoning: response.reasoning,
                calls,
            })
        })
    }
}

fn schema(spec: &ToolSpec, settings: &NeedleConfig) -> Value {
    let mut schema = json!({
        "name": spec.name,
        "description": spec.description,
        "parameters": spec.input_schema,
    });
    if let Some(triggers) = settings.triggers.get(&spec.name) {
        schema["triggers"] = json!(triggers);
    }
    schema
}

fn tool_results(exchange: &ToolExchange) -> String {
    Value::Array(
        exchange
            .calls
            .iter()
            .map(|(_, output)| {
                if output.is_error {
                    json!({ "error": output.content })
                } else {
                    output.content.clone()
                }
            })
            .collect(),
    )
    .to_string()
}

fn tool_reply(exchanges: &[ToolExchange]) -> String {
    if exchanges.is_empty() {
        return "Needle found no matching tool call.".into();
    }
    exchanges
        .iter()
        .flat_map(|exchange| &exchange.calls)
        .map(|(_, output)| match output.content.as_str() {
            Some(text) if !output.is_error => text.to_owned(),
            _ if output.is_error => format!("Tool error: {}", output.content),
            _ => output.content.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

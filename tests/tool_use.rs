//! Exercises the library tool loop through a bound agent using an in-process function and a
//! scripted adapter that inspects its follow-up request.

use anyhow::{Result, bail};
use arta::{
    agent::AgentOptions,
    config::Config,
    harness::Harness,
    model::{Model, ModelId},
    provider::{
        PromptObserver, Provider,
        tool::{ToolAdapter, ToolCall, ToolRequest, ToolResponse},
    },
    stream::Message,
    tool::{Tool, ToolArguments, ToolOutput, ToolSpec},
};
use futures_util::future::BoxFuture;
use serde_json::json;
use std::sync::Arc;

/// Implements metadata, inference, and tool parsing in one fixture so tests can observe both
/// sides of a tool exchange.
struct ToolProvider;

impl Model for ToolProvider {
    fn id(&self) -> &str {
        "test"
    }

    fn trained_context_length(&self) -> u32 {
        4096
    }
}

impl Provider for ToolProvider {
    fn model(&self, _: &str) -> Result<Arc<dyn Model>> {
        Ok(Arc::new(Self))
    }

    fn send<'a>(&'a self, _: &'a str, _: &'a str) -> BoxFuture<'a, Result<Message>> {
        Box::pin(async { bail!("expected tool-aware inference") })
    }

    fn tool_adapter(&self) -> Option<&dyn ToolAdapter> {
        Some(self)
    }
}

impl ToolAdapter for ToolProvider {
    fn infer<'a>(&'a self, request: ToolRequest<'a>) -> BoxFuture<'a, Result<ToolResponse>> {
        Box::pin(async move {
            assert_eq!(request.model_id, "test");
            assert_eq!(request.system_prompt, Some("Answer using the lookup tool."));
            assert_eq!(request.tools.len(), 1);
            assert_eq!(request.tools[0].name, "lookup");
            if let Some(exchange) = request.exchanges.last() {
                assert_eq!(exchange.calls.len(), 1);
                let (call, output) = &exchange.calls[0];
                assert_eq!(call.id, "lookup-1");
                assert_eq!(output.content, json!({"value": "blue"}));
                assert!(!output.is_error);
                return Ok(ToolResponse::Reply("The answer is blue.".into()));
            }
            Ok(ToolResponse::Calls {
                content: None,
                reasoning: None,
                calls: vec![ToolCall {
                    id: "lookup-1".into(),
                    name: "lookup".into(),
                    arguments: serde_json::from_value(json!({"key": request.prompt}))?,
                }],
            })
        })
    }

    fn infer_observed<'a>(
        &'a self,
        request: ToolRequest<'a>,
        observer: Option<PromptObserver>,
    ) -> BoxFuture<'a, Result<ToolResponse>> {
        if let Some(observe) = observer {
            let tool_result = request
                .exchanges
                .last()
                .map(|step| &step.calls[0].1.content);
            observe(&format!(
                "<rendered:{}> {}",
                request.tools[0].name,
                tool_result.map_or_else(String::new, serde_json::Value::to_string)
            ));
        }
        self.infer(request)
    }
}

struct Lookup(ToolSpec);

impl Tool for Lookup {
    fn spec(&self) -> &ToolSpec {
        &self.0
    }

    fn call(&self, arguments: ToolArguments) -> BoxFuture<'_, Result<ToolOutput>> {
        Box::pin(async move {
            assert_eq!(arguments["key"], "favorite color");
            Ok(ToolOutput::success(json!({"value": "blue"})))
        })
    }
}

#[tokio::test]
async fn an_agent_runs_a_tool_and_returns_the_follow_up_reply() -> Result<()> {
    let mut harness = Harness::new(&Config::default())?;
    harness.register_provider("test", || Ok(Arc::new(ToolProvider)))?;
    let lookup = Arc::new(Lookup(ToolSpec {
        name: "lookup".into(),
        description: "Look up a value".into(),
        input_schema: serde_json::from_value(json!({
            "type": "object",
            "properties": { "key": { "type": "string" } },
            "required": ["key"]
        }))?,
    }));
    harness.register_tool(lookup.clone())?;
    harness.create_session_with_agents(
        "chat",
        [(
            "agent",
            ModelId {
                provider: "test",
                model: "test",
            },
            AgentOptions {
                system_prompt: Some("Answer using the lookup tool."),
                ..Default::default()
            },
        )],
    )?;

    let reply = harness
        .agent("chat", "agent")?
        .prompt("favorite color")
        .await?
        .into_text()
        .await?;
    assert_eq!(reply, "The answer is blue.");

    let (sent, mut captured) = tokio::sync::mpsc::unbounded_channel();
    let observe = Arc::new(move |input: &str| sent.send(input.to_owned()).unwrap());
    let reply = harness
        .agent("chat", "agent")?
        .chat_with_context_observed(
            &[],
            "favorite color",
            "",
            [lookup as Arc<dyn Tool>],
            Some(observe),
        )
        .await?;
    assert_eq!(captured.try_recv()?, "<rendered:lookup> ");
    assert_eq!(
        captured.try_recv()?,
        r#"<rendered:lookup> {"value":"blue"}"#
    );
    assert!(captured.try_recv().is_err());
    assert_eq!(reply.into_text().await?, "The answer is blue.");
    Ok(())
}

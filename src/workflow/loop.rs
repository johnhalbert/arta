use super::{Run, WorkflowEvent, event::Progress, node::Node};
use crate::tool::{Tool, ToolArguments, ToolOutput, ToolSpec, Tools};
use anyhow::{Context, Result};
use futures_util::future::BoxFuture;
use serde_json::{Value, json};
use std::sync::{Arc, OnceLock};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoopIteration {
    pub id: String,
    pub number: usize,
}

pub(super) async fn execute<'a>(run: &mut Run<'a>, id: &str, nodes: &'a [Node]) -> Result<()> {
    let (last, preceding) = nodes.split_last().context("loop must contain nodes")?;
    let (node, agent) = final_agent(last);
    for number in 1.. {
        run.loops.push(LoopIteration {
            id: id.to_owned(),
            number,
        });
        run.sequence(preceding).await?;
        if matches!(last, Node::Loop { .. }) {
            run.sequence(std::slice::from_ref(last)).await?;
        }
        let decision = run.agent(node, agent, true).await?;
        run.loops.pop();
        if decision == Some(LoopDecision::Break) {
            break;
        }
    }
    Ok(())
}

fn final_agent(node: &Node) -> (&str, &str) {
    match node {
        Node::Agent { id, agent } => (id, agent),
        Node::Loop { nodes, .. } => {
            final_agent(nodes.last().expect("validated loop cannot be empty"))
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum LoopDecision {
    Continue,
    Break,
}

/// Registers the final agent's control tools and records its decision for one iteration.
pub(super) struct LoopControl<'a> {
    iteration: &'a LoopIteration,
    decision: Arc<OnceLock<LoopDecision>>,
}

impl<'a> LoopControl<'a> {
    pub(super) fn register(
        tools: &mut Tools,
        iteration: &'a LoopIteration,
        progress: &Progress,
    ) -> Result<Self> {
        let decision = Arc::new(OnceLock::new());
        let Value::Object(schema) = json!({
            "type": "object",
            "properties": {"message": {"type": "string", "description": "Feedback or results to pass to the next workflow node."}},
            "additionalProperties": false,
        }) else {
            unreachable!()
        };
        for (action, name, description) in [
            (
                LoopDecision::Continue,
                "continue",
                "Finish this turn and run another iteration of the current workflow loop.",
            ),
            (
                LoopDecision::Break,
                "break",
                "Finish this turn, exit the current workflow loop, and proceed with the workflow.",
            ),
        ] {
            tools.register(Arc::new(LoopTool {
                action,
                decision: Arc::clone(&decision),
                iteration: iteration.clone(),
                progress: progress.clone(),
                spec: ToolSpec {
                    name: name.into(),
                    description: description.into(),
                    input_schema: schema.clone(),
                },
            }))?;
        }
        Ok(Self {
            iteration,
            decision,
        })
    }

    pub(super) fn instructions(&self) -> String {
        let id = &self.iteration.id;
        format!(
            "You are deciding whether workflow loop '{id}' should repeat. After doing your work for this turn, call `continue` to repeat the loop or `break` to leave it. Use the tool's optional `message` argument to pass your feedback or results to the next node."
        )
    }

    pub(super) fn decision(&self) -> Result<LoopDecision> {
        self.decision.get().copied().with_context(|| {
            format!(
                "loop '{}' requires the final agent to call 'continue' or 'break'",
                self.iteration.id
            )
        })
    }
}

struct LoopTool {
    action: LoopDecision,
    decision: Arc<OnceLock<LoopDecision>>,
    iteration: LoopIteration,
    progress: Progress,
    spec: ToolSpec,
}

impl LoopTool {
    fn decide(&self, arguments: &ToolArguments) -> ToolOutput {
        let message = match arguments.get("message") {
            Some(Value::String(text)) => text.as_str(),
            None => "",
            Some(_) => return ToolOutput::error("message must be a string"),
        };
        if arguments.keys().any(|name| name != "message") {
            return ToolOutput::error("Only the optional message argument is accepted");
        }
        if self.decision.set(self.action).is_err() {
            return ToolOutput::error("A decision has already been made for this loop turn");
        }
        ToolOutput::success(message)
    }
}

impl Tool for LoopTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn finishes_turn(&self) -> bool {
        true
    }

    fn call(&self, arguments: ToolArguments) -> BoxFuture<'_, Result<ToolOutput>> {
        Box::pin(async move {
            let output = self.decide(&arguments);
            self.progress
                .emit(|| WorkflowEvent::LoopToolCall {
                    iteration: self.iteration.clone(),
                    name: self.spec.name.clone(),
                    arguments,
                    error: output
                        .is_error
                        .then(|| output.content.as_str().unwrap_or_default().to_owned()),
                })
                .await;
            Ok(output)
        })
    }
}

use super::{LoopIteration, WorkflowEvent, event::Progress};
use crate::tool::{Tool, ToolArguments, ToolOutput, ToolSpec, Tools};
use anyhow::{Result, ensure};
use futures_util::future::BoxFuture;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::HashSet,
    sync::{Arc, OnceLock},
};

/// One step in a workflow. A loop executes its children in order on each iteration.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Node {
    Agent { id: String, agent: String },
    Loop { id: String, nodes: Vec<Node> },
}

impl Node {
    pub(crate) fn visit_sequence<'a>(
        nodes: &'a [Self],
        ids: &mut HashSet<&'a str>,
        visit: &mut impl FnMut(&str, &str) -> Result<()>,
    ) -> Result<()> {
        ensure!(!nodes.is_empty(), "a workflow or loop must contain nodes");
        for node in nodes {
            let id = match node {
                Self::Agent { id, .. } | Self::Loop { id, .. } => id,
            };
            ensure!(!id.trim().is_empty(), "node id cannot be empty");
            ensure!(ids.insert(id), "duplicate node id '{id}'");
            match node {
                Self::Agent { agent, .. } => visit(id, agent)?,
                Self::Loop { nodes, .. } => Self::visit_sequence(nodes, ids, visit)?,
            }
        }
        Ok(())
    }

    pub(crate) fn final_agent(&self) -> (&str, &str) {
        match self {
            Self::Agent { id, agent } => (id, agent),
            Self::Loop { nodes, .. } => nodes
                .last()
                .expect("validated loop cannot be empty")
                .final_agent(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum LoopDecision {
    Continue,
    Break,
}

impl LoopDecision {
    pub(super) fn register(
        tools: &mut Tools,
        iteration: &LoopIteration,
        progress: &Progress,
    ) -> Result<Arc<OnceLock<Self>>> {
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
                Self::Continue,
                "continue",
                "Finish this turn and run another iteration of the current workflow loop.",
            ),
            (
                Self::Break,
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
        Ok(decision)
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

//! Nested loop execution and its finishing tools. The deciding agent receives a fresh shared
//! decision slot every iteration. There is no configured iteration cap.

use super::{Run, WorkflowEvent, event::Progress, node::Node};
use crate::tool::{Tool, ToolArguments, ToolOutput, ToolSpec, Tools};
use anyhow::{Context, Result};
use futures_util::future::BoxFuture;
use serde_json::{Value, json};
use std::sync::{Arc, OnceLock};

/// Location within one enclosing loop. Event paths list these from outermost to innermost.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoopIteration {
    /// ID of the loop node in the definition.
    pub id: String,
    /// One-based iteration number within this invocation of the loop.
    pub number: usize,
}

/// Run preceding siblings and ask the final leaf to decide. If the last child is a loop,
/// finish its iterations first and revisit its deepest final leaf for the outer decision.
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

/// Descend through successive last children to find the deciding leaf. Validation guarantees
/// every sequence has a last child.
fn final_agent(node: &Node) -> (&str, &str) {
    match node {
        Node::Agent { id, agent } => (id, agent),
        Node::Loop { nodes, .. } => {
            final_agent(nodes.last().expect("validated loop cannot be empty"))
        }
        Node::Image { .. } | Node::Audio { .. } => {
            unreachable!("validated loops must end with an agent")
        }
    }
}

/// Outcome of a successful finishing call for one loop iteration.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum LoopDecision {
    /// Start another iteration at this loop's first child.
    Continue,
    /// Exit this loop and resume the enclosing sequence.
    Break,
}

/// Registers the final agent's control tools and records its decision for one iteration.
pub(super) struct LoopControl<'a> {
    /// Borrowed loop identity used to generate instructions and missing-decision errors.
    iteration: &'a LoopIteration,
    /// Shared with both tools and read after the reply has drained.
    decision: Arc<OnceLock<LoopDecision>>,
}

impl<'a> LoopControl<'a> {
    /// Insert both controls into the deciding turn's registry. They share a single-assignment
    /// cell and identify the same iteration.
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

    /// Name the current loop and describe the optional feedback passed through the finishing
    /// call.
    pub(super) fn instructions(&self) -> String {
        let id = &self.iteration.id;
        format!(
            "You are deciding whether workflow loop '{id}' should repeat. After doing your work for this turn, call `continue` to repeat the loop or `break` to leave it. Use the tool's optional `message` argument to pass your feedback or results to the next node."
        )
    }

    /// Require a successful control call even if inference produced an ordinary text reply.
    pub(super) fn decision(&self) -> Result<LoopDecision> {
        self.decision.get().copied().with_context(|| {
            format!(
                "loop '{}' requires the final agent to call 'continue' or 'break'",
                self.iteration.id
            )
        })
    }
}

/// One of the two control actions. Emits progress for accepted and rejected arguments.
struct LoopTool {
    /// Value to record when this particular control tool succeeds.
    action: LoopDecision,
    /// Single shared slot preventing a second decision within one visit.
    decision: Arc<OnceLock<LoopDecision>>,
    /// Owned iteration identity copied into the progress event for this attempt.
    iteration: LoopIteration,
    /// Emitter used to record calls and rejected arguments.
    progress: Progress,
    /// The advertised control name and optional-message schema for this action.
    spec: ToolSpec,
}

impl LoopTool {
    /// Validate the optional string message and reject unknown keys before writing the
    /// decision. A rejected attempt leaves an unused slot available for retry.
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

//! Per-node memory and execution. Each visit creates a short-lived agent bound to the
//! selected model. The node retains successful turns and tool exchanges across model switches
//! and loop iterations.

use super::{
    LoopIteration, WorkflowEvent,
    event::Progress,
    r#loop::{LoopControl, LoopDecision},
    model::ActiveModel,
};
use crate::{
    agent::Agent,
    config::AgentProfile,
    provider::{Turn, tool::ToolExchange},
    stream::Frame,
    tool::Tools,
};
use anyhow::Result;
use futures_util::StreamExt;
use std::sync::Arc;

/// A workflow node's conversation, independent of the model currently loaded.
#[derive(Default)]
pub(super) struct WorkflowAgent {
    /// Completed visits to this node in order, each with the actual handoff prompt and final
    /// reply.
    history: Vec<Turn>,
    /// One entry per history index, containing the tool batches replayed inside that turn.
    exchanges: Vec<Vec<ToolExchange>>,
}

impl WorkflowAgent {
    /// Install decision tools for a deciding visit. Replay this node's history, stream output
    /// events, then append the turn and its exchanges together after validating any required
    /// decision.
    pub(super) async fn run(
        &mut self,
        model: &ActiveModel,
        profile: &AgentProfile,
        mut tools: Tools,
        prompt: String,
        loop_iteration: Option<&LoopIteration>,
        progress: &Progress,
    ) -> Result<(String, Option<LoopDecision>)> {
        let control = loop_iteration
            .map(|iteration| LoopControl::register(&mut tools, iteration, progress))
            .transpose()?;
        let instructions = control
            .as_ref()
            .map(LoopControl::instructions)
            .unwrap_or_default();
        let agent = Agent::new(
            Arc::clone(&model.provider),
            model.model(),
            Arc::new(tools),
            profile.options(),
        );
        let (reply, exchanges) = agent
            .chat_recorded(
                &self.history,
                &self.exchanges,
                &prompt,
                &instructions,
                progress.prompt_observer(),
            )
            .await?;
        let mut stream = reply.into_stream();
        let mut reply = String::new();
        while let Some(frame) = stream.next().await.transpose()? {
            if let Frame::Chunk(chunk) = frame {
                reply.push_str(&chunk);
                progress.emit(|| WorkflowEvent::Output(chunk)).await;
            }
        }
        let decision = control.as_ref().map(LoopControl::decision).transpose()?;
        self.history.push(Turn {
            prompt,
            reply: reply.clone(),
        });
        self.exchanges.push(exchanges);
        Ok((reply, decision))
    }
}

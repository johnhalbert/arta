use super::{
    LoopIteration, WorkflowEvent, event::Progress, model::ActiveModel, node::LoopDecision,
};
use crate::{
    agent::Agent,
    config::AgentProfile,
    provider::{Turn, tool::ToolExchange},
    stream::Frame,
    tool::Tools,
};
use anyhow::{Context, Result};
use futures_util::StreamExt;
use std::sync::Arc;

/// A workflow node's conversation, independent of the model currently loaded.
#[derive(Default)]
pub(super) struct WorkflowAgent {
    history: Vec<Turn>,
    exchanges: Vec<Vec<ToolExchange>>,
}

impl WorkflowAgent {
    pub(super) async fn run(
        &mut self,
        model: &ActiveModel,
        profile: &AgentProfile,
        mut tools: Tools,
        prompt: String,
        loop_iteration: Option<&LoopIteration>,
        progress: &Progress,
    ) -> Result<(String, Option<LoopDecision>)> {
        let decision = loop_iteration
            .map(|iteration| LoopDecision::register(&mut tools, iteration, progress))
            .transpose()?;
        let loop_id = loop_iteration.map(|iteration| iteration.id.as_str());
        let instructions = loop_id.map(|id| format!(
            "You are deciding whether workflow loop '{id}' should repeat. After doing your work for this turn, call `continue` to repeat the loop or `break` to leave it. Use the tool's optional `message` argument to pass your feedback or results to the next node."
        )).unwrap_or_default();
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
        let decision = loop_id
            .zip(decision)
            .map(|(id, decision)| {
                decision.get().copied().with_context(|| {
                    format!("loop '{id}' requires the final agent to call 'continue' or 'break'")
                })
            })
            .transpose()?;
        self.history.push(Turn {
            prompt,
            reply: reply.clone(),
        });
        self.exchanges.push(exchanges);
        Ok((reply, decision))
    }
}

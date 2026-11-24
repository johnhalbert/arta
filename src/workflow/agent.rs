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

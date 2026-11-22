use crate::{
    model::Model,
    provider::{Provider, Turn, tool},
    stream::Message,
    tool::Tools,
};
use anyhow::Result;
use std::sync::Arc;

/// Runs prompts against a model selected when the agent is created.
pub struct Agent {
    model: Arc<dyn Model>,
    provider: Arc<dyn Provider>,
    tools: Arc<Tools>,
}

impl Agent {
    pub(crate) fn new(
        provider: Arc<dyn Provider>,
        model: Arc<dyn Model>,
        tools: Arc<Tools>,
    ) -> Self {
        Self {
            model,
            provider,
            tools,
        }
    }

    pub fn provider(&self) -> &dyn Provider {
        self.provider.as_ref()
    }

    pub fn model(&self) -> &dyn Model {
        self.model.as_ref()
    }

    /// Run one independent prompt against the agent's model.
    pub async fn prompt(&self, prompt: &str) -> Result<Message> {
        tool::send(self.provider(), self.model.id(), prompt, &self.tools).await
    }

    /// Run a prompt with conversation history supplied by the caller.
    pub async fn chat(&self, history: &[Turn], prompt: &str) -> Result<Message> {
        tool::chat(
            self.provider(),
            self.model.id(),
            history,
            prompt,
            &self.tools,
        )
        .await
    }
}

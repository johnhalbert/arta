use crate::{model::Model, provider::Provider, stream::Message};
use anyhow::Result;
use std::sync::Arc;

/// Runs prompts against a model selected when the agent is created.
pub struct Agent {
    model: Arc<dyn Model>,
    provider: Arc<dyn Provider>,
}

impl Agent {
    pub(crate) fn new(provider: Arc<dyn Provider>, model: Arc<dyn Model>) -> Self {
        Self { model, provider }
    }

    pub fn provider(&self) -> &dyn Provider {
        self.provider.as_ref()
    }

    pub fn model(&self) -> &dyn Model {
        self.model.as_ref()
    }

    /// Run one independent prompt against the agent's model.
    pub async fn prompt(&self, prompt: &str) -> Result<Message> {
        self.provider.send(self.model.id(), prompt).await
    }
}

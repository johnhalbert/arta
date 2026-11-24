pub mod llama_cpp;
pub mod registry;
pub mod tool;

use crate::{model::Model, stream::Message};
use anyhow::{Result, bail};
use futures_util::future::BoxFuture;
use std::sync::Arc;

/// One completed exchange in a conversation, in user then assistant order.
#[derive(Clone)]
pub struct Turn {
    pub prompt: String,
    pub reply: String,
}

/// Manages models and runs inference without retaining conversation state.
pub trait Provider: Send + Sync {
    /// Get a shared reference to a model that this provider can use for inference.
    fn model(&self, model_id: &str) -> Result<Arc<dyn Model>>;

    /// Release the current model's resources before loading its replacement.
    fn switch_model(&self, _current_id: &str, _model_id: &str) -> Result<Arc<dyn Model>> {
        bail!("this provider does not support model switching")
    }

    fn send<'a>(&'a self, model_id: &'a str, prompt: &'a str) -> BoxFuture<'a, Result<Message>>;

    fn tool_adapter(&self) -> Option<&dyn tool::ToolAdapter> {
        None
    }

    /// Reply using the completed turns in chronological order.
    fn chat<'a>(
        &'a self,
        model_id: &'a str,
        history: &'a [Turn],
        prompt: &'a str,
    ) -> BoxFuture<'a, Result<Message>> {
        if history.is_empty() {
            return self.send(model_id, prompt);
        }
        Box::pin(async { bail!("this provider does not support multi-turn conversations") })
    }

    /// Reply with an optional system prompt that precedes the conversation.
    fn chat_with_system<'a>(
        &'a self,
        model_id: &'a str,
        system_prompt: Option<&'a str>,
        history: &'a [Turn],
        prompt: &'a str,
    ) -> BoxFuture<'a, Result<Message>> {
        if system_prompt.is_none() {
            return self.chat(model_id, history, prompt);
        }
        Box::pin(async { bail!("this provider does not support system prompts") })
    }
}

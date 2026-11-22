pub mod llama_cpp;
pub mod registry;

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

    fn send<'a>(&'a self, model_id: &'a str, prompt: &'a str) -> BoxFuture<'a, Result<Message>>;

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
}

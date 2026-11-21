pub mod llama_cpp;
pub mod registry;

use crate::{model::Model, stream::Message};
use anyhow::Result;
use futures_util::future::BoxFuture;
use std::sync::Arc;

/// Manages its models and runs each prompt without retaining conversation state.
pub trait Provider: Send + Sync {
    /// Get a shared reference to a model that this provider can use for inference.
    fn model(&self, model_id: &str) -> Result<Arc<dyn Model>>;

    fn send<'a>(&'a self, model_id: &'a str, prompt: &'a str) -> BoxFuture<'a, Result<Message>>;
}

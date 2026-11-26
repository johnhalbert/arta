//! Provider contract for model selection, prompt rendering, inference, and optional tool use.
//! Implementations expose capabilities through trait methods. Callers supply conversation
//! state on every request.

pub mod audio;
pub mod audio_cpp;
pub mod cactus;
pub mod image;
pub mod llama_cpp;
mod native;
pub mod registry;
pub mod stable_diffusion_cpp;
pub mod tool;

use crate::{config::Config, model::Model, stream::Message};
use anyhow::{Result, bail};
use futures_util::future::BoxFuture;
use std::sync::Arc;

/// One completed exchange in a conversation, in user then assistant order.
#[derive(Clone)]
pub struct Turn {
    /// User input as supplied to the completed turn.
    pub prompt: String,
    /// Assistant text retained for subsequent turns.
    pub reply: String,
}

/// Receives the provider's rendered text before tokenization, once per inference.
pub type PromptObserver = Arc<dyn Fn(&str) + Send + Sync>;

/// Receives the current provisional assistant text. Each snapshot replaces the previous one.
pub type ReplyPreviewObserver = Arc<dyn Fn(&str) + Send + Sync>;

/// Optional notifications for a frontend displaying an ongoing turn.
#[derive(Clone, Default)]
pub struct TurnObservers {
    /// Rendered input for each inference step.
    pub prompt: Option<PromptObserver>,
    /// Display-only snapshots from inference steps that buffer their final response.
    pub reply_preview: Option<ReplyPreviewObserver>,
}

/// Manages models and runs inference without retaining conversation state.
pub trait Provider: Send + Sync {
    /// Get a shared reference to a model that this provider can use for inference.
    fn model(&self, model_id: &str) -> Result<Arc<dyn Model>>;

    /// Release the current model's resources before loading its replacement.
    fn switch_model(&self, _current_id: &str, _model_id: &str) -> Result<Arc<dyn Model>> {
        bail!("this provider does not support model switching")
    }

    /// Release resources held for a model. Providers with no model cache can keep the default.
    fn unload_model(&self, _model_id: &str) -> Result<()> {
        Ok(())
    }

    /// Create a provider with agent-specific settings, sharing process-wide resources as needed.
    fn configured(&self, _config: &Config) -> Result<Arc<dyn Provider>> {
        bail!("this provider does not support agent-specific settings")
    }

    /// Run one prompt without supplied history and return its asynchronous text chunks.
    fn send<'a>(&'a self, model_id: &'a str, prompt: &'a str) -> BoxFuture<'a, Result<Message>>;

    /// Return the adapter for structured tool inference when the implementation supports it.
    fn tool_adapter(&self) -> Option<&dyn tool::ToolAdapter> {
        None
    }

    /// Check whether a model can receive automatically advertised tools.
    fn supports_model_tools(&self, _model_id: &str) -> Result<bool> {
        Ok(self.tool_adapter().is_some())
    }

    /// Return the image generation backend when this provider supports that capability.
    fn image_generator(self: Arc<Self>) -> Option<Arc<dyn image::ImageGenerator>> {
        None
    }

    /// Return the audio generation backend when this provider supports that capability.
    fn audio_generator(self: Arc<Self>) -> Option<Arc<dyn audio::AudioGenerator>> {
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

    /// Reply and report the rendered input when the provider supports prompt observation.
    fn chat_observed<'a>(
        &'a self,
        model_id: &'a str,
        system_prompt: Option<&'a str>,
        history: &'a [Turn],
        prompt: &'a str,
        _observer: Option<PromptObserver>,
    ) -> BoxFuture<'a, Result<Message>> {
        self.chat_with_system(model_id, system_prompt, history, prompt)
    }
}

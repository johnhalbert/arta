//! Owns loaded decoder weights, the resolved chat template, and a backend guard that outlives
//! the weights. Public model handles share only the identifier and trained context length.

mod template;

use super::{ChatConfig, LlamaCPPModelConfig};
use crate::{
    model::Model,
    provider::{PromptObserver, Turn},
};
use anyhow::{Context, Result, ensure};
use llama_cpp_4::prelude::{
    ChatParams, ChatTemplates, LlamaBackend, LlamaContext, LlamaContextParams, LlamaModel,
    LlamaToken,
};
use serde_json::{Value, json};
use std::{path::Path, sync::Arc};
use template::ModelTemplate;

/// Resident resources shared by the cache and active inference workers.
pub(super) struct LlamaCPPModel {
    /// Metadata that agents can retain after the cache evicts the native weights.
    pub(super) info: Arc<ModelInfo>,
    /// Rendering path resolved once from configuration and GGUF metadata.
    template: ModelTemplate,
    /// Shared rendering options used for both ordinary and tool-aware prompts.
    pub(super) chat: Arc<ChatConfig>,
    /// Native decoder weights and vocabulary used by every context created for this load.
    pub(super) inner: LlamaModel,
    /// Declared after the weights so the native model drops before its backend guard.
    backend: Arc<LlamaBackend>,
}

impl LlamaCPPModel {
    /// Check the base file and every shard, load with model-specific parameters, reject
    /// unsupported architectures, and select the prompt template before publishing the loaded
    /// model.
    pub(super) fn load(
        id: Arc<str>,
        backend: Arc<LlamaBackend>,
        config: &LlamaCPPModelConfig,
        chat: Arc<ChatConfig>,
    ) -> Result<Self> {
        let path = &config.path;
        check_model_file(path)?;
        for shard in &config.additional_shards {
            check_model_file(shard)?;
        }
        let parameters = config.params(&backend)?;
        let inner = if config.additional_shards.is_empty() {
            LlamaModel::load_from_file(&backend, path, &parameters)
        } else {
            let paths: Vec<_> = std::iter::once(path)
                .chain(&config.additional_shards)
                .collect();
            LlamaModel::load_from_splits(&backend, &paths, &parameters)
        }
        .with_context(|| format!("failed to load GGUF model '{}'", path.display()))?;
        ensure!(
            inner.has_decoder() && !inner.has_encoder() && !inner.is_diffusion(),
            "model '{}' does not support decoder-only text generation",
            path.display()
        );

        let template = ModelTemplate::from_model(&inner, chat.template.as_deref())?;

        Ok(Self {
            info: Arc::new(ModelInfo {
                id,
                trained_context_length: inner.n_ctx_train(),
            }),
            template,
            chat,
            inner,
            backend,
        })
    }

    /// Produce token IDs, stop strings, and optional template parameters from supplied
    /// conversation state.
    pub(super) fn prepare_prompt(
        &self,
        system_prompt: Option<&str>,
        history: &[Turn],
        prompt: &str,
        observer: Option<&PromptObserver>,
    ) -> Result<(Vec<LlamaToken>, Vec<String>, Option<ChatParams>)> {
        self.template.prepare_prompt(
            &self.inner,
            &self.chat,
            system_prompt,
            history,
            prompt,
            observer,
        )
    }

    /// Allocate an inference context borrowing this model and its backend guard.
    pub(super) fn new_context(&self, params: LlamaContextParams) -> Result<LlamaContext<'_>> {
        self.inner
            .new_context(&self.backend, params)
            .context("failed to create inference context")
    }

    /// Obtain the resolved template after verifying that it advertises tool-call support.
    pub(super) fn tool_template(&self) -> Result<&ChatTemplates> {
        self.template.tool_template()
    }
}

/// Check that a configured path resolves to a regular file before passing it to the native
/// loader.
fn check_model_file(path: &Path) -> Result<()> {
    let metadata = path
        .metadata()
        .with_context(|| format!("cannot access model '{}'", path.display()))?;
    ensure!(
        metadata.is_file(),
        "model '{}' must be a file",
        path.display()
    );
    Ok(())
}

/// Agent-visible identity and capacity that can outlive the native weights.
pub(super) struct ModelInfo {
    /// Configuration key used to reacquire the resident model.
    id: Arc<str>,
    /// Model-reported training length used to compute the default context capacity.
    trained_context_length: u32,
}

impl Model for ModelInfo {
    fn id(&self) -> &str {
        &self.id
    }

    fn trained_context_length(&self) -> u32 {
        self.trained_context_length
    }
}

/// Build role-tagged messages for ordinary chat in chronological order.
pub(super) fn chat_messages(
    system_prompt: Option<&str>,
    history: &[Turn],
    prompt: &str,
) -> Vec<Value> {
    chat_messages_with(system_prompt, history, prompt, |_, _| {})
}

/// Insert each previous turn's tool exchanges between its user prompt and assistant reply.
/// Finish with the new user message.
pub(super) fn chat_messages_with(
    system_prompt: Option<&str>,
    history: &[Turn],
    prompt: &str,
    mut add_exchanges: impl FnMut(usize, &mut Vec<Value>),
) -> Vec<Value> {
    let mut messages =
        Vec::with_capacity(history.len() * 2 + 1 + usize::from(system_prompt.is_some()));
    if let Some(content) = system_prompt {
        messages.push(json!({ "role": "system", "content": content }));
    }
    for (index, turn) in history.iter().enumerate() {
        messages.push(json!({ "role": "user", "content": turn.prompt }));
        add_exchanges(index, &mut messages);
        messages.push(json!({ "role": "assistant", "content": turn.reply }));
    }
    messages.push(json!({ "role": "user", "content": prompt }));
    messages
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_prompt_precedes_the_history_and_new_turn() {
        let history = [Turn {
            prompt: "First question".into(),
            reply: "First answer".into(),
        }];
        assert_eq!(
            chat_messages(Some("Use short answers.\n"), &history, "Next question"),
            vec![
                json!({"role": "system", "content": "Use short answers.\n"}),
                json!({"role": "user", "content": "First question"}),
                json!({"role": "assistant", "content": "First answer"}),
                json!({"role": "user", "content": "Next question"}),
            ]
        );
    }
}

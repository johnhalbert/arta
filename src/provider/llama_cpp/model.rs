mod template;

use super::{ChatConfig, LlamaCPPModelConfig};
use crate::{model::Model, provider::Turn};
use anyhow::{Context, Result, ensure};
use llama_cpp_4::prelude::{
    ChatParams, ChatTemplates, LlamaBackend, LlamaContext, LlamaContextParams, LlamaModel,
    LlamaToken,
};
use serde_json::{Value, json};
use std::{path::Path, sync::Arc};
use template::ModelTemplate;

pub(super) struct LlamaCPPModel {
    id: Arc<str>,
    template: ModelTemplate,
    pub(super) chat: Arc<ChatConfig>,
    pub(super) inner: LlamaModel,
    // Keep the backend alive until after the native model drops.
    backend: Arc<LlamaBackend>,
}

impl LlamaCPPModel {
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
            id,
            template,
            chat,
            inner,
            backend,
        })
    }

    pub(super) fn prepare_prompt(
        &self,
        system_prompt: Option<&str>,
        history: &[Turn],
        prompt: &str,
    ) -> Result<(Vec<LlamaToken>, Vec<String>, Option<ChatParams>)> {
        self.template
            .prepare_prompt(&self.inner, &self.chat, system_prompt, history, prompt)
    }

    pub(super) fn new_context(&self, params: LlamaContextParams) -> Result<LlamaContext<'_>> {
        self.inner
            .new_context(&self.backend, params)
            .context("failed to create inference context")
    }

    pub(super) fn tool_template(&self) -> Result<&ChatTemplates> {
        self.template.tool_template()
    }
}

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

impl Model for LlamaCPPModel {
    fn id(&self) -> &str {
        &self.id
    }

    fn trained_context_length(&self) -> u32 {
        self.inner.n_ctx_train()
    }
}

pub(super) fn chat_messages(
    system_prompt: Option<&str>,
    history: &[Turn],
    prompt: &str,
) -> Vec<Value> {
    let mut messages =
        Vec::with_capacity(history.len() * 2 + 1 + usize::from(system_prompt.is_some()));
    if let Some(content) = system_prompt {
        messages.push(json!({ "role": "system", "content": content }));
    }
    for turn in history {
        messages.push(json!({ "role": "user", "content": turn.prompt }));
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

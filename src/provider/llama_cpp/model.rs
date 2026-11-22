mod template;

use crate::{model::Model, provider::Turn};
use anyhow::{Context, Result, ensure};
use llama_cpp_4::prelude::{
    ChatTemplates, LlamaBackend, LlamaContext, LlamaContextParams, LlamaModel, LlamaModelParams,
    LlamaToken, supports_gpu_offload,
};
use serde_json::{Value, json};
use std::{num::NonZeroU32, path::Path, sync::Arc};
use template::ModelTemplate;

pub(super) struct LlamaCPPModel {
    id: Arc<str>,
    template: ModelTemplate,
    pub(super) inner: LlamaModel,
    // Keep the backend alive until after the native model drops.
    backend: Arc<LlamaBackend>,
}

impl LlamaCPPModel {
    pub(super) fn load(
        id: Arc<str>,
        backend: Arc<LlamaBackend>,
        path: &Path,
        gpu_layers: u32,
    ) -> Result<Self> {
        let metadata = path
            .metadata()
            .with_context(|| format!("cannot access model '{}'", path.display()))?;
        ensure!(
            metadata.is_file(),
            "model '{}' must be a file",
            path.display()
        );

        ensure!(
            gpu_layers == 0 || (cfg!(feature = "cuda") && supports_gpu_offload()),
            "GPU offload is unavailable. Build with --features cuda and use a CUDA device, or set --gpu-layers 0"
        );

        let parameters = LlamaModelParams::default().with_n_gpu_layers(gpu_layers);
        let inner = LlamaModel::load_from_file(&backend, path, &parameters)
            .with_context(|| format!("failed to load GGUF model '{}'", path.display()))?;
        ensure!(
            inner.has_decoder() && !inner.has_encoder() && !inner.is_diffusion(),
            "model '{}' does not support decoder-only text generation",
            path.display()
        );

        let template = ModelTemplate::from_model(&inner)?;

        Ok(Self {
            id,
            template,
            inner,
            backend,
        })
    }

    pub(super) fn prepare_prompt(
        &self,
        history: &[Turn],
        prompt: &str,
    ) -> Result<(Vec<LlamaToken>, Vec<String>)> {
        self.template.prepare_prompt(&self.inner, history, prompt)
    }

    pub(super) fn new_context(&self, context_length: NonZeroU32) -> Result<LlamaContext<'_>> {
        self.inner
            .new_context(
                &self.backend,
                LlamaContextParams::default().with_n_ctx(Some(context_length)),
            )
            .context("failed to create inference context")
    }

    pub(super) fn tool_template(&self) -> Result<&ChatTemplates> {
        self.template.tool_template()
    }
}

impl Model for LlamaCPPModel {
    fn id(&self) -> &str {
        &self.id
    }

    fn trained_context_length(&self) -> u32 {
        self.inner.n_ctx_train()
    }
}

pub(super) fn chat_messages(history: &[Turn], prompt: &str) -> Vec<Value> {
    let mut messages = Vec::with_capacity(history.len() * 2 + 1);
    for turn in history {
        messages.push(json!({ "role": "user", "content": turn.prompt }));
        messages.push(json!({ "role": "assistant", "content": turn.reply }));
    }
    messages.push(json!({ "role": "user", "content": prompt }));
    messages
}

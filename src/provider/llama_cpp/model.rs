use crate::{model::Model, provider::Turn};
use anyhow::{Context, Result, ensure};
use llama_cpp_4::{StringFromModelError, prelude::*};
use serde_json::{Value, json};
use std::{num::NonZeroU32, path::Path, sync::Arc};

pub(super) struct LlamaCPPModel {
    id: Arc<str>,
    templates: Option<ChatTemplates>,
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

        let templates = match inner.chat_template(None) {
            Ok(template) if !template.is_empty() => Some(
                ChatTemplates::from_model(&inner, None).context("invalid model chat template")?,
            ),
            Ok(_) | Err(StringFromModelError::ReturnedError(-1)) => None,
            Err(error) => return Err(error).context("failed to read model chat template"),
        };

        Ok(Self {
            id,
            templates,
            inner,
            backend,
        })
    }

    pub(super) fn prepare_prompt(
        &self,
        history: &[Turn],
        prompt: &str,
    ) -> Result<(Vec<LlamaToken>, Vec<String>)> {
        let add_bos = if self.inner.get_vocab().get_add_bos() {
            AddBos::Always
        } else {
            AddBos::Never
        };
        if let Some(templates) = &self.templates {
            let mut messages = Vec::with_capacity(history.len() * 2 + 1);
            for turn in history {
                messages.push(json!({ "role": "user", "content": turn.prompt }));
                messages.push(json!({ "role": "assistant", "content": turn.reply }));
            }
            messages.push(json!({ "role": "user", "content": prompt }));
            let messages = Value::Array(messages);
            let params = ChatApplyParams::new(messages.to_string()).with_enable_thinking(false);
            let chat = templates
                .apply(&params)
                .context("failed to format prompt")?;
            let tokens = chat
                .tokenize_prompt(&self.inner, add_bos)
                .context("failed to tokenize chat prompt")?;
            let stops = serde_json::from_str(&chat.additional_stops_json)
                .context("failed to read chat template stop strings")?;
            return Ok((tokens, stops));
        }

        let mut transcript;
        let prompt = if history.is_empty() {
            prompt
        } else {
            transcript = String::new();
            for turn in history {
                transcript.push_str("User: ");
                transcript.push_str(&turn.prompt);
                transcript.push_str("\nAssistant: ");
                transcript.push_str(&turn.reply);
                transcript.push('\n');
            }
            transcript.push_str("User: ");
            transcript.push_str(prompt);
            transcript.push_str("\nAssistant:");
            &transcript
        };
        let tokens = self
            .inner
            .str_to_token_with_special(prompt, add_bos, Special::Plaintext)
            .context("failed to tokenize prompt")?;
        Ok((tokens, Vec::new()))
    }

    pub(super) fn new_context(&self, context_length: NonZeroU32) -> Result<LlamaContext<'_>> {
        self.inner
            .new_context(
                &self.backend,
                LlamaContextParams::default().with_n_ctx(Some(context_length)),
            )
            .context("failed to create inference context")
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

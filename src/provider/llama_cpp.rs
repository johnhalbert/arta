use super::Provider;
use anyhow::{Context, Result, ensure};
use llama_cpp_4::StringFromModelError;
pub use llama_cpp_4::llama_backend::LlamaBackend;
use llama_cpp_4::prelude::*;
use serde_json::json;
use std::{num::NonZeroU32, path::PathBuf};

pub struct LlamaCPP<'backend> {
    templates: Option<ChatTemplates>,
    model: LlamaModel,
    backend: &'backend LlamaBackend,
    context_length: NonZeroU32,
    max_tokens: NonZeroU32,
}

pub struct LlamaCPPConfig {
    pub model_path: PathBuf,
    pub context_length: Option<NonZeroU32>,
    pub max_tokens: NonZeroU32,
    pub gpu_layers: u32,
}

impl<'backend> LlamaCPP<'backend> {
    /// Load a model using an initialized backend that can be shared by providers.
    pub fn new(backend: &'backend LlamaBackend, config: LlamaCPPConfig) -> Result<Self> {
        let path = &config.model_path;
        let metadata = path
            .metadata()
            .with_context(|| format!("cannot access model '{}'", path.display()))?;
        ensure!(
            metadata.is_file(),
            "model '{}' must be a file",
            path.display()
        );

        ensure!(
            config.gpu_layers == 0 || (cfg!(feature = "cuda") && supports_gpu_offload()),
            "GPU offload is unavailable. Build with --features cuda and use a CUDA device, or set --gpu-layers 0"
        );

        let parameters = LlamaModelParams::default().with_n_gpu_layers(config.gpu_layers);
        let model = LlamaModel::load_from_file(backend, path, &parameters)
            .with_context(|| format!("failed to load GGUF model '{}'", path.display()))?;
        ensure!(
            model.has_decoder() && !model.has_encoder() && !model.is_diffusion(),
            "model '{}' does not support decoder-only text generation",
            path.display()
        );
        let trained_context_length = model.n_ctx_train();
        let context_length = config
            .context_length
            .or_else(|| NonZeroU32::new(trained_context_length.min(4096)))
            .context("model has no usable context length")?;
        ensure!(
            context_length.get() <= trained_context_length,
            "requested context length {context_length} exceeds the model's trained context length of {trained_context_length} tokens"
        );
        ensure!(
            context_length.get() <= i32::MAX as u32,
            "context length exceeds llama.cpp's token position limit"
        );

        let templates = match model.chat_template(None) {
            Ok(template) if !template.is_empty() => Some(
                ChatTemplates::from_model(&model, None).context("invalid model chat template")?,
            ),
            Ok(_) | Err(StringFromModelError::ReturnedError(-1)) => None,
            Err(error) => return Err(error).context("failed to read model chat template"),
        };

        Ok(Self {
            model,
            templates,
            backend,
            context_length,
            max_tokens: config.max_tokens,
        })
    }

    fn prepare_prompt(&self, message: &str) -> Result<(Vec<LlamaToken>, Vec<String>)> {
        let add_bos = if self.model.get_vocab().get_add_bos() {
            AddBos::Always
        } else {
            AddBos::Never
        };
        if let Some(templates) = &self.templates {
            let messages = json!([{ "role": "user", "content": message }]);
            let params = ChatApplyParams::new(messages.to_string()).with_enable_thinking(false);
            let chat = templates
                .apply(&params)
                .context("failed to format prompt")?;
            let tokens = chat
                .tokenize_prompt(&self.model, add_bos)
                .context("failed to tokenize chat prompt")?;
            let mut stops: Vec<String> = serde_json::from_str(&chat.additional_stops_json)
                .context("failed to read chat template stop strings")?;
            stops.retain(|stop| !stop.is_empty());
            return Ok((tokens, stops));
        }

        let tokens = self
            .model
            .str_to_token_with_special(message, add_bos, Special::Plaintext)
            .context("failed to tokenize prompt")?;
        Ok((tokens, Vec::new()))
    }
}

impl Provider for LlamaCPP<'_> {
    fn send(&self, message: &str) -> Result<String> {
        ensure!(!message.contains('\0'), "prompt contains a NUL character");
        let (tokens, stops) = self.prepare_prompt(message)?;
        ensure!(!tokens.is_empty(), "prompt produced no tokens");
        let context_length = self.context_length.get() as usize;
        ensure!(
            tokens.len() < context_length,
            "prompt uses {} tokens and leaves no room for a response in the {context_length}-token context. Shorten the prompt or increase --context-length",
            tokens.len()
        );

        // Each prompt gets a fresh context while sharing the loaded model.
        let mut ctx = self
            .model
            .new_context(
                self.backend,
                LlamaContextParams::default().with_n_ctx(Some(self.context_length)),
            )
            .context("failed to create inference context")?;
        let capacity = context_length.min(ctx.n_ctx_seq() as usize);
        ensure!(
            tokens.len() < capacity,
            "prompt exceeds the usable inference context"
        );

        let batch_size = tokens.len().min(ctx.n_batch() as usize);
        ensure!(
            batch_size > 0,
            "inference context has an empty batch capacity"
        );
        let mut batch = LlamaBatch::new(batch_size, 1);
        let mut position = 0;
        for chunk in tokens.chunks(batch_size) {
            batch.clear();
            for (index, &token) in chunk.iter().enumerate() {
                batch.add(token, position, &[0], index + 1 == chunk.len())?;
                position += 1;
            }
            ctx.decode(&mut batch).context("failed to decode prompt")?;
        }

        let sampler = LlamaSampler::greedy();
        let mut output = Vec::new();
        let limit = (self.max_tokens.get() as usize).min(capacity - tokens.len());
        for generated in 1..=limit {
            let token = sampler
                .try_sample(&ctx, -1)
                .context("failed to sample response")?;
            if self.model.is_eog_token(token) {
                break;
            }
            let previous_length = output.len();
            output.extend(self.model.token_to_bytes(token, Special::Plaintext)?);
            if let Some(position) = stops
                .iter()
                .filter_map(|stop| {
                    // A stop string can span tokens or end inside the newest token.
                    let start = previous_length.saturating_sub(stop.len() - 1);
                    output[start..]
                        .windows(stop.len())
                        .position(|window| window == stop.as_bytes())
                        .map(|offset| start + offset)
                })
                .min()
            {
                output.truncate(position);
                break;
            }
            if generated == limit {
                break;
            }

            batch.clear();
            batch.add(token, position, &[0], true)?;
            ctx.decode(&mut batch)
                .context("failed to decode response")?;
            position += 1;
        }

        // A UTF-8 character can span multiple tokens.
        Ok(String::from_utf8_lossy(&output).into_owned())
    }
}

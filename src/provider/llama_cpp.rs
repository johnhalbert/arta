mod model;

use super::Provider;
use crate::model::Model;
use anyhow::{Context, Result, ensure};
use llama_cpp_4::prelude::{LlamaBackend, LlamaBatch, LlamaSampler, Special};
pub use model::LlamaCPPModel;
use std::{num::NonZeroU32, path::Path};

pub struct LlamaCPP {
    backend: LlamaBackend,
    context_length: Option<NonZeroU32>,
    max_tokens: NonZeroU32,
}

pub struct LlamaCPPConfig {
    pub context_length: Option<NonZeroU32>,
    pub max_tokens: NonZeroU32,
    pub verbose: bool,
}

impl LlamaCPP {
    /// Initialize the llama.cpp backend for this provider.
    pub fn new(config: LlamaCPPConfig) -> Result<Self> {
        let mut backend = LlamaBackend::init().context("failed to initialize llama.cpp")?;
        if !config.verbose {
            backend.void_logs();
        }
        Ok(Self {
            backend,
            context_length: config.context_length,
            max_tokens: config.max_tokens,
        })
    }

    /// Load a GGUF model that this provider can use for inference.
    pub fn load_model(&self, path: impl AsRef<Path>, gpu_layers: u32) -> Result<LlamaCPPModel<'_>> {
        LlamaCPPModel::load(&self.backend, path.as_ref(), gpu_layers)
    }
}

impl Provider for LlamaCPP {
    type Model<'provider> = LlamaCPPModel<'provider>;

    fn send<'provider>(
        &'provider self,
        model: &Self::Model<'provider>,
        prompt: &str,
    ) -> Result<String> {
        let trained_context_length = model.trained_context_length();
        let context_length = self
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

        ensure!(!prompt.contains('\0'), "prompt contains a NUL character");
        let (tokens, stops) = model.prepare_prompt(prompt)?;
        ensure!(!tokens.is_empty(), "prompt produced no tokens");
        let context_capacity = context_length.get() as usize;
        ensure!(
            tokens.len() < context_capacity,
            "prompt uses {} tokens and leaves no room for a response in the {context_length}-token context. Shorten the prompt or increase --context-length",
            tokens.len()
        );

        // Each prompt gets a fresh context while sharing the loaded model.
        let mut ctx = model.new_context(context_length)?;
        let capacity = context_capacity.min(ctx.n_ctx_seq() as usize);
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
            if model.inner.is_eog_token(token) {
                break;
            }
            let previous_length = output.len();
            output.extend(model.inner.token_to_bytes(token, Special::Plaintext)?);
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

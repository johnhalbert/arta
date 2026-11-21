mod config;
mod model;
mod text;

use super::{Provider, registry::ProviderRegistration};
use crate::{config::Config, model::Model, stream::Message};
use anyhow::{Context, Result, bail, ensure};
pub use config::{LlamaCPPConfig, LlamaCPPModelConfig, LlamaCPPOverrides};
use futures_util::{future::BoxFuture, stream};
use llama_cpp_4::prelude::{LlamaBackend, LlamaBatch, LlamaSampler, Special};
use model::LlamaCPPModel;
use std::{
    collections::{HashMap, hash_map::Entry},
    num::NonZeroU32,
    sync::Arc,
};
use tokio::{
    runtime::Handle,
    sync::{mpsc, oneshot},
};

pub struct LlamaCPP {
    models: HashMap<Arc<str>, Arc<LlamaCPPModel>>,
    context_length: Option<NonZeroU32>,
    max_tokens: NonZeroU32,
}

inventory::submit! {
    ProviderRegistration { id: LlamaCPP::ID, create: LlamaCPP::from_config }
}

impl LlamaCPP {
    pub const ID: &str = "llama.cpp";

    fn from_config(config: &Config) -> Result<Arc<dyn Provider>> {
        let mut settings: LlamaCPPConfig = config.provider_settings(Self::ID)?;
        for model in &mut settings.models {
            model.path = config.resolve_path(&model.path);
        }
        Ok(Arc::new(Self::new(settings)?))
    }

    /// Initialize the backend and load the configured models.
    pub fn new(config: LlamaCPPConfig) -> Result<Self> {
        let mut provider = Self {
            models: HashMap::with_capacity(config.models.len()),
            context_length: config.context_length,
            max_tokens: config.max_tokens,
        };
        if config.models.is_empty() {
            return Ok(provider);
        }
        let mut backend = LlamaBackend::init().context("failed to initialize llama.cpp")?;
        if !config.verbose {
            backend.void_logs();
        }
        let backend = Arc::new(backend);
        for model in config.models {
            ensure!(
                !model.id.trim().is_empty(),
                "model identifier cannot be empty"
            );
            let id: Arc<str> = model.id.into();
            match provider.models.entry(Arc::clone(&id)) {
                Entry::Vacant(entry) => {
                    entry.insert(Arc::new(LlamaCPPModel::load(
                        id,
                        Arc::clone(&backend),
                        &model.path,
                        model.gpu_layers,
                    )?));
                }
                Entry::Occupied(entry) => bail!("model '{}' is already registered", entry.key()),
            }
        }

        Ok(provider)
    }

    fn loaded_model(&self, model_id: &str) -> Result<&Arc<LlamaCPPModel>> {
        self.models
            .get(model_id)
            .with_context(|| format!("unknown llama.cpp model '{model_id}'"))
    }
}

impl Provider for LlamaCPP {
    fn model(&self, model_id: &str) -> Result<Arc<dyn Model>> {
        Ok(self.loaded_model(model_id)?.clone())
    }

    fn send<'a>(&'a self, model_id: &'a str, prompt: &'a str) -> BoxFuture<'a, Result<Message>> {
        Box::pin(async move {
            let model = Arc::clone(self.loaded_model(model_id)?);
            let prompt = prompt.to_owned();
            let context_length = self.context_length;
            let max_tokens = self.max_tokens;
            let runtime =
                Handle::try_current().context("llama.cpp inference requires a Tokio runtime")?;
            let (ready, prepared) = oneshot::channel();
            let (sender, receiver) = mpsc::channel(1);
            let worker = runtime.spawn_blocking(move || -> Result<()> {
                if sender.is_closed() {
                    return Ok(());
                }
                let mut chunks = generate(&model, &prompt, context_length, max_tokens, || {
                    sender.is_closed()
                })?;
                if ready.send(()).is_err() {
                    return Ok(());
                }
                while !sender.is_closed() {
                    let Some(chunk) = chunks.next() else { break };
                    if sender.blocking_send(chunk?).is_err() {
                        break;
                    }
                }
                Ok(())
            });

            if prepared.await.is_err() {
                worker
                    .await
                    .context("llama.cpp inference worker failed")??;
                bail!("llama.cpp inference worker stopped before preparing the response");
            }

            let chunks = stream::try_unfold((receiver, worker), |(mut receiver, worker)| async {
                if let Some(chunk) = receiver.recv().await {
                    Ok(Some((chunk, (receiver, worker))))
                } else {
                    worker
                        .await
                        .context("llama.cpp inference worker failed")??;
                    Ok(None)
                }
            });
            Ok(Message::new(chunks))
        })
    }
}

fn generate(
    model: &LlamaCPPModel,
    prompt: &str,
    context_length: Option<NonZeroU32>,
    max_tokens: NonZeroU32,
    cancelled: impl Fn() -> bool,
) -> Result<impl Iterator<Item = Result<String>>> {
    let trained_context_length = model.trained_context_length();
    let context_length = context_length
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
        ensure!(!cancelled(), "llama.cpp inference cancelled");
        batch.clear();
        for (index, &token) in chunk.iter().enumerate() {
            batch.add(token, position, &[0], index + 1 == chunk.len())?;
            position += 1;
        }
        ctx.decode(&mut batch).context("failed to decode prompt")?;
    }

    let sampler = LlamaSampler::greedy();
    let limit = (max_tokens.get() as usize).min(capacity - tokens.len());
    let mut previous_token = None;
    let pieces = (0..limit)
        .map(move |_| -> Result<Option<Vec<u8>>> {
            if cancelled() {
                return Ok(None);
            }
            if let Some(token) = previous_token {
                batch.clear();
                batch.add(token, position, &[0], true)?;
                ctx.decode(&mut batch)
                    .context("failed to decode response")?;
                position += 1;
            }

            let token = sampler
                .try_sample(&ctx, -1)
                .context("failed to sample response")?;
            if model.inner.is_eog_token(token) {
                return Ok(None);
            }
            previous_token = Some(token);
            Ok(Some(model.inner.token_to_bytes(token, Special::Plaintext)?))
        })
        .map_while(Result::transpose);

    Ok(text::text_chunks(pieces, stops))
}

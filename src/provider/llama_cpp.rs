mod config;
mod logging;
mod model;
mod sampling;
mod text;
mod tool;

use super::{PromptObserver, Provider, Turn, registry::ProviderRegistration, tool::ToolAdapter};
use crate::{config::Config, model::Model, stream::Message};
use anyhow::{Context, Result, bail, ensure};
pub use config::{
    AttentionType, CacheType, ChatConfig, ChatReasoningFormat, ChatToolChoice, ContextConfig,
    FlashAttention, LlamaCPPConfig, LlamaCPPModelConfig, LlamaCPPOverrides, MetadataOverride,
    MetadataValue, ModelLazyMode, ModelLoadMode, ModelSplitMode, NumaConfig, RopeScaling,
    TensorBufferOverride,
};
use futures_util::{future::BoxFuture, stream};
use llama_cpp_4::prelude::{ChatParams, LlamaBackend, LlamaBatch, LlamaToken, Special};
use model::LlamaCPPModel;
use sampling::Sampler;
pub use sampling::{LogitBias, SamplerStage, SamplingConfig};
use std::{
    cell::Cell,
    collections::{HashMap, hash_map::Entry},
    num::NonZeroU32,
    sync::{Arc, Mutex},
};
use tokio::{
    runtime::Handle,
    sync::{mpsc, oneshot},
};

pub struct LlamaCPP {
    models: HashMap<Arc<str>, LlamaCPPModelConfig>,
    loaded: Mutex<HashMap<Arc<str>, Arc<LlamaCPPModel>>>,
    backend: Arc<Mutex<Option<Arc<LlamaBackend>>>>,
    chat: Arc<ChatConfig>,
    inference: Arc<InferenceConfig>,
    verbose: bool,
}

inventory::submit! {
    ProviderRegistration { id: LlamaCPP::ID, create: LlamaCPP::from_config }
}

impl LlamaCPP {
    pub const ID: &str = "llama.cpp";

    fn from_config(config: &Config) -> Result<Arc<dyn Provider>> {
        let mut settings: LlamaCPPConfig = config.provider_settings(Self::ID)?;
        settings.resolve_paths(config);
        Ok(Arc::new(Self::new(settings)?))
    }

    /// Initialize the backend and register models for on-demand loading.
    pub fn new(config: LlamaCPPConfig) -> Result<Self> {
        Self::with_backend(config, Arc::default())
    }

    fn with_backend(
        mut config: LlamaCPPConfig,
        backend: Arc<Mutex<Option<Arc<LlamaBackend>>>>,
    ) -> Result<Self> {
        config.context.validate()?;
        if let Some(sampling) = &config.sampling {
            sampling.validate()?;
        }
        config.chat.load_template()?;
        {
            let mut backend = backend
                .lock()
                .map_err(|_| anyhow::anyhow!("llama.cpp backend is unavailable"))?;
            if backend.is_none() {
                *backend = config.init_backend()?;
            }
        }
        let mut models = HashMap::with_capacity(config.models.len());
        for model in config.models {
            ensure!(
                !model.id.trim().is_empty(),
                "model identifier cannot be empty"
            );
            match models.entry(Arc::from(model.id.as_str())) {
                Entry::Vacant(entry) => {
                    entry.insert(model);
                }
                Entry::Occupied(entry) => bail!("model '{}' is already registered", entry.key()),
            }
        }

        Ok(Self {
            models,
            loaded: Mutex::default(),
            backend,
            verbose: config.verbose,
            chat: Arc::new(config.chat),
            inference: Arc::new(InferenceConfig {
                context_length: config.context_length,
                max_tokens: config.max_tokens,
                context: config.context,
                sampling: config.sampling,
            }),
        })
    }

    fn load_model(&self, model_id: &str, previous: Option<&str>) -> Result<Arc<LlamaCPPModel>> {
        let (id, config) = self
            .models
            .get_key_value(model_id)
            .with_context(|| format!("unknown llama.cpp model '{model_id}'"))?;
        // Keep inference from acquiring the old model between unloading and loading.
        let mut loaded = self
            .loaded
            .lock()
            .map_err(|_| anyhow::anyhow!("llama.cpp model cache is unavailable"))?;
        let _logs = logging::Scope::new(self.verbose);
        if let Some(previous) = previous.filter(|&previous| previous != model_id) {
            Self::remove_model(&mut loaded, previous)?;
        }
        match loaded.entry(id.clone()) {
            Entry::Occupied(entry) => Ok(entry.get().clone()),
            Entry::Vacant(entry) => {
                let backend = self
                    .backend
                    .lock()
                    .map_err(|_| anyhow::anyhow!("llama.cpp backend is unavailable"))?
                    .clone()
                    .context("llama.cpp backend is unavailable")?;
                let model = Arc::new(LlamaCPPModel::load(
                    id.clone(),
                    backend,
                    config,
                    self.chat.clone(),
                )?);
                entry.insert(model.clone());
                Ok(model)
            }
        }
    }

    fn remove_model(loaded: &mut HashMap<Arc<str>, Arc<LlamaCPPModel>>, id: &str) -> Result<()> {
        if let Some(model) = loaded.get(id) {
            ensure!(
                Arc::strong_count(model) == 1,
                "model '{id}' is still in use"
            );
            drop(loaded.remove(id));
        }
        Ok(())
    }
}

impl Provider for LlamaCPP {
    fn model(&self, model_id: &str) -> Result<Arc<dyn Model>> {
        Ok(self.load_model(model_id, None)?.info.clone())
    }

    fn switch_model(&self, current_id: &str, model_id: &str) -> Result<Arc<dyn Model>> {
        Ok(self.load_model(model_id, Some(current_id))?.info.clone())
    }

    fn unload_model(&self, model_id: &str) -> Result<()> {
        let _logs = logging::Scope::new(self.verbose);
        let mut loaded = self
            .loaded
            .lock()
            .map_err(|_| anyhow::anyhow!("llama.cpp model cache is unavailable"))?;
        Self::remove_model(&mut loaded, model_id)
    }

    fn configured(&self, config: &Config) -> Result<Arc<dyn Provider>> {
        let mut settings: LlamaCPPConfig = config.provider_settings(Self::ID)?;
        settings.resolve_paths(config);
        Ok(Arc::new(Self::with_backend(
            settings,
            self.backend.clone(),
        )?))
    }

    fn send<'a>(&'a self, model_id: &'a str, prompt: &'a str) -> BoxFuture<'a, Result<Message>> {
        self.chat(model_id, &[], prompt)
    }

    fn tool_adapter(&self) -> Option<&dyn ToolAdapter> {
        Some(self)
    }

    fn chat<'a>(
        &'a self,
        model_id: &'a str,
        history: &'a [Turn],
        prompt: &'a str,
    ) -> BoxFuture<'a, Result<Message>> {
        self.chat_with_system(model_id, None, history, prompt)
    }

    fn chat_with_system<'a>(
        &'a self,
        model_id: &'a str,
        system_prompt: Option<&'a str>,
        history: &'a [Turn],
        prompt: &'a str,
    ) -> BoxFuture<'a, Result<Message>> {
        self.chat_observed(model_id, system_prompt, history, prompt, None)
    }

    fn chat_observed<'a>(
        &'a self,
        model_id: &'a str,
        system_prompt: Option<&'a str>,
        history: &'a [Turn],
        prompt: &'a str,
        observer: Option<PromptObserver>,
    ) -> BoxFuture<'a, Result<Message>> {
        Box::pin(async move {
            let model = self.load_model(model_id, None)?;
            let history = history.to_vec();
            let prompt = prompt.to_owned();
            let system_prompt = system_prompt.map(str::to_owned);
            let inference = Arc::clone(&self.inference);
            let verbose = self.verbose;
            let runtime =
                Handle::try_current().context("llama.cpp inference requires a Tokio runtime")?;
            let (ready, prepared) = oneshot::channel();
            let (sender, receiver) = mpsc::channel(1);
            let worker = runtime.spawn_blocking(move || -> Result<()> {
                let _logs = logging::Scope::new(verbose);
                if sender.is_closed() {
                    return Ok(());
                }
                ensure!(!prompt.contains('\0'), "prompt contains a NUL character");
                let (tokens, stops, chat) = model.prepare_prompt(
                    system_prompt.as_deref(),
                    &history,
                    &prompt,
                    observer.as_ref(),
                )?;
                let pieces = generate(
                    &model,
                    tokens,
                    &inference,
                    Sampling::Text(chat.as_ref()),
                    || sender.is_closed(),
                )?;
                if ready.send(()).is_err() {
                    return Ok(());
                }
                let mut chunks = text::text_chunks(pieces, stops);
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

enum Sampling<'a> {
    Text(Option<&'a ChatParams>),
    Tools {
        chat: &'a ChatParams,
        exhausted: &'a Cell<bool>,
    },
}

struct InferenceConfig {
    context_length: Option<u32>,
    max_tokens: NonZeroU32,
    context: ContextConfig,
    sampling: Option<SamplingConfig>,
}

fn generate<'a>(
    model: &'a LlamaCPPModel,
    tokens: Vec<LlamaToken>,
    config: &InferenceConfig,
    sampling: Sampling<'a>,
    cancelled: impl Fn() -> bool + 'a,
) -> Result<impl Iterator<Item = Result<Vec<u8>>> + 'a> {
    let trained_context_length = model.info.trained_context_length();
    let context_length = match config.context_length {
        None => trained_context_length.min(4096),
        Some(0) => trained_context_length,
        Some(length) => length,
    };
    let context_length =
        NonZeroU32::new(context_length).context("model has no usable context length")?;
    ensure!(
        context_length.get() <= i32::MAX as u32,
        "context length exceeds llama.cpp's token position limit"
    );

    ensure!(!tokens.is_empty(), "prompt produced no tokens");
    let context_capacity = context_length.get() as usize;
    ensure!(
        tokens.len() < context_capacity,
        "prompt uses {} tokens and leaves no room for a response in the {context_length}-token context. Shorten the prompt or increase --context-length",
        tokens.len()
    );

    // Each prompt gets a fresh context while sharing the loaded model.
    let mut ctx = model.new_context(config.context.params(context_length))?;
    let capacity = context_capacity.min(ctx.n_ctx_seq() as usize);
    ensure!(
        tokens.len() < capacity,
        "prompt exceeds the usable inference context"
    );

    let batch_size = config.context.prompt_batch_size(&ctx, tokens.len());
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

    let mut sampler = Sampler::new(&model.inner, config.sampling.as_ref(), &sampling, &tokens)?;
    let limit = (config.max_tokens.get() as usize).min(capacity - tokens.len());
    let mut previous_token = None;
    Ok((0..limit)
        .map(move |index| -> Result<Option<Vec<u8>>> {
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
                .sample(&mut ctx)
                .context("failed to sample response")?;
            if model.inner.is_eog_token(token) {
                return Ok(None);
            }
            previous_token = Some(token);
            let bytes = match sampling {
                Sampling::Text(_) => model.inner.token_to_bytes(token, Special::Plaintext)?,
                Sampling::Tools { exhausted, .. } => {
                    exhausted.set(index + 1 == limit);
                    model.inner.token_to_raw_bytes(token, Special::Tokenize)?
                }
            };
            Ok(Some(bytes))
        })
        .map_while(Result::transpose))
}

//! Provider-wide settings plus the narrower set of overrides accepted from a profile or the
//! CLI. Model placement is applied at weight load. Chat and inference settings are reused for
//! every step on this provider.

mod chat;
mod context;
mod model;

pub use chat::{ChatConfig, ChatReasoningFormat, ChatToolChoice};
pub use context::{AttentionType, CacheType, ContextConfig, FlashAttention, RopeScaling};
pub use model::{
    LlamaCPPModelConfig, MetadataOverride, MetadataValue, ModelLazyMode, ModelLoadMode,
    ModelSplitMode, TensorBufferOverride,
};

use super::sampling::SamplingConfig;
use super::{LlamaCPP, logging};
use crate::config::Config;
use anyhow::{Context, Result, ensure};
use llama_cpp_4::llama_backend::{LlamaBackend, NumaStrategy};
use serde::{Deserialize, Serialize};
use std::{
    num::NonZeroU32,
    path::{self, PathBuf},
    sync::Arc,
};

/// Deserialized llama.cpp table. Each configured model shares the context, chat, and sampling
/// settings.
#[derive(Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct LlamaCPPConfig {
    /// Total context in tokens. Omission caps the trained length at 4096. Zero requests the
    /// full trained length.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    /// Maximum tokens generated per inference step, also bounded by free context. Defaults to
    /// 256.
    pub max_tokens: NonZeroU32,
    /// Enable native diagnostics in scoped model operations.
    pub verbose: bool,
    /// NUMA policy applied when the shared native backend is initialized.
    pub numa: NumaConfig,
    /// Parameters applied when creating a fresh native inference context.
    pub context: ContextConfig,
    /// Template selection, rendering options, and extra stop strings.
    pub chat: ChatConfig,
    /// Omission selects greedy sampling. A table enables the native common sampler.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling: Option<SamplingConfig>,
    /// Uniquely named GGUF model entries available for on-demand loading.
    pub models: Vec<LlamaCPPModelConfig>,
}

impl Default for LlamaCPPConfig {
    fn default() -> Self {
        Self {
            context_length: None,
            max_tokens: NonZeroU32::new(256).unwrap(),
            verbose: false,
            numa: NumaConfig::default(),
            context: ContextConfig::default(),
            chat: ChatConfig::default(),
            sampling: None,
            models: Vec::new(),
        }
    }
}

impl LlamaCPPConfig {
    /// Resolve model, shard and template files against the base TOML directory.
    pub(super) fn resolve_paths(&mut self, config: &Config) {
        self.chat.resolve_paths(config);
        for model in &mut self.models {
            model.resolve_paths(config);
        }
    }

    /// Skip initialization for an empty model registry. Otherwise initialize the backend,
    /// install diagnostics, and apply the selected NUMA strategy.
    pub(super) fn init_backend(&self) -> Result<Option<Arc<LlamaBackend>>> {
        if self.models.is_empty() {
            return Ok(None);
        }
        let backend = LlamaBackend::init().context("failed to initialize llama.cpp")?;
        logging::install();
        let _logs = logging::Scope::new(self.verbose);
        if let Some(strategy) = match self.numa {
            NumaConfig::Disabled => None,
            NumaConfig::Distribute => Some(NumaStrategy::DISTRIBUTE),
            NumaConfig::Isolate => Some(NumaStrategy::ISOLATE),
            NumaConfig::Numactl => Some(NumaStrategy::NUMACTL),
            NumaConfig::Mirror => Some(NumaStrategy::MIRROR),
        } {
            // The dynamic CPU backend must be loaded before initializing NUMA.
            // The backend guard owns initialization and the strategy is a native enum value.
            unsafe { llama_cpp_sys_4::llama_numa_init(strategy.into()) };
        }
        Ok(Some(Arc::new(backend)))
    }
}

/// Native NUMA initialization strategy, selected once when constructing the shared backend.
#[derive(Clone, Copy, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NumaConfig {
    /// Skip explicit NUMA initialization.
    #[default]
    Disabled,
    /// Request the native policy that distributes work across NUMA nodes.
    Distribute,
    /// Request isolation to the current NUMA node.
    Isolate,
    /// Use the affinity supplied by numactl.
    Numactl,
    /// Request the native mirrored NUMA policy.
    Mirror,
}

/// Optional overrides for the selected llama.cpp model and its inference settings.
#[derive(Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LlamaCPPOverrides {
    /// Replacement base GGUF file for the selected model. Replacing it clears an old shard
    /// list.
    pub model_path: Option<PathBuf>,
    /// Positive context limit supplied by this override layer.
    pub context_length: Option<NonZeroU32>,
    /// Replacement response budget per inference step.
    pub max_tokens: Option<NonZeroU32>,
    /// Replacement number of layers to offload for the selected model.
    pub gpu_layers: Option<u32>,
    /// Optional replacement for the provider's diagnostic flag.
    pub verbose: Option<bool>,
}

impl LlamaCPPOverrides {
    /// Leave absent overrides alone. For llama.cpp, update provider-wide limits and the
    /// selected model entry. Creating a missing entry requires a path. Serialize back into
    /// the in-memory config after successful validation.
    pub fn apply(self, config: &mut Config) -> Result<()> {
        if self.model_path.is_none()
            && self.context_length.is_none()
            && self.max_tokens.is_none()
            && self.gpu_layers.is_none()
            && self.verbose.is_none()
        {
            return Ok(());
        }

        ensure!(
            config.app.provider == LlamaCPP::ID,
            "llama.cpp overrides require the '{}' provider",
            LlamaCPP::ID
        );
        let mut settings: LlamaCPPConfig = config.provider_settings(LlamaCPP::ID)?;
        settings.context_length = self
            .context_length
            .map(NonZeroU32::get)
            .or(settings.context_length);
        settings.max_tokens = self.max_tokens.unwrap_or(settings.max_tokens);
        settings.verbose = self.verbose.unwrap_or(settings.verbose);

        if self.model_path.is_some() || self.gpu_layers.is_some() {
            let path = self
                .model_path
                .map(path::absolute)
                .transpose()
                .context("failed to resolve --model path")?;
            let id = &config.app.model;
            match settings.models.iter_mut().find(|model| &model.id == id) {
                Some(model) => {
                    if let Some(path) = path {
                        model.path = path;
                        model.additional_shards.clear();
                    }
                    model.gpu_layers = self.gpu_layers.unwrap_or(model.gpu_layers);
                }
                None => settings.models.push(LlamaCPPModelConfig {
                    id: id.clone(),
                    path: path.with_context(|| {
                        format!("model '{id}' needs a config entry or --model PATH")
                    })?,
                    gpu_layers: self.gpu_layers.unwrap_or_default(),
                    ..Default::default()
                }),
            }
        }

        config.set_provider_settings(LlamaCPP::ID, settings)
    }
}

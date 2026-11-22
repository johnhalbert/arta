use super::LlamaCPP;
use crate::config::Config;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    num::NonZeroU32,
    path::{self, PathBuf},
};

#[derive(Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct LlamaCPPConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<NonZeroU32>,
    pub max_tokens: NonZeroU32,
    pub verbose: bool,
    pub models: Vec<LlamaCPPModelConfig>,
}

impl Default for LlamaCPPConfig {
    fn default() -> Self {
        Self {
            context_length: None,
            max_tokens: NonZeroU32::new(256).unwrap(),
            verbose: false,
            models: Vec::new(),
        }
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LlamaCPPModelConfig {
    pub id: String,
    pub path: PathBuf,
    #[serde(default)]
    pub gpu_layers: u32,
}

/// Optional overrides for the selected llama.cpp model and its inference settings.
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LlamaCPPOverrides {
    pub model_path: Option<PathBuf>,
    pub context_length: Option<NonZeroU32>,
    pub max_tokens: Option<NonZeroU32>,
    pub gpu_layers: Option<u32>,
    pub verbose: Option<bool>,
}

impl LlamaCPPOverrides {
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
        settings.context_length = self.context_length.or(settings.context_length);
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
                    }
                    model.gpu_layers = self.gpu_layers.unwrap_or(model.gpu_layers);
                }
                None => settings.models.push(LlamaCPPModelConfig {
                    id: id.clone(),
                    path: path.with_context(|| {
                        format!("model '{id}' needs a config entry or --model PATH")
                    })?,
                    gpu_layers: self.gpu_layers.unwrap_or_default(),
                }),
            }
        }

        config.set_provider_settings(LlamaCPP::ID, settings)
    }
}

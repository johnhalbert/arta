//! Provider-wide runtime settings and model registrations.

use super::{CactusCompute, needle::NeedleConfig};
use crate::config::Config;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{num::NonZeroU32, path::PathBuf};

/// Settings for the native worker shared by Cactus model implementations.
#[derive(Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct RuntimeConfig {
    /// Path to an Arta executable to run in worker mode. Defaults to the running CLI.
    pub worker_path: Option<PathBuf>,
    /// Optional local Cactus shared library. Otherwise the worker resolves the pinned engine.
    pub library_path: Option<PathBuf>,
    /// Whether to use cached or local files only.
    pub offline: bool,
    #[serde(rename = "python", skip_serializing)]
    _legacy_python: Option<PathBuf>,
    #[serde(rename = "telemetry", skip_serializing)]
    _legacy_telemetry: Option<bool>,
}

/// One model registered with the Cactus provider.
#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CactusModelConfig {
    /// A Needle 3 instance with its own weights and inference settings.
    Needle(NeedleConfig),
}

/// TOML settings under `[providers.cactus]`.
#[derive(Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct CactusConfig {
    /// Settings common to all Cactus models.
    pub runtime: RuntimeConfig,
    /// Model entries keyed by their Arta IDs.
    pub models: Vec<CactusModelConfig>,
}

impl Default for CactusConfig {
    fn default() -> Self {
        Self {
            runtime: RuntimeConfig::default(),
            models: vec![CactusModelConfig::Needle(NeedleConfig::default())],
        }
    }
}

impl CactusConfig {
    pub(super) fn resolve_paths(&mut self, config: &Config) -> Result<()> {
        for (key, path) in [
            ("worker_path", &mut self.runtime.worker_path),
            ("library_path", &mut self.runtime.library_path),
        ] {
            if let Some(path) = path {
                ensure!(!path.as_os_str().is_empty(), "Cactus {key} cannot be empty");
                *path = config.resolve_path(path);
            }
        }
        for model in &mut self.models {
            match model {
                CactusModelConfig::Needle(needle) => needle.resolve_paths(config)?,
            }
        }
        Ok(())
    }

    /// Apply the shared CLI output budget to the selected Cactus model.
    pub(crate) fn override_max_tokens(config: &mut Config, max_tokens: NonZeroU32) -> Result<()> {
        let mut settings: Self = config.provider_settings(CactusCompute::ID)?;
        let model_id = &config.app.model;
        let needle = settings
            .models
            .iter_mut()
            .find_map(|model| match model {
                CactusModelConfig::Needle(needle) if &needle.id == model_id => Some(needle),
                _ => None,
            })
            .with_context(|| format!("unknown Cactus model '{model_id}'"))?;
        needle.max_tokens = max_tokens;
        config.set_provider_settings(CactusCompute::ID, settings)
    }
}

//! Needle weights, tool schemas, and inference options.

use crate::{config::Config, provider::tool::MAX_CALLS_PER_PROMPT};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, num::NonZeroU32, path::PathBuf};

/// Settings for one Needle 3 model entry under `[[providers.cactus.models]]`.
#[derive(Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct NeedleConfig {
    /// Identifier used for model selection in Arta.
    pub id: String,
    /// Local `.cact` archive, including a custom or sliced Needle 3 model.
    pub weights: Option<PathBuf>,
    /// Runtime index archive path accepted by the SDK.
    pub tool_index_path: Option<PathBuf>,
    /// Native response buffer size in bytes.
    pub buffer_size: NonZeroU32,
    /// Native CPU thread count. Zero uses the SDK's automatic setting.
    pub threads: u32,
    /// Maximum generated tokens per inference step.
    pub max_tokens: NonZeroU32,
    /// Maximum tool call batches within the Arta tool-call budget.
    pub max_steps: NonZeroU32,
    /// System facts supplied alongside any agent system prompt.
    pub system: Option<String>,
    /// Insert the local date and time into the system facts.
    pub auto_date: bool,
    /// Treat each user prompt as independent while keeping its own tool exchanges.
    pub stateless: bool,
    /// Reject tool batches with engine-reported ungrounded or negated arguments.
    pub strict: bool,
    /// Optional minimum confidence for accepting a result.
    pub min_confidence: Option<f64>,
    /// Regex triggers keyed by exact Arta tool name.
    pub triggers: BTreeMap<String, Vec<String>>,
    /// Optional JSON file containing a Needle extraction schema for prompts without tools.
    pub extraction_schema: Option<PathBuf>,
}

impl Default for NeedleConfig {
    fn default() -> Self {
        Self {
            id: "needle".into(),
            weights: None,
            tool_index_path: None,
            buffer_size: NonZeroU32::new(65_536).unwrap(),
            threads: 0,
            max_tokens: NonZeroU32::new(512).unwrap(),
            max_steps: NonZeroU32::new(MAX_CALLS_PER_PROMPT).unwrap(),
            system: None,
            auto_date: true,
            stateless: false,
            strict: true,
            min_confidence: None,
            triggers: BTreeMap::new(),
            extraction_schema: None,
        }
    }
}

impl NeedleConfig {
    pub(in crate::provider::cactus) fn resolve_paths(&mut self, config: &Config) -> Result<()> {
        for (key, path) in [
            ("weights", &mut self.weights),
            ("tool_index_path", &mut self.tool_index_path),
            ("extraction_schema", &mut self.extraction_schema),
        ] {
            if let Some(path) = path {
                ensure!(!path.as_os_str().is_empty(), "Needle {key} cannot be empty");
                *path = config.resolve_path(path);
            }
        }
        Ok(())
    }

    pub(super) fn validate(&self) -> Result<()> {
        ensure!(
            !self.id.trim().is_empty(),
            "model identifier cannot be empty"
        );
        ensure!(
            self.max_steps.get() <= MAX_CALLS_PER_PROMPT,
            "Needle max_steps must be 1 through {MAX_CALLS_PER_PROMPT}"
        );
        for (key, value) in [
            ("threads", self.threads),
            ("buffer_size", self.buffer_size.get()),
            ("max_tokens", self.max_tokens.get()),
        ] {
            ensure!(
                i32::try_from(value).is_ok(),
                "Needle {key} exceeds the runtime limit"
            );
        }
        if let Some(confidence) = self.min_confidence {
            ensure!(
                (0.0..=1.0).contains(&confidence),
                "Needle min_confidence must be between 0 and 1"
            );
        }
        Ok(())
    }
}

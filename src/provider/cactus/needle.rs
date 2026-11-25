//! Needle 3 model metadata and structured extraction.

mod config;
mod inference;
pub(super) mod worker;

pub use config::NeedleConfig;
use inference::Inference;
pub(super) use inference::NeedleTurn;

use super::RuntimeConfig;
use crate::{model::Model, provider::PromptObserver};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{fs::File, io::Read, sync::Arc};

pub(super) struct Needle {
    settings: NeedleConfig,
}

struct NeedleModel {
    id: String,
    trained_context_length: u32,
}

impl Model for NeedleModel {
    fn id(&self) -> &str {
        &self.id
    }

    fn trained_context_length(&self) -> u32 {
        self.trained_context_length
    }
}

impl Needle {
    pub(super) fn new(settings: NeedleConfig) -> Result<Self> {
        settings.validate()?;
        Ok(Self { settings })
    }

    pub(super) fn model(&self) -> Result<Arc<dyn Model>> {
        let trained_context_length = match self.settings.weights.as_deref() {
            None => 8192, // max_seq_len in the published Needle 3 archive
            Some(path) => {
                // The little-endian Needle 3 .cact header stores max_seq_len at word 13.
                let mut header = [0; 56];
                File::open(path)
                    .and_then(|mut file| file.read_exact(&mut header))
                    .with_context(|| {
                        format!("failed to read Needle weights '{}'", path.display())
                    })?;
                ensure!(
                    header[..4] == 0x05E12A84_u32.to_le_bytes(),
                    "'{}' is not a Needle 3 .cact archive",
                    path.display()
                );
                let length = u32::from_le_bytes(header[52..56].try_into().unwrap());
                ensure!(length > 0, "Needle weights have a zero context length");
                length
            }
        };
        Ok(Arc::new(NeedleModel {
            id: self.settings.id.clone(),
            trained_context_length,
        }))
    }

    pub(super) fn turn(self: &Arc<Self>, runtime: Arc<RuntimeConfig>) -> NeedleTurn {
        NeedleTurn::new(self.clone(), runtime)
    }

    pub(super) async fn extract(
        &self,
        runtime: &RuntimeConfig,
        system: Option<&str>,
        text: &str,
        observer: Option<&PromptObserver>,
    ) -> Result<String> {
        let path = self
            .settings
            .extraction_schema
            .as_deref()
            .context("Needle requires a visible tool or an extraction_schema for this model")?;
        let bytes = std::fs::read(path)
            .with_context(|| format!("failed to read Needle schema '{}'", path.display()))?;
        let schema: Value = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid Needle schema '{}'", path.display()))?;
        ensure!(
            schema
                .get("name")
                .and_then(Value::as_str)
                .is_some_and(|name| !name.trim().is_empty())
                && schema.get("parameters").is_some_and(Value::is_object),
            "Needle extraction schema requires a name and a parameters object"
        );
        let mut inference = Inference::start(&self.settings, runtime, system, vec![schema]).await?;
        let response = inference.complete(text, observer).await?;
        Ok(match response.actionable_calls(&self.settings)?.first() {
            Some(call) => json!(call.arguments).to_string(),
            None => "null".into(),
        })
    }
}

//! Text-to-audio capability and its adapter for language-model tool calls.

use crate::tool::{Tool, ToolArguments, ToolOutput, ToolSpec};
use anyhow::{Context, Result, ensure};
use futures_util::future::BoxFuture;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf, sync::Arc};

/// Model-specific request options. Names and values match audio.cpp's `--request-option`.
pub type AudioOptions = BTreeMap<String, String>;

/// Generates a WAV artifact using a configured model.
pub trait AudioGenerator: Send + Sync {
    fn default_model(&self) -> Option<&str>;
    fn model_ids(&self) -> Vec<&str>;

    /// Write one WAV and return its absolute path.
    fn generate<'a>(
        &'a self,
        model: &'a str,
        prompt: &'a str,
        options: &'a AudioOptions,
    ) -> BoxFuture<'a, Result<PathBuf>>;
}

pub(crate) fn model_id<'a>(
    generator: &'a dyn AudioGenerator,
    requested: Option<&'a str>,
) -> Result<&'a str> {
    let model = requested
        .or_else(|| generator.default_model())
        .context("no audio models are configured")?;
    ensure!(
        generator.model_ids().contains(&model),
        "unknown audio model '{model}'"
    );
    Ok(model)
}

pub(crate) struct GenerateAudio {
    generator: Arc<dyn AudioGenerator>,
    spec: ToolSpec,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Arguments {
    prompt: String,
    model: Option<String>,
    #[serde(default)]
    options: AudioOptions,
}

impl GenerateAudio {
    pub(crate) fn new(generator: Arc<dyn AudioGenerator>) -> Option<Self> {
        let default = generator.default_model()?;
        let Value::Object(input_schema) = json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Text to speak for a speech model, or a description for a music or sound effects model."
                },
                "model": {
                    "type": "string", "enum": generator.model_ids(),
                    "description": format!("Audio model ID. Defaults to {default}.")
                },
                "options": {
                    "type": "object", "additionalProperties": {"type": "string"},
                    "description": "Optional request settings supported by the selected audio model."
                }
            },
            "required": ["prompt"],
            "additionalProperties": false
        }) else {
            unreachable!()
        };
        Some(Self {
            generator,
            spec: ToolSpec {
                name: "generate_audio".into(),
                description: "Generate speech, music, or sound effects with a configured local audio model and return the saved WAV path."
                    .into(),
                input_schema,
            },
        })
    }

    async fn generate(&self, arguments: ToolArguments) -> Result<Value> {
        let args: Arguments = serde_json::from_value(Value::Object(arguments))?;
        let model = model_id(self.generator.as_ref(), args.model.as_deref())?;
        let path = self
            .generator
            .generate(model, &args.prompt, &args.options)
            .await?;
        Ok(json!({"path": path.display().to_string(), "model": model}))
    }
}

impl Tool for GenerateAudio {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn omit_if_unsupported(&self) -> bool {
        true
    }

    fn call(&self, arguments: ToolArguments) -> BoxFuture<'_, Result<ToolOutput>> {
        Box::pin(async move {
            Ok(match self.generate(arguments).await {
                Ok(result) => ToolOutput::success(result),
                Err(error) => ToolOutput::error(format!("{error:#}")),
            })
        })
    }
}

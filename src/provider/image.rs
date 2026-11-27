//! Text-to-image capability and its adapter for language-model tool calls.

use crate::tool::{Tool, ToolArguments, ToolOutput, ToolSpec};
use anyhow::{Context, Result, ensure};
use futures_util::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{num::NonZeroU32, path::PathBuf, sync::Arc};

pub(crate) const DEFAULT_PROVIDER_ID: &str = "stable-diffusion.cpp";

/// Optional generation settings shared by model defaults, tools, and workflow nodes.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ImageOptions {
    pub width: Option<NonZeroU32>,
    pub height: Option<NonZeroU32>,
    pub steps: Option<NonZeroU32>,
    pub seed: Option<i64>,
    pub negative_prompt: Option<String>,
}

/// Generates an image artifact using a configured model.
pub trait ImageGenerator: Send + Sync {
    /// Default model used when a caller omits its ID.
    fn default_model(&self) -> Option<&str>;

    /// Model IDs available to the caller.
    fn model_ids(&self) -> Vec<&str>;

    /// Write one PNG and return its absolute path.
    fn generate<'a>(
        &'a self,
        model: &'a str,
        prompt: &'a str,
        options: &'a ImageOptions,
    ) -> BoxFuture<'a, Result<PathBuf>>;
}

pub(crate) fn model_id<'a>(
    generator: &'a dyn ImageGenerator,
    requested: Option<&'a str>,
) -> Result<&'a str> {
    let model = requested
        .or_else(|| generator.default_model())
        .context("no image models are configured")?;
    ensure!(
        generator.model_ids().contains(&model),
        "unknown image model '{model}'"
    );
    Ok(model)
}

pub(crate) struct GenerateImage {
    generator: Arc<dyn ImageGenerator>,
    spec: ToolSpec,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Arguments {
    model: Option<String>,
    prompt: String,
    #[serde(default)]
    options: ImageOptions,
}

impl GenerateImage {
    pub(crate) fn new(generator: Arc<dyn ImageGenerator>) -> Option<Self> {
        let default = generator.default_model()?;
        let Value::Object(input_schema) = json!({
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Describe the image to generate."},
                "model": {
                    "type": "string", "enum": generator.model_ids(),
                    "description": format!("Image model ID. Defaults to {default}.")
                },
                "options": {
                    "type": "object",
                    "properties": {
                        "width": {"type": "integer", "minimum": 1},
                        "height": {"type": "integer", "minimum": 1},
                        "steps": {"type": "integer", "minimum": 1},
                        "seed": {"type": "integer", "minimum": -1},
                        "negative_prompt": {"type": "string"}
                    },
                    "additionalProperties": false
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
                name: "generate_image".into(),
                description: "Generate one image from a text prompt and return the saved PNG path."
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

impl Tool for GenerateImage {
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

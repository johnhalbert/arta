//! Options for applying a model's chat template. Template files are resolved and read during
//! provider setup. Each inference step supplies its messages and optional function
//! definitions to the resulting template.

use crate::config::Config;
use anyhow::{Context, Result, ensure};
use llama_cpp_4::{
    chat::{ReasoningFormat, ToolChoice},
    prelude::ChatApplyParams,
};
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};

/// Provider-wide overrides for template rendering, response parsing, and generation stops.
#[derive(Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChatConfig {
    /// Inline Jinja template or built-in template name, mutually exclusive with a template
    /// file.
    pub template: Option<String>,
    /// Config-relative file whose contents become the template during provider
    /// initialization.
    pub template_file: Option<PathBuf>,
    /// Additional template variables encoded as a JSON object.
    pub template_kwargs: Option<toml::Table>,
    /// Pass the template's thinking switch, disabled by default.
    pub enable_thinking: bool,
    /// Ask the template to support multiple calls in a response. Arta executes the resulting
    /// batch in order.
    pub parallel_tool_calls: Option<bool>,
    /// Policy passed to the template when this inference step advertises tools.
    pub tool_choice: Option<ChatToolChoice>,
    /// Native parser mode for separating reasoning content from the reply.
    pub reasoning_format: Option<ChatReasoningFormat>,
    /// Leave reasoning and tool calls inline in assistant content when enabled.
    pub force_pure_content: Option<bool>,
    /// Override native selection of the Jinja rendering path.
    pub use_jinja: Option<bool>,
    /// Ask the template to append its assistant-generation prompt.
    pub add_generation_prompt: Option<bool>,
    /// Template rendering flag for a beginning-of-sequence marker. Tokenization separately
    /// consults the vocabulary.
    pub add_bos: Option<bool>,
    /// Template rendering flag for an end-of-sequence marker.
    pub add_eos: Option<bool>,
    /// Additional nonempty literal stop strings combined with any returned by the template.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
}

impl ChatConfig {
    /// Resolve an external template against the loaded configuration's directory.
    pub(super) fn resolve_paths(&mut self, config: &Config) {
        if let Some(path) = &mut self.template_file {
            *path = config.resolve_path(path);
        }
    }

    /// Read the selected file once and validate template exclusivity, nonblank source, and
    /// nonempty stop strings.
    pub(in crate::provider::llama_cpp) fn load_template(&mut self) -> Result<()> {
        if let Some(path) = self.template_file.take() {
            ensure!(
                self.template.is_none(),
                "set only one of chat.template and chat.template_file"
            );
            self.template =
                Some(fs::read_to_string(&path).with_context(|| {
                    format!("failed to read chat template '{}'", path.display())
                })?);
        }
        ensure!(
            self.template
                .as_ref()
                .is_none_or(|template| !template.trim().is_empty()),
            "chat template must not be empty"
        );
        ensure!(
            self.stop.iter().all(|stop| !stop.is_empty()),
            "chat.stop strings must not be empty"
        );
        Ok(())
    }

    /// Build native chat parameters from serialized messages. Apply tool choice and
    /// parallel-call flags only when tools are present.
    pub(in crate::provider::llama_cpp) fn params(
        &self,
        messages: String,
        tools: Option<String>,
    ) -> Result<ChatApplyParams> {
        let mut params = ChatApplyParams::new(messages).with_enable_thinking(self.enable_thinking);
        if let Some(tools) = tools {
            params = params.with_tools(tools);
            if let Some(value) = self.tool_choice {
                params = params.with_tool_choice(value.into());
            }
            if let Some(value) = self.parallel_tool_calls {
                params = params.with_parallel_tool_calls(value);
            }
        }
        if let Some(kwargs) = &self.template_kwargs {
            params = params.with_template_kwargs(serde_json::to_string(kwargs)?);
        }
        if let Some(value) = self.reasoning_format {
            params = params.with_reasoning_format(value.into());
        }
        if let Some(value) = self.force_pure_content {
            params = params.with_force_pure_content(value);
        }
        if let Some(value) = self.use_jinja {
            params = params.with_use_jinja(value);
        }
        if let Some(value) = self.add_generation_prompt {
            params = params.with_add_generation_prompt(value);
        }
        Ok(params.with_add_bos_eos(self.add_bos.unwrap_or(false), self.add_eos.unwrap_or(false)))
    }
}

/// Model-visible tool choice requested through the template.
#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatToolChoice {
    /// Allow the model to choose a tool call or a reply.
    Auto,
    /// Request a tool call.
    Required,
    /// Suppress tool selection in the template.
    None,
}

impl From<ChatToolChoice> for ToolChoice {
    fn from(value: ChatToolChoice) -> Self {
        match value {
            ChatToolChoice::Auto => Self::Auto,
            ChatToolChoice::Required => Self::Required,
            ChatToolChoice::None => Self::None,
        }
    }
}

/// Native format used when parsing reasoning in a templated response.
#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatReasoningFormat {
    /// Disable separate reasoning extraction.
    None,
    /// Use the native automatic reasoning format.
    Auto,
    /// Extract thinking-tag content into the reasoning field.
    Deepseek,
    /// Select the native legacy DeepSeek handling of thinking tags.
    DeepseekLegacy,
}

impl From<ChatReasoningFormat> for ReasoningFormat {
    fn from(value: ChatReasoningFormat) -> Self {
        match value {
            ChatReasoningFormat::None => Self::None,
            ChatReasoningFormat::Auto => Self::Auto,
            ChatReasoningFormat::Deepseek => Self::DeepSeek,
            ChatReasoningFormat::DeepseekLegacy => Self::DeepSeekLegacy,
        }
    }
}

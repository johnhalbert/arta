use crate::config::Config;
use anyhow::{Context, Result, ensure};
use llama_cpp_4::{
    chat::{ReasoningFormat, ToolChoice},
    prelude::ChatApplyParams,
};
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};

#[derive(Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChatConfig {
    pub template: Option<String>,
    pub template_file: Option<PathBuf>,
    pub template_kwargs: Option<toml::Table>,
    pub enable_thinking: bool,
    pub parallel_tool_calls: Option<bool>,
    pub tool_choice: Option<ChatToolChoice>,
    pub reasoning_format: Option<ChatReasoningFormat>,
    pub force_pure_content: Option<bool>,
    pub use_jinja: Option<bool>,
    pub add_generation_prompt: Option<bool>,
    pub add_bos: Option<bool>,
    pub add_eos: Option<bool>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
}

impl ChatConfig {
    pub(super) fn resolve_paths(&mut self, config: &Config) {
        if let Some(path) = &mut self.template_file {
            *path = config.resolve_path(path);
        }
    }

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

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatToolChoice {
    Auto,
    Required,
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

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatReasoningFormat {
    None,
    Auto,
    Deepseek,
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

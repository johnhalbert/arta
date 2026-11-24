//! Selects a configured or embedded template and renders ordinary input. Plain-text models
//! use a literal first request or a role-labelled transcript. Tool inference requires a
//! template with tool-call capability.

use super::chat_messages;
use crate::provider::{PromptObserver, Turn, llama_cpp::ChatConfig};
use anyhow::{Context, Result, bail, ensure};
use llama_cpp_4::{
    StringFromModelError,
    prelude::{AddBos, ChatParams, ChatTemplates, LlamaModel, LlamaToken, Special},
};
use serde_json::Value;

/// Rendering mode selected once when the model loads. The template variant can contain either
/// an explicit override or the model default.
pub(super) enum ModelTemplate {
    /// Compiled chat template from the GGUF metadata or an explicit override.
    Embedded(ChatTemplates),
    /// Build role-labelled text when needed and tokenize literal input without template
    /// processing.
    PlainText,
}

impl ModelTemplate {
    /// Prefer the explicit override. Otherwise use the embedded template when present and
    /// fall back to plain text for an empty or missing template. Other native errors
    /// propagate.
    pub(super) fn from_model(model: &LlamaModel, template: Option<&str>) -> Result<Self> {
        if let Some(template) = template {
            return ChatTemplates::from_model(model, Some(template))
                .map(Self::Embedded)
                .context("invalid configured chat template");
        }
        match model.chat_template(None) {
            Ok(template) if !template.is_empty() => ChatTemplates::from_model(model, None)
                .map(Self::Embedded)
                .context("invalid model chat template"),
            Ok(_) | Err(StringFromModelError::ReturnedError(-1)) => Ok(Self::PlainText),
            Err(error) => Err(error).context("failed to read model chat template"),
        }
    }

    /// Read the template capability JSON and require the tool-call flag before structured
    /// inference.
    pub(super) fn tool_template(&self) -> Result<&ChatTemplates> {
        let Self::Embedded(templates) = self else {
            bail!("tool use requires a chat template");
        };
        let caps: Value = serde_json::from_str(&templates.caps_json()?)
            .context("failed to read chat template capabilities")?;
        ensure!(
            caps["supports_tool_calls"].as_bool() == Some(true),
            "the model's chat template does not support tool calls"
        );
        Ok(templates)
    }

    /// Observe the exact rendered string before tokenization. For templated input, combine
    /// template and configured stops. Plain text uses configured stops and treats
    /// special-token spellings as literal text.
    pub(super) fn prepare_prompt(
        &self,
        model: &LlamaModel,
        config: &ChatConfig,
        system_prompt: Option<&str>,
        history: &[Turn],
        prompt: &str,
        observer: Option<&PromptObserver>,
    ) -> Result<(Vec<LlamaToken>, Vec<String>, Option<ChatParams>)> {
        ensure!(
            !system_prompt.is_some_and(|prompt| prompt.contains('\0')),
            "system prompt contains a NUL character"
        );
        let add_bos = if model.get_vocab().get_add_bos() {
            AddBos::Always
        } else {
            AddBos::Never
        };

        if let Self::Embedded(templates) = self {
            let messages = chat_messages(system_prompt, history, prompt);
            let params = config.params(Value::Array(messages).to_string(), None)?;
            let chat = templates
                .apply(&params)
                .context("failed to format prompt")?;
            if let Some(observe) = observer {
                observe(&chat.prompt);
            }
            let tokens = chat
                .tokenize_prompt(model, add_bos)
                .context("failed to tokenize chat prompt")?;
            let mut stops: Vec<String> = serde_json::from_str(&chat.additional_stops_json)
                .context("failed to read chat template stop strings")?;
            stops.extend(config.stop.iter().cloned());
            return Ok((tokens, stops, Some(chat)));
        }

        let mut transcript;
        let prompt = if history.is_empty() && system_prompt.is_none() {
            prompt
        } else {
            transcript = String::new();
            if let Some(system_prompt) = system_prompt {
                transcript.push_str("System: ");
                transcript.push_str(system_prompt);
                transcript.push_str("\n\n");
            }
            for turn in history {
                transcript.push_str("User: ");
                transcript.push_str(&turn.prompt);
                transcript.push_str("\nAssistant: ");
                transcript.push_str(&turn.reply);
                transcript.push('\n');
            }
            transcript.push_str("User: ");
            transcript.push_str(prompt);
            transcript.push_str("\nAssistant:");
            &transcript
        };
        if let Some(observe) = observer {
            observe(prompt);
        }
        let tokens = model
            .str_to_token_with_special(prompt, add_bos, Special::Plaintext)
            .context("failed to tokenize prompt")?;
        Ok((tokens, config.stop.clone(), None))
    }
}

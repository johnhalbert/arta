use super::chat_messages;
use crate::provider::Turn;
use anyhow::{Context, Result, bail, ensure};
use llama_cpp_4::{
    StringFromModelError,
    prelude::{AddBos, ChatApplyParams, ChatTemplates, LlamaModel, LlamaToken, Special},
};
use serde_json::Value;

pub(super) enum ModelTemplate {
    Embedded(ChatTemplates),
    PlainText,
}

impl ModelTemplate {
    pub(super) fn from_model(model: &LlamaModel) -> Result<Self> {
        match model.chat_template(None) {
            Ok(template) if !template.is_empty() => ChatTemplates::from_model(model, None)
                .map(Self::Embedded)
                .context("invalid model chat template"),
            Ok(_) | Err(StringFromModelError::ReturnedError(-1)) => Ok(Self::PlainText),
            Err(error) => Err(error).context("failed to read model chat template"),
        }
    }

    pub(super) fn tool_template(&self) -> Result<&ChatTemplates> {
        let Self::Embedded(templates) = self else {
            bail!("tool use requires a model with an embedded chat template");
        };
        let caps: Value = serde_json::from_str(&templates.caps_json()?)
            .context("failed to read chat template capabilities")?;
        ensure!(
            caps["supports_tool_calls"].as_bool() == Some(true),
            "the model's chat template does not support tool calls"
        );
        Ok(templates)
    }

    pub(super) fn prepare_prompt(
        &self,
        model: &LlamaModel,
        history: &[Turn],
        prompt: &str,
    ) -> Result<(Vec<LlamaToken>, Vec<String>)> {
        let add_bos = if model.get_vocab().get_add_bos() {
            AddBos::Always
        } else {
            AddBos::Never
        };

        if let Self::Embedded(templates) = self {
            let messages = chat_messages(history, prompt);
            let params = ChatApplyParams::new(Value::Array(messages).to_string())
                .with_enable_thinking(false);
            let chat = templates
                .apply(&params)
                .context("failed to format prompt")?;
            let tokens = chat
                .tokenize_prompt(model, add_bos)
                .context("failed to tokenize chat prompt")?;
            let stops = serde_json::from_str(&chat.additional_stops_json)
                .context("failed to read chat template stop strings")?;
            return Ok((tokens, stops));
        }

        let mut transcript;
        let prompt = if history.is_empty() {
            prompt
        } else {
            transcript = String::new();
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
        let tokens = model
            .str_to_token_with_special(prompt, add_bos, Special::Plaintext)
            .context("failed to tokenize prompt")?;
        Ok((tokens, Vec::new()))
    }
}

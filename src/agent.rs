use crate::{
    model::Model,
    provider::{Provider, Turn, tool},
    stream::Message,
    tool::{Tool, Tools},
};
use anyhow::Result;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

/// Settings attached to a named agent in a session.
#[derive(Clone, Copy, Default)]
pub struct AgentOptions<'a> {
    pub workspace: Option<&'a Path>,
    pub system_prompt: Option<&'a str>,
}

/// Runs prompts against a model selected when the agent is created.
pub struct Agent {
    model: Arc<dyn Model>,
    provider: Arc<dyn Provider>,
    tools: Arc<Tools>,
    workspace: Option<PathBuf>,
    system_prompt: Option<String>,
}

impl Agent {
    pub(crate) fn new(
        provider: Arc<dyn Provider>,
        model: Arc<dyn Model>,
        tools: Arc<Tools>,
        options: AgentOptions<'_>,
    ) -> Self {
        Self {
            model,
            provider,
            tools,
            workspace: options.workspace.map(Path::to_owned),
            system_prompt: options.system_prompt.map(str::to_owned),
        }
    }

    pub fn provider(&self) -> &dyn Provider {
        self.provider.as_ref()
    }

    pub fn model(&self) -> &dyn Model {
        self.model.as_ref()
    }

    pub fn workspace(&self) -> Option<&Path> {
        self.workspace.as_deref()
    }

    /// Run one independent prompt against the agent's model.
    pub async fn prompt(&self, prompt: &str) -> Result<Message> {
        if self.system_prompt.is_some() {
            return self.chat(&[], prompt).await;
        }
        tool::send(self.provider(), self.model.id(), prompt, &self.tools).await
    }

    /// Run a prompt with conversation history supplied by the caller.
    pub async fn chat(&self, history: &[Turn], prompt: &str) -> Result<Message> {
        self.chat_with_tools(history, prompt, self.system_prompt.as_deref(), &self.tools)
            .await
    }

    /// Run a turn with the supplied tools and optional instructions appended to the agent's system prompt.
    pub async fn chat_with_context(
        &self,
        history: &[Turn],
        prompt: &str,
        instructions: &str,
        tools: impl IntoIterator<Item = Arc<dyn Tool>>,
    ) -> Result<Message> {
        let system_prompt = match self.system_prompt.as_deref() {
            base if instructions.is_empty() => base.map(std::borrow::Cow::Borrowed),
            Some(base) => Some(format!("{base}\n\n{instructions}").into()),
            None => Some(instructions.into()),
        };
        let mut selected = Tools::default();
        for tool in tools {
            selected.register(tool)?;
        }
        self.chat_with_tools(history, prompt, system_prompt.as_deref(), &selected)
            .await
    }

    async fn chat_with_tools(
        &self,
        history: &[Turn],
        prompt: &str,
        system_prompt: Option<&str>,
        tools: &Tools,
    ) -> Result<Message> {
        tool::chat(
            self.provider(),
            self.model.id(),
            system_prompt,
            history,
            prompt,
            tools,
        )
        .await
    }
}

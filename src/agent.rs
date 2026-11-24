use crate::{
    model::Model,
    provider::{PromptObserver, Provider, Turn, tool},
    stream::Message,
    tool::{Tool, Tools},
};
use anyhow::{Context, Result};
use std::{
    borrow::Cow,
    path::{Path, PathBuf},
    sync::Arc,
};

/// Settings attached to a named agent in a session.
#[derive(Clone, Copy, Default)]
pub struct AgentOptions<'a> {
    pub workspace: Option<&'a Path>,
    pub system_prompt: Option<&'a str>,
}

/// Runs prompts against the agent's selected model.
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

    /// Switch models within the current provider while keeping the agent's settings.
    pub async fn switch_model(&mut self, model_id: &str) -> Result<()> {
        let provider = self.provider.clone();
        let current_id = self.model.id().to_owned();
        let model_id = model_id.to_owned();
        self.model =
            tokio::task::spawn_blocking(move || provider.switch_model(&current_id, &model_id))
                .await
                .context("model switch worker failed")??;
        Ok(())
    }

    pub fn workspace(&self) -> Option<&Path> {
        self.workspace.as_deref()
    }

    /// Release the selected model's resources while keeping its selection.
    pub async fn unload_model(&self) -> Result<()> {
        let provider = self.provider.clone();
        let model_id = self.model.id().to_owned();
        tokio::task::spawn_blocking(move || provider.unload_model(&model_id))
            .await
            .context("model unload worker failed")?
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
        self.chat_with_tools(
            history,
            prompt,
            self.system_prompt.as_deref(),
            &self.tools,
            None,
        )
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
        self.chat_with_context_observed(history, prompt, instructions, tools, None)
            .await
    }

    /// Run a context turn, optionally observing the provider's input at each inference step.
    pub async fn chat_with_context_observed(
        &self,
        history: &[Turn],
        prompt: &str,
        instructions: &str,
        tools: impl IntoIterator<Item = Arc<dyn Tool>>,
        observer: Option<PromptObserver>,
    ) -> Result<Message> {
        let system_prompt = self.system_prompt_with(instructions);
        let mut selected = Tools::default();
        for tool in tools {
            selected.register(tool)?;
        }
        self.chat_with_tools(
            history,
            prompt,
            system_prompt.as_deref(),
            &selected,
            observer,
        )
        .await
    }

    pub(crate) async fn chat_recorded(
        &self,
        history: &[Turn],
        past_exchanges: &[Vec<tool::ToolExchange>],
        prompt: &str,
        instructions: &str,
        observer: Option<PromptObserver>,
    ) -> Result<(Message, Vec<tool::ToolExchange>)> {
        let system_prompt = self.system_prompt_with(instructions);
        tool::chat_recorded(
            self.provider(),
            tool::ToolRequest {
                model_id: self.model.id(),
                system_prompt: system_prompt.as_deref(),
                history,
                prompt,
                tools: &[],
                exchanges: &[],
            },
            &self.tools,
            past_exchanges,
            observer,
        )
        .await
    }

    fn system_prompt_with<'a>(&'a self, instructions: &'a str) -> Option<Cow<'a, str>> {
        match self.system_prompt.as_deref() {
            base if instructions.is_empty() => base.map(Cow::Borrowed),
            Some(base) => Some(format!("{base}\n\n{instructions}").into()),
            None => Some(instructions.into()),
        }
    }

    async fn chat_with_tools(
        &self,
        history: &[Turn],
        prompt: &str,
        system_prompt: Option<&str>,
        tools: &Tools,
        observer: Option<PromptObserver>,
    ) -> Result<Message> {
        tool::chat(
            self.provider(),
            self.model.id(),
            system_prompt,
            history,
            prompt,
            tools,
            observer,
        )
        .await
    }
}

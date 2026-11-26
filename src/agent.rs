//! An agent binds one provider and model to a workspace, base instructions, and tool
//! snapshot. Turn history belongs to the calling REPL, workflow node, or library client.

use crate::{
    model::Model,
    provider::{PromptObserver, Provider, Turn, TurnObservers, tool},
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
    /// Directory used when connecting local MCP processes for this profile.
    pub workspace: Option<&'a Path>,
    /// Base instructions prepended to each chat, including turns with additional skill
    /// instructions.
    pub system_prompt: Option<&'a str>,
}

/// Runs prompts against the agent's selected model.
pub struct Agent {
    /// Selection updated after a successful switch. Inference passes its provider-local ID.
    model: Arc<dyn Model>,
    /// Shared implementation used for inference, switching, and resource release.
    provider: Arc<dyn Provider>,
    /// Registry snapshot captured at binding time, used for prompts and recorded turns.
    tools: Arc<Tools>,
    /// Owned copy of the optional profile directory.
    workspace: Option<PathBuf>,
    /// Base instructions preserved across prompts and model switches.
    system_prompt: Option<String>,
}

impl Agent {
    /// Clone borrowed profile settings into the binding and share the supplied provider,
    /// metadata, and tools.
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

    /// Borrow the provider currently responsible for inference and model switching.
    pub fn provider(&self) -> &dyn Provider {
        self.provider.as_ref()
    }

    /// Borrow metadata for the selection. The llama.cpp implementation keeps the heavyweight
    /// model in its provider cache.
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

    /// Borrow the workspace copied into this binding at creation.
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
            TurnObservers::default(),
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
        self.chat_with_context_and_observers(
            history,
            prompt,
            instructions,
            tools,
            TurnObservers {
                prompt: observer,
                ..Default::default()
            },
        )
        .await
    }

    /// Run a context turn with automatic and explicitly selected tools, optionally reporting
    /// rendered inputs and provisional reply text.
    pub async fn chat_with_context_and_observers(
        &self,
        history: &[Turn],
        prompt: &str,
        instructions: &str,
        tools: impl IntoIterator<Item = Arc<dyn Tool>>,
        observers: TurnObservers,
    ) -> Result<Message> {
        let system_prompt = self.system_prompt_with(instructions);
        let mut selected = self.tools.selected(|tool| tool.omit_if_unsupported());
        for tool in tools {
            selected.register(tool)?;
        }
        self.chat_with_tools(
            history,
            prompt,
            system_prompt.as_deref(),
            &selected,
            observers,
        )
        .await
    }

    /// Run one workflow turn with the binding's tools. Replay earlier per-turn exchanges
    /// through the adapter and return the new exchanges for the node to retain.
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
            TurnObservers {
                prompt: observer,
                ..Default::default()
            },
        )
        .await
    }

    /// Join profile and per-turn instructions with a blank line. Borrow the existing text
    /// when only one source supplies it.
    fn system_prompt_with<'a>(&'a self, instructions: &'a str) -> Option<Cow<'a, str>> {
        match self.system_prompt.as_deref() {
            base if instructions.is_empty() => base.map(Cow::Borrowed),
            Some(base) => Some(format!("{base}\n\n{instructions}").into()),
            None => Some(instructions.into()),
        }
    }

    /// Pass an explicitly selected registry and caller-owned history to the
    /// provider-independent tool loop.
    async fn chat_with_tools(
        &self,
        history: &[Turn],
        prompt: &str,
        system_prompt: Option<&str>,
        tools: &Tools,
        observers: TurnObservers,
    ) -> Result<Message> {
        tool::chat(
            self.provider(),
            self.model.id(),
            system_prompt,
            history,
            prompt,
            tools,
            observers,
        )
        .await
    }
}

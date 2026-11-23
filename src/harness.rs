use crate::{
    agent::{Agent, AgentOptions},
    config::Config,
    mcp::McpTools,
    model::{Model, ModelId},
    provider::{Provider, registry::ProviderRegistry, tool},
    session::Session,
    tool::{Tool, Tools},
};
use anyhow::{Context, Result, bail, ensure};
use std::{
    collections::{HashMap, hash_map::Entry},
    path::Path,
    sync::Arc,
};

/// Owns the sessions and providers that agents use for inference.
pub struct Harness {
    sessions: HashMap<String, Session>,
    providers: HashMap<String, Arc<dyn Provider>>,
    tools: Arc<Tools>,
}

impl Harness {
    pub fn new(config: &Config) -> Result<Self> {
        Self::with_registry(config, &ProviderRegistry::discover()?)
    }

    pub fn with_registry(config: &Config, registry: &ProviderRegistry) -> Result<Self> {
        for id in config.provider_ids() {
            ensure!(registry.contains(id), "unknown configured provider '{id}'");
        }
        let mut harness = Self {
            sessions: HashMap::new(),
            providers: HashMap::new(),
            tools: Arc::default(),
        };
        for (id, create) in registry.list() {
            harness
                .register_provider(id, || create(config))
                .with_context(|| format!("failed to initialize provider '{id}'"))?;
        }
        Ok(harness)
    }

    /// Create a provider and register it under a unique identifier.
    pub fn register_provider(
        &mut self,
        provider_id: &str,
        create: impl FnOnce() -> Result<Arc<dyn Provider>>,
    ) -> Result<()> {
        ensure!(
            !provider_id.trim().is_empty(),
            "provider identifier cannot be empty"
        );
        match self.providers.entry(provider_id.to_owned()) {
            Entry::Vacant(entry) => {
                entry.insert(create()?);
                Ok(())
            }
            Entry::Occupied(_) => bail!("provider '{provider_id}' is already registered"),
        }
    }

    fn provider_handle(&self, provider_id: &str) -> Result<&Arc<dyn Provider>> {
        self.providers
            .get(provider_id)
            .with_context(|| format!("unknown provider '{provider_id}'"))
    }

    pub fn provider(&self, provider_id: &str) -> Result<&dyn Provider> {
        Ok(self.provider_handle(provider_id)?.as_ref())
    }

    /// Get a shared reference to a model owned by a registered provider.
    pub fn model(&self, model: ModelId<'_>) -> Result<Arc<dyn Model>> {
        self.provider(model.provider)?.model(model.model)
    }

    /// Register a tool for sessions created after this call and for direct sends.
    pub fn register_tool(&mut self, tool: Arc<dyn Tool>) -> Result<()> {
        Arc::make_mut(&mut self.tools).register(tool)
    }

    /// Connect the configured MCP servers and register their tools for new sessions.
    pub async fn connect_mcp(&mut self, config: &Config) -> Result<()> {
        self.connect_mcp_in(config, None).await
    }

    /// Connect MCP servers with a workspace as the working directory of local processes.
    pub async fn connect_mcp_in(
        &mut self,
        config: &Config,
        workspace: Option<&Path>,
    ) -> Result<()> {
        self.connect_mcp_tools_in(config, workspace).await.map(drop)
    }

    /// Connect configured MCP servers and return their tools grouped by server name.
    pub async fn connect_mcp_tools_in(
        &mut self,
        config: &Config,
        workspace: Option<&Path>,
    ) -> Result<McpTools> {
        if config.mcp.servers.is_empty() {
            return Ok(McpTools::default());
        }
        let mut tools = (*self.tools).clone();
        let directory = config.resolve_path(Path::new("."));
        let servers = config
            .mcp
            .register(&directory, workspace.unwrap_or(&directory), &mut tools)
            .await?;
        self.tools = Arc::new(tools);
        Ok(servers)
    }

    /// Create a session with at least one named agent and bind each agent to a model.
    pub fn create_session(
        &mut self,
        session_id: &str,
        agents: &[(&str, ModelId<'_>)],
    ) -> Result<()> {
        self.create_session_with_agents(
            session_id,
            agents
                .iter()
                .map(|&(name, model)| (name, model, AgentOptions::default())),
        )
    }

    /// Create a session with a model and options for each named agent.
    pub fn create_session_with_agents<'a>(
        &mut self,
        session_id: &str,
        agents: impl IntoIterator<Item = (&'a str, ModelId<'a>, AgentOptions<'a>)>,
    ) -> Result<()> {
        ensure!(
            !session_id.trim().is_empty(),
            "session identifier cannot be empty"
        );
        ensure!(
            !self.sessions.contains_key(session_id),
            "session '{session_id}' is already registered"
        );
        let session = Session::new(agents, |model, options| {
            let provider = self.provider_handle(model.provider)?;
            Ok(Agent::new(
                Arc::clone(provider),
                provider.model(model.model)?,
                Arc::clone(&self.tools),
                options,
            ))
        })?;
        self.sessions.insert(session_id.to_owned(), session);
        Ok(())
    }

    pub fn session(&self, session_id: &str) -> Result<&Session> {
        self.sessions
            .get(session_id)
            .with_context(|| format!("unknown session '{session_id}'"))
    }

    pub fn agent(&self, session_id: &str, agent_id: &str) -> Result<&Agent> {
        self.session(session_id)?.agent(agent_id)
    }

    /// Drop a session and its agents. The harness continues to own its providers.
    pub fn close_session(&mut self, session_id: &str) -> Result<()> {
        self.sessions
            .remove(session_id)
            .with_context(|| format!("unknown session '{session_id}'"))?;
        Ok(())
    }

    pub async fn send(&self, model: ModelId<'_>, prompt: &str) -> Result<String> {
        tool::send(
            self.provider(model.provider)?,
            model.model,
            prompt,
            &self.tools,
        )
        .await?
        .into_text()
        .await
    }
}

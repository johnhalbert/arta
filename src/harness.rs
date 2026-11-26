//! Composition root for provider instances, session bindings, and shared tools.
//! [`Harness::create_session`] captures tool handles for a group of agents.
//! [`Harness::send`] and [`Harness::run_workflow`] also work without a session.

use crate::{
    agent::{Agent, AgentOptions},
    config::Config,
    mcp::McpTools,
    model::{Model, ModelId},
    provider::{
        Provider,
        image::{GenerateImage, ImageGenerator},
        registry::ProviderRegistry,
        stable_diffusion_cpp::StableDiffusionCpp,
        tool,
    },
    session::Session,
    tool::{Tool, Tools},
    workflow::Workflow,
};
use anyhow::{Context, Result, bail, ensure};
use std::{
    collections::{HashMap, HashSet, hash_map::Entry},
    path::Path,
    sync::Arc,
};

/// Owns the sessions and providers that agents use for inference.
pub struct Harness {
    /// Published sessions keyed by caller-provided ID. A failed binding leaves this map
    /// intact.
    sessions: HashMap<String, Session>,
    /// Initialized instances shared by their agents and workflow bindings.
    providers: HashMap<String, Arc<dyn Provider>>,
    /// Current registry for direct sends and future session bindings. Writes use
    /// copy-on-write.
    tools: Arc<Tools>,
    /// Keys to omit when constructing a workflow node's workspace-specific tool set.
    mcp_tool_names: HashSet<String>,
    /// Base settings for named workflows and provider reuse comparisons.
    config: Config,
}

impl Harness {
    /// Discover linked provider factories and instantiate each with the supplied
    /// configuration.
    pub fn new(config: &Config) -> Result<Self> {
        Self::with_registry(config, &ProviderRegistry::discover()?)
    }

    /// Reject unknown provider tables before running factories. Keep a copy of the base
    /// settings for later workflow loads.
    pub fn with_registry(config: &Config, registry: &ProviderRegistry) -> Result<Self> {
        for id in config.provider_ids() {
            ensure!(registry.contains(id), "unknown configured provider '{id}'");
        }
        let mut harness = Self {
            sessions: HashMap::new(),
            providers: HashMap::new(),
            tools: Arc::default(),
            mcp_tool_names: HashSet::new(),
            config: config.clone(),
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
                let provider = create()?;
                if provider_id == StableDiffusionCpp::ID
                    && let Some(generator) = Arc::clone(&provider).image_generator()
                    && let Some(tool) = GenerateImage::new(generator)
                {
                    Arc::make_mut(&mut self.tools).register(Arc::new(tool))?;
                }
                entry.insert(provider);
                Ok(())
            }
            Entry::Occupied(_) => bail!("provider '{provider_id}' is already registered"),
        }
    }

    /// Look up the shared handle retained by agents and workflow model bindings.
    pub(crate) fn provider_handle(&self, provider_id: &str) -> Result<&Arc<dyn Provider>> {
        self.providers
            .get(provider_id)
            .with_context(|| format!("unknown provider '{provider_id}'"))
    }

    /// Borrow a registered provider by its registry ID.
    pub fn provider(&self, provider_id: &str) -> Result<&dyn Provider> {
        Ok(self.provider_handle(provider_id)?.as_ref())
    }

    /// Resolve a provider's image generation capability by its registry ID.
    pub fn image_generator(&self, provider_id: &str) -> Result<Arc<dyn ImageGenerator>> {
        Arc::clone(self.provider_handle(provider_id)?)
            .image_generator()
            .with_context(|| format!("provider '{provider_id}' does not support image generation"))
    }

    /// Get a shared handle for a model from a registered provider.
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
        self.mcp_tool_names.extend(
            servers
                .values()
                .flatten()
                .map(|tool| tool.spec().name.clone()),
        );
        self.tools = Arc::new(tools);
        Ok(servers)
    }

    /// Load and run a named workflow using this harness's configuration.
    pub async fn run_workflow(&mut self, name: &str, prompt: &str) -> Result<String> {
        let workflow = Workflow::load(&self.config, name)?;
        workflow.run(self, prompt).await
    }

    /// Borrow the base settings used when deciding whether a workflow can reuse a provider.
    pub(crate) fn workflow_config(&self) -> &Config {
        &self.config
    }

    /// Take a copy of the registered tools and remove harness-discovered MCP entries. Each
    /// workflow profile reconnects its selected servers in its own workspace.
    pub(crate) fn workflow_tools(&self) -> Tools {
        let mut tools = (*self.tools).clone();
        for name in &self.mcp_tool_names {
            tools.remove(name);
        }
        tools
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

    /// Resolve a session ID or report which ID was missing.
    pub fn session(&self, session_id: &str) -> Result<&Session> {
        self.sessions
            .get(session_id)
            .with_context(|| format!("unknown session '{session_id}'"))
    }

    /// Resolve the session first, then look up the named agent inside it.
    pub fn agent(&self, session_id: &str, agent_id: &str) -> Result<&Agent> {
        self.session(session_id)?.agent(agent_id)
    }

    /// Borrow one agent exclusively, for example to switch its selected model.
    pub fn agent_mut(&mut self, session_id: &str, agent_id: &str) -> Result<&mut Agent> {
        self.sessions
            .get_mut(session_id)
            .with_context(|| format!("unknown session '{session_id}'"))?
            .agent_mut(agent_id)
    }

    /// Drop a session and its agents. The harness continues to own its providers.
    pub fn close_session(&mut self, session_id: &str) -> Result<()> {
        self.sessions
            .remove(session_id)
            .with_context(|| format!("unknown session '{session_id}'"))?;
        Ok(())
    }

    /// Send a single prompt with the harness's current tools and collect the reply. This path
    /// requires no session or conversation history.
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

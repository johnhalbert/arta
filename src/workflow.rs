//! Loads an ordered tree of agent, image, audio and loop nodes. Each execution owns its model binding,
//! per-node conversations, and workspace-specific MCP connections. [`Workflow::events`]
//! drives the same execution as [`Workflow::run`].

mod agent;
mod event;
mod r#loop;
mod model;
pub mod node;

use self::{
    agent::WorkflowAgent, event::Progress, r#loop::LoopDecision, model::ActiveModel, node::Node,
};
use crate::{
    config::{AgentProfile, Config},
    harness::Harness,
    provider::{
        audio::{self, AudioOptions},
        image::{self, ImageOptions},
    },
    tool::{Tool, Tools},
};
use anyhow::{Context, Result, ensure};
pub use event::{WorkflowEvent, WorkflowEvents};
use futures_util::{
    StreamExt,
    future::{BoxFuture, ready},
    stream,
};
pub use r#loop::LoopIteration;
use serde::Deserialize;
use std::{
    collections::{BTreeMap, HashMap, HashSet, btree_map, hash_map},
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::sync::mpsc;

/// Location used to find on-disk workflow packages.
#[derive(Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WorkflowConfig {
    /// Parent of named workflow folders. Relative paths start at the TOML directory.
    pub path: Option<PathBuf>,
}

/// A workflow definition and the Markdown agent profiles it references.
pub struct Workflow {
    /// Display identifier from the YAML definition.
    id: String,
    /// Validated tree traversed in declaration order.
    nodes: Vec<Node>,
    /// Resolved profiles keyed by the filename stem referenced by each agent leaf.
    agents: BTreeMap<String, Profile>,
}

/// YAML header and node tree before profile resolution.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Definition {
    /// Workflow label read independently of the directory name.
    id: String,
    /// Root sequence checked by the recursive node visitor.
    nodes: Vec<Node>,
}

/// One referenced Markdown agent and the effective configuration after its overrides.
struct Profile {
    /// Instructions, workspace and named MCP servers shared by nodes referencing this
    /// profile.
    agent: AgentProfile,
    /// Effective config after loading this profile and applying invocation overrides.
    settings: Config,
}

impl Workflow {
    /// Read `workflow.yaml` and the referenced agent Markdown files from a named directory.
    pub fn load(config: &Config, name: &str) -> Result<Self> {
        Self::load_with_overrides(config, name, |_| Ok(()))
    }

    /// Load a workflow and apply additional settings to each agent profile.
    pub fn load_with_overrides(
        config: &Config,
        name: &str,
        apply_overrides: impl Fn(&mut Config) -> Result<()>,
    ) -> Result<Self> {
        ensure!(
            !name.is_empty()
                && name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')),
            "invalid workflow name '{name}': use letters, digits, underscores, or hyphens"
        );
        let root =
            config.workflows.path.as_deref().context(
                "set workflows.path or pass --workflow-path to select a workflow directory",
            )?;
        let directory = config.resolve_path(root).join(name);
        let file = directory.join("workflow.yaml");
        let source = fs::read_to_string(&file)
            .with_context(|| format!("failed to read workflow '{}'", file.display()))?;
        let definition: Definition = serde_saphyr::from_str(&source)
            .with_context(|| format!("invalid workflow '{}'", file.display()))?;
        ensure!(
            !definition.id.trim().is_empty(),
            "workflow id cannot be empty"
        );
        let mut ids = HashSet::new();
        let mut agents = BTreeMap::new();
        Node::visit_sequence(&definition.nodes, &mut ids, &mut |id, name| {
            if let btree_map::Entry::Vacant(agent) = agents.entry(name.to_owned()) {
                let mut settings = config.clone();
                let profile = AgentProfile::load_from(&directory, name, &mut settings)
                    .with_context(|| format!("failed to load agent '{name}' for node '{id}'"))?;
                apply_overrides(&mut settings)
                    .with_context(|| format!("invalid workflow overrides for node '{id}'"))?;
                agent.insert(Profile {
                    agent: profile,
                    settings,
                });
            }
            Ok(())
        })
        .with_context(|| format!("invalid workflow '{}'", file.display()))?;
        Ok(Self {
            id: definition.id,
            nodes: definition.nodes,
            agents,
        })
    }

    /// Return the ID declared in YAML. Directory lookup uses the name passed to load.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Borrow the validated tree in execution order, for example to draw an outline.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Run the workflow with fresh node histories and return the final node's output.
    pub async fn run(&self, harness: &mut Harness, prompt: &str) -> Result<String> {
        self.run_with_progress(harness, prompt, &Progress::default())
            .await
    }

    /// Run while streaming node, model, and output events. Dropping the stream cancels the run.
    pub fn events<'a>(
        &'a self,
        harness: &'a mut Harness,
        prompt: &'a str,
        include_model_input: bool,
    ) -> WorkflowEvents<'a> {
        let (sender, receiver) = mpsc::unbounded_channel();
        let execution = stream::once(async move {
            let progress = Progress::new(sender.clone(), include_model_input);
            if let Err(error) = self.run_with_progress(harness, prompt, &progress).await {
                let _ = sender.send(Err(error));
            }
        })
        .filter_map(|()| ready(None));
        let events = stream::unfold(receiver, |mut receiver| async {
            receiver.recv().await.map(|event| (event, receiver))
        });
        stream::select(events, execution).boxed()
    }

    /// Check the request and provider IDs, create fresh run state, and execute the root
    /// sequence. Attempt final model release on success or failure, preserving the execution
    /// error if both fail.
    async fn run_with_progress(
        &self,
        harness: &Harness,
        prompt: &str,
        progress: &Progress,
    ) -> Result<String> {
        ensure!(!prompt.trim().is_empty(), "prompt cannot be empty");
        for profile in self.agents.values() {
            harness.provider(&profile.settings.app.provider)?;
        }
        let mut run = Run {
            harness,
            profiles: &self.agents,
            agents: HashMap::new(),
            connections: HashMap::new(),
            model: None,
            request: prompt,
            previous: None,
            progress,
            loops: Vec::new(),
        };
        let result = run.sequence(&self.nodes).await;
        let cleanup = ActiveModel::release(&mut run.model).await;
        result.with_context(|| format!("workflow '{}' failed", self.id))?;
        cleanup.context("failed to unload the final workflow model")?;
        Ok(run.previous.unwrap_or_default())
    }
}

/// State for one invocation. Node IDs key histories, profile names key settings, and
/// workspace/server pairs key connections.
struct Run<'a> {
    /// Source of registered providers and workflow-visible in-process tools.
    harness: &'a Harness,
    /// Validated name-to-profile map borrowed from the definition.
    profiles: &'a BTreeMap<String, Profile>,
    /// Conversation state per node ID, created on first visit.
    agents: HashMap<&'a str, WorkflowAgent>,
    /// Discovered MCP handles reused for the same workspace and server name during this run.
    connections: HashMap<(PathBuf, String), Vec<Arc<dyn Tool>>>,
    /// Binding currently owned by the executor. The final binding is released during cleanup.
    model: Option<ActiveModel>,
    /// Unmodified request included in every node's prompt.
    request: &'a str,
    /// Immediate predecessor's output across all nodes and loop boundaries.
    previous: Option<String>,
    /// Optional event sink shared with node agents and model selection.
    progress: &'a Progress,
    /// Active iteration stack copied into node-start events.
    loops: Vec<LoopIteration>,
}

impl<'a> Run<'a> {
    /// Await siblings in definition order. Boxing breaks the recursive future type for nested
    /// loops.
    fn sequence<'b>(&'b mut self, nodes: &'a [Node]) -> BoxFuture<'b, Result<()>> {
        Box::pin(async move {
            for node in nodes {
                self.node(node).await?;
            }
            Ok(())
        })
    }

    /// Dispatch a leaf or enter the loop interpreter for a subtree.
    async fn node(&mut self, node: &'a Node) -> Result<()> {
        match node {
            Node::Agent { id, agent } => {
                self.agent(id, agent, false).await?;
            }
            Node::Image {
                id,
                provider,
                model,
                prompt,
                options,
            } => {
                self.image(
                    id,
                    provider.as_deref().unwrap_or(image::DEFAULT_PROVIDER_ID),
                    model.as_deref(),
                    prompt.as_deref(),
                    options,
                )
                .await
                .with_context(|| format!("image node '{id}' failed"))?;
            }
            Node::Audio {
                id,
                provider,
                model,
                prompt,
                options,
            } => {
                self.audio(
                    id,
                    provider.as_deref().unwrap_or(audio::DEFAULT_PROVIDER_ID),
                    model.as_deref(),
                    prompt.as_deref(),
                    options,
                )
                .await
                .with_context(|| format!("audio node '{id}' failed"))?;
            }
            Node::Loop { id, nodes } => r#loop::execute(self, id, nodes).await?,
        }
        Ok(())
    }

    async fn image(
        &mut self,
        id: &str,
        provider: &str,
        model: Option<&str>,
        prompt: Option<&str>,
        options: &ImageOptions,
    ) -> Result<()> {
        let generator = self.harness.image_generator(provider)?;
        let model = image::model_id(generator.as_ref(), model)?;
        self.progress
            .emit(|| WorkflowEvent::ImageStarted {
                id: id.to_owned(),
                provider: provider.to_owned(),
                model: model.to_owned(),
                loops: self.loops.clone(),
            })
            .await;
        ActiveModel::release(&mut self.model).await?;
        let path = generator
            .generate(model, self.artifact_prompt(prompt), options)
            .await?;
        self.finish_artifact(path).await;
        Ok(())
    }

    async fn audio(
        &mut self,
        id: &str,
        provider: &str,
        model: Option<&str>,
        prompt: Option<&str>,
        options: &AudioOptions,
    ) -> Result<()> {
        let generator = self.harness.audio_generator(provider)?;
        let model = audio::model_id(generator.as_ref(), model)?;
        self.progress
            .emit(|| WorkflowEvent::AudioStarted {
                id: id.to_owned(),
                provider: provider.to_owned(),
                model: model.to_owned(),
                loops: self.loops.clone(),
            })
            .await;
        ActiveModel::release(&mut self.model).await?;
        let path = generator
            .generate(model, self.artifact_prompt(prompt), options)
            .await?;
        self.finish_artifact(path).await;
        Ok(())
    }

    fn artifact_prompt<'b>(&'b self, prompt: Option<&'b str>) -> &'b str {
        prompt
            .or_else(|| {
                self.previous
                    .as_deref()
                    .filter(|text| !text.trim().is_empty())
            })
            .unwrap_or(self.request)
    }

    async fn finish_artifact(&mut self, path: PathBuf) {
        let output = path.display().to_string();
        self.progress
            .emit(|| WorkflowEvent::Output(output.clone()))
            .await;
        self.progress.emit(|| WorkflowEvent::NodeFinished).await;
        self.previous = Some(output);
    }

    /// Attach the node ID and profile name to failures from a visit.
    async fn agent(
        &mut self,
        id: &'a str,
        name: &str,
        decides_loop: bool,
    ) -> Result<Option<LoopDecision>> {
        self.agent_turn(id, name, decides_loop)
            .await
            .with_context(|| format!("node '{id}' (agent '{name}') failed"))
    }

    /// Announce the node, resolve tools and model, and pass the original request plus any
    /// nonempty predecessor reply to the agent. Save the reply for the next handoff after
    /// success.
    async fn agent_turn(
        &mut self,
        id: &'a str,
        name: &str,
        decides_loop: bool,
    ) -> Result<Option<LoopDecision>> {
        let profile = &self.profiles[name];
        self.progress
            .emit(|| WorkflowEvent::NodeStarted {
                id: id.to_owned(),
                agent: name.to_owned(),
                loops: self.loops.clone(),
            })
            .await;
        let tools = self.tools(profile).await?;
        let model = ActiveModel::select(
            &mut self.model,
            self.harness,
            &profile.settings,
            self.progress,
        )
        .await?;
        let prompt = match &self.previous {
            Some(reply) if !reply.is_empty() => format!(
                "Workflow request:\n{}\n\nPrevious node's reply:\n{reply}",
                self.request
            ),
            _ => self.request.to_owned(),
        };
        let (reply, decision) = self
            .agents
            .entry(id)
            .or_default()
            .run(
                model,
                &profile.agent,
                tools,
                prompt,
                self.loops.last().filter(|_| decides_loop),
                self.progress,
            )
            .await?;
        self.progress.emit(|| WorkflowEvent::NodeFinished).await;
        self.previous = Some(reply);
        Ok(decision)
    }

    /// Start with harness tools excluding harness-discovered MCP entries. Add only the
    /// profile's named servers, reusing connections for the same workspace and server within
    /// the run.
    async fn tools(&mut self, profile: &Profile) -> Result<Tools> {
        let mut tools = self.harness.workflow_tools();
        let directory = profile.settings.resolve_path(Path::new("."));
        let workspace = profile.agent.options().workspace.unwrap_or(&directory);
        let mut seen = HashSet::new();
        for name in profile
            .agent
            .mcp_servers()
            .iter()
            .filter(|name| seen.insert(*name))
        {
            let key = (workspace.to_owned(), name.clone());
            if let hash_map::Entry::Vacant(entry) = self.connections.entry(key.clone()) {
                let discovered = profile.settings.mcp.servers[name]
                    .connect_in(name, &directory, workspace)
                    .await
                    .with_context(|| format!("failed to connect to MCP server '{name}'"))?;
                entry.insert(discovered);
            }
            for tool in &self.connections[&key] {
                tools.register(Arc::clone(tool))?;
            }
        }
        Ok(tools)
    }
}

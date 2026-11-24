//! Model Context Protocol client adapter. Startup registers discovered tools with the
//! harness. Workflow nodes use the same discovery code in their profile's workspace. Each
//! tool retains its remote name and a shared running connection.

use crate::tool::{Tool, ToolArguments, ToolOutput, ToolSpec, Tools};
use anyhow::{Context, Result, ensure};
use futures_util::future::BoxFuture;
use rmcp::{
    ClientLifecycleMode, ClientServiceExt, RoleClient,
    model::{CallToolRequestParams, ClientConfig, Implementation, ProtocolVersion},
    service::RunningService,
    transport::{
        StreamableHttpClientTransport, TokioChildProcess,
        streamable_http_client::StreamableHttpClientTransportConfig,
    },
};
use serde::Deserialize;
use serde_json::Value;
use std::{
    collections::BTreeMap,
    env,
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::Duration,
};
use tokio::{process::Command, time::timeout};

/// Connections keyed by the names used in profiles and REPL mentions.
#[derive(Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct McpConfig {
    /// Servers visited in name order during startup discovery.
    pub servers: BTreeMap<String, McpServerConfig>,
}

/// Discovered handles grouped by server so a mention can expose a whole group.
pub type McpTools = BTreeMap<String, Vec<Arc<dyn Tool>>>;

impl McpConfig {
    /// Connect servers sequentially and register their handles. The harness stages these
    /// changes on a registry copy before publishing them.
    pub(crate) async fn register(
        &self,
        directory: &Path,
        workspace: &Path,
        tools: &mut Tools,
    ) -> Result<McpTools> {
        let mut servers = BTreeMap::new();
        for (name, server) in &self.servers {
            let discovered = server
                .connect_in(name, directory, workspace)
                .await
                .with_context(|| format!("failed to connect to MCP server '{name}'"))?;
            for tool in &discovered {
                tools.register(Arc::clone(tool))?;
            }
            servers.insert(name.clone(), discovered);
        }
        Ok(servers)
    }
}

/// Untagged TOML transport configuration selected by the presence of a command or URL.
#[derive(Clone, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum McpServerConfig {
    /// Launch a local process and speak MCP over its stdin and stdout.
    Stdio {
        /// Executable name for PATH lookup or a path resolved against the configuration
        /// directory.
        command: PathBuf,
        /// Arguments passed directly to the child process.
        #[serde(default)]
        args: Vec<String>,
        /// Environment entries added to or overriding the inherited child environment.
        #[serde(default)]
        env: BTreeMap<String, String>,
        /// Allowlist of original remote names. Omission exposes all advertised tools.
        tools: Option<Vec<String>>,
    },
    /// Connect a Streamable HTTP client to the configured service.
    Http {
        /// Endpoint passed to the transport.
        url: String,
        /// Environment variable read at connection time for the bearer token.
        bearer_token_env: Option<String>,
        /// Optional allowlist applied to the names returned by discovery.
        tools: Option<Vec<String>>,
    },
}

/// Deadline for one server's startup, MCP handshake, and complete tool listing together.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// Deadline for one remote function invocation.
const CALL_TIMEOUT: Duration = Duration::from_secs(60);
/// Negotiated rmcp client service shared by tools discovered from one server.
type Connection = RunningService<RoleClient, ClientConfig>;

impl McpServerConfig {
    /// Connect to a server, discover tools, and expose them with a server name prefix.
    pub async fn connect(&self, name: &str, directory: &Path) -> Result<Vec<Arc<dyn Tool>>> {
        self.connect_in(name, directory, directory).await
    }

    /// Validate the server name and bound the entire handshake and tool listing to 30
    /// seconds.
    pub(crate) async fn connect_in(
        &self,
        name: &str,
        directory: &Path,
        workspace: &Path,
    ) -> Result<Vec<Arc<dyn Tool>>> {
        ensure!(
            !name.is_empty()
                && name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-')),
            "invalid MCP server name '{name}'"
        );
        timeout(CONNECT_TIMEOUT, self.discover(name, directory, workspace))
            .await
            .context("MCP connection or tool discovery timed out")?
    }

    /// Start the transport, negotiate the client lifecycle, list tools, and check every
    /// allowlisted name. Preserve the remote dispatch name while prefixing the local name.
    async fn discover(
        &self,
        name: &str,
        directory: &Path,
        workspace: &Path,
    ) -> Result<Vec<Arc<dyn Tool>>> {
        let client = ClientConfig::new(
            Default::default(),
            Implementation::new("arta", env!("CARGO_PKG_VERSION")),
        );
        let lifecycle = ClientLifecycleMode::Auto {
            preferred_versions: vec![ProtocolVersion::LATEST],
            legacy_version: None,
        };
        let (connection, selected) = match self {
            Self::Stdio {
                command,
                args,
                env,
                tools,
            } => {
                let directory = std::path::absolute(directory)?;
                let executable = if command.components().count() > 1 {
                    directory.join(command)
                } else {
                    command.to_owned()
                };
                let mut command = Command::new(executable);
                command
                    .args(args)
                    .envs(env)
                    .current_dir(std::path::absolute(workspace)?)
                    .kill_on_drop(true);
                let (transport, _) = TokioChildProcess::builder(command)
                    .stderr(Stdio::null())
                    .spawn()
                    .context("failed to start the MCP server process")?;
                (
                    client.serve_with_lifecycle(transport, lifecycle).await?,
                    tools,
                )
            }
            Self::Http {
                url,
                bearer_token_env,
                tools,
            } => {
                let mut config = StreamableHttpClientTransportConfig::with_uri(url.as_str());
                if let Some(variable) = bearer_token_env {
                    let token = env::var(variable).with_context(|| {
                        format!("cannot read MCP bearer token from '{variable}'")
                    })?;
                    ensure!(!token.is_empty(), "MCP bearer token '{variable}' is empty");
                    config = config.auth_header(token);
                }
                let transport = StreamableHttpClientTransport::from_config(config);
                (
                    client.serve_with_lifecycle(transport, lifecycle).await?,
                    tools,
                )
            }
        };
        let connection = Arc::new(connection);
        let available = connection
            .list_all_tools()
            .await
            .context("MCP tools/list failed")?;
        if let Some(selected) = selected {
            for tool in selected {
                ensure!(
                    available.iter().any(|entry| entry.name == tool.as_str()),
                    "MCP server does not advertise the configured tool '{tool}'"
                );
            }
        }
        Ok(available
            .into_iter()
            .filter(|tool| {
                selected
                    .as_ref()
                    .is_none_or(|names| names.iter().any(|name| tool.name == name.as_str()))
            })
            .map(|tool| {
                Arc::new(McpTool {
                    spec: ToolSpec {
                        name: format!("{name}__{}", tool.name.replace('.', "_")),
                        description: tool.description.unwrap_or_default().into_owned(),
                        input_schema: (*tool.input_schema).clone(),
                    },
                    remote_name: tool.name.into_owned(),
                    connection: Arc::clone(&connection),
                }) as Arc<dyn Tool>
            })
            .collect())
    }
}

/// One advertised schema and a shared transport. Calls use a 60-second timeout and prefer
/// structured content over newline-joined text blocks.
struct McpTool {
    /// Locally prefixed name with the remote description and input schema.
    spec: ToolSpec,
    /// Server's original function name, preserved across local aliasing.
    remote_name: String,
    /// Keeps the negotiated transport alive while any discovered handle retains it.
    connection: Arc<Connection>,
}

impl Tool for McpTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn call(&self, arguments: ToolArguments) -> BoxFuture<'_, Result<ToolOutput>> {
        Box::pin(async move {
            let request =
                CallToolRequestParams::new(self.remote_name.clone()).with_arguments(arguments);
            let result = timeout(CALL_TIMEOUT, self.connection.call_tool(request))
                .await
                .context("MCP tools/call timed out")??;
            let is_error = result.is_error.unwrap_or_default();
            let content = result.structured_content.unwrap_or_else(|| {
                let text = result
                    .content
                    .iter()
                    .map(|block| {
                        block
                            .as_text()
                            .map_or("[Non-text MCP content]", |text| &text.text)
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                Value::String(text)
            });
            Ok(ToolOutput { content, is_error })
        })
    }
}

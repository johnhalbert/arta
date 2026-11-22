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

#[derive(Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct McpConfig {
    pub servers: BTreeMap<String, McpServerConfig>,
}

impl McpConfig {
    pub(crate) async fn register(
        &self,
        directory: &Path,
        workspace: &Path,
        tools: &mut Tools,
    ) -> Result<()> {
        for (name, server) in &self.servers {
            let discovered = server
                .connect_in(name, directory, workspace)
                .await
                .with_context(|| format!("failed to connect to MCP server '{name}'"))?;
            for tool in discovered {
                tools.register(tool)?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum McpServerConfig {
    Stdio {
        command: PathBuf,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        env: BTreeMap<String, String>,
        tools: Option<Vec<String>>,
    },
    Http {
        url: String,
        bearer_token_env: Option<String>,
        tools: Option<Vec<String>>,
    },
}

const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const CALL_TIMEOUT: Duration = Duration::from_secs(60);
type Connection = RunningService<RoleClient, ClientConfig>;

impl McpServerConfig {
    /// Connect to a server, discover tools, and expose them with a server name prefix.
    pub async fn connect(&self, name: &str, directory: &Path) -> Result<Vec<Arc<dyn Tool>>> {
        self.connect_in(name, directory, directory).await
    }

    async fn connect_in(
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

struct McpTool {
    spec: ToolSpec,
    remote_name: String,
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

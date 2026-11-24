//! Configuration shared by the binary and library. The CLI caches its resolved startup plan.
//! Library callers can load independent TOML files and pass each configuration to its own
//! harness.

mod agent;

use crate::{
    mcp::McpConfig,
    provider::llama_cpp::LlamaCPPOverrides,
    skill::SkillConfig,
    workflow::{Workflow, WorkflowConfig},
};
pub use agent::AgentProfile;
use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    collections::BTreeMap,
    fs,
    io::ErrorKind,
    num::NonZeroU32,
    path::{self, Path, PathBuf},
    sync::LazyLock,
};

/// Cache the resolved startup plan or its first initialization error. A bare invocation uses
/// the default arguments to enter the conversation interface.
static CONFIG: LazyLock<Result<StartupConfig>> = LazyLock::new(|| {
    let args = if std::env::args_os().len() == 1 {
        Args::default()
    } else {
        Args::parse()
    };
    args.into_config()
});

/// Resolved settings and optional inputs that determine the binary's execution mode.
pub struct StartupConfig {
    /// Base configuration for constructing the harness.
    pub settings: Config,
    /// Literal request or the stdin sentinel `-`. Absence selects a conversation.
    pub prompt: Option<String>,
    /// Standalone profile supplying the session name, workspace, and system instructions.
    pub agent: Option<AgentProfile>,
    /// Validated definition with a separate effective configuration for each referenced
    /// profile.
    pub workflow: Option<Workflow>,
}

impl StartupConfig {
    /// Use the selected profile's filename stem, or the session agent name `default`.
    pub fn agent_name(&self) -> &str {
        self.agent.as_ref().map_or("default", AgentProfile::name)
    }
}

/// Return the CLI configuration, initialized once from TOML and CLI arguments.
pub fn get() -> Result<&'static StartupConfig, &'static anyhow::Error> {
    CONFIG.as_ref()
}

#[derive(Default, Parser, Debug)]
#[command(
    version,
    about = "Run a prompt or workflow against a local GGUF model, or start a conversation",
    after_help = "Run arta with no arguments to open the terminal UI using config.toml."
)]
/// Command-line inputs before profile merging and path resolution.
struct Args {
    /// Path to a TOML configuration file (defaults to config.toml when present).
    #[arg(long, value_name = "PATH")]
    config: Option<PathBuf>,

    /// Run the named agent from ~/.arta/agents/NAME.md.
    #[arg(long, value_name = "NAME")]
    agent: Option<String>,

    /// Run a named workflow under the configured workflow path.
    #[arg(
        long,
        value_name = "NAME",
        conflicts_with = "agent",
        requires = "prompt"
    )]
    workflow: Option<String>,

    /// Override the directory containing workflows for this run.
    #[arg(long, value_name = "PATH", requires = "workflow")]
    workflow_path: Option<PathBuf>,

    /// Override the provider selected in the configuration.
    #[arg(long, value_name = "ID")]
    provider: Option<String>,

    /// Override the model identifier selected in the configuration.
    #[arg(long, value_name = "ID")]
    model_id: Option<String>,

    /// Override the selected model's GGUF file.
    #[arg(short, long, value_name = "PATH")]
    model: Option<PathBuf>,

    /// Prompt text. Use - to read UTF-8 text from stdin.
    #[arg(short, long, value_name = "TEXT", required_unless_present_any = ["agent", "config"])]
    prompt: Option<String>,

    /// Total token budget for the prompt and response (defaults to 4096 or the model's limit, whichever is lower).
    #[arg(short, long, value_name = "TOKENS")]
    context_length: Option<NonZeroU32>,

    /// Maximum number of tokens to generate (defaults to 256).
    #[arg(long, value_name = "TOKENS")]
    max_tokens: Option<NonZeroU32>,

    /// Number of layers to offload to an available GPU.
    #[arg(long, value_name = "LAYERS")]
    gpu_layers: Option<u32>,

    /// Show llama.cpp diagnostics on stderr.
    #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "true")]
    verbose: Option<bool>,
}

impl Args {
    /// Layer the selected agent onto TOML, then apply CLI overrides. For a workflow, apply
    /// the CLI settings to each profile after it loads.
    fn into_config(self) -> Result<StartupConfig> {
        let mut settings = Config::load(self.config.as_deref())?;
        let agent = self
            .agent
            .as_deref()
            .map(|name| AgentProfile::load(name, &mut settings))
            .transpose()?;
        let overrides = CliOverrides {
            provider: self.provider,
            model: self.model_id,
            llama: LlamaCPPOverrides {
                model_path: self.model,
                context_length: self.context_length,
                max_tokens: self.max_tokens,
                gpu_layers: self.gpu_layers,
                verbose: self.verbose,
            },
        };
        if self.workflow.is_none() {
            overrides.apply(&mut settings)?;
        }
        if let Some(path) = self.workflow_path {
            settings.workflows.path = Some(path::absolute(path)?);
        }
        let workflow = self
            .workflow
            .as_deref()
            .map(|name| {
                Workflow::load_with_overrides(&settings, name, |settings| overrides.apply(settings))
            })
            .transpose()?;

        Ok(StartupConfig {
            settings,
            prompt: self.prompt,
            agent,
            workflow,
        })
    }
}

/// Selection and inference overrides applied together after profile loading.
struct CliOverrides {
    /// Explicit provider selection applied before model-specific settings.
    provider: Option<String>,
    /// Provider-local model ID selected for the remaining overrides.
    model: Option<String>,
    /// CLI inference and file overrides for the resulting selection.
    llama: LlamaCPPOverrides,
}

impl CliOverrides {
    /// Select provider and model first so llama.cpp overrides update the intended model
    /// entry.
    fn apply(&self, settings: &mut Config) -> Result<()> {
        if let Some(provider) = &self.provider {
            settings.app.provider.clone_from(provider);
        }
        if let Some(model) = &self.model {
            settings.app.model.clone_from(model);
        }
        self.llama.clone().apply(settings)
    }
}

/// TOML root. Provider-specific tables stay untyped until the corresponding factory reads
/// them.
#[derive(Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Current provider and model IDs used to bind an agent.
    pub app: AppConfig,
    /// Connections available to startup discovery and workflow profiles.
    pub mcp: McpConfig,
    /// Directories scanned for skill packages.
    pub skills: SkillConfig,
    /// Root for named workflow definitions.
    pub workflows: WorkflowConfig,
    /// Raw TOML keyed by factory ID. Each provider owns the schema of its table.
    providers: BTreeMap<String, toml::Table>,
    /// Base for relative TOML paths, captured on load and excluded from deserialization.
    #[serde(skip)]
    directory: PathBuf,
}

/// Default model selection, initially provider `llama.cpp` and model `local`.
#[derive(Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AppConfig {
    /// Registry key used to look up the provider.
    pub provider: String,
    /// Model key interpreted by that provider.
    pub model: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            provider: "llama.cpp".into(),
            model: "local".into(),
        }
    }
}

impl Config {
    /// Load the requested file, or look for `config.toml` in the working directory.
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let required = path.is_some();
        let path = path::absolute(path.unwrap_or_else(|| Path::new("config.toml")))?;
        let source = match fs::read_to_string(&path) {
            Err(error) if !required && error.kind() == ErrorKind::NotFound => {
                return Ok(Self::default());
            }
            result => {
                result.with_context(|| format!("failed to read config '{}'", path.display()))?
            }
        };
        let mut config: Self = toml::from_str(&source)
            .with_context(|| format!("invalid config '{}'", path.display()))?;
        config.directory = path
            .parent()
            .context("config file has no parent directory")?
            .to_owned();
        Ok(config)
    }

    /// Deserialize the named table as the caller's type, using its default when the table is
    /// absent.
    pub fn provider_settings<T: DeserializeOwned + Default>(&self, provider: &str) -> Result<T> {
        self.providers
            .get(provider)
            .map(|settings| {
                toml::Value::Table(settings.clone())
                    .try_into()
                    .with_context(|| format!("invalid config for provider '{provider}'"))
            })
            .unwrap_or_else(|| Ok(T::default()))
    }

    /// Serialize a complete provider table and replace the previous entry on success.
    pub fn set_provider_settings(
        &mut self,
        provider: &str,
        settings: impl Serialize,
    ) -> Result<()> {
        let settings = toml::Table::try_from(settings)
            .with_context(|| format!("failed to update config for provider '{provider}'"))?;
        self.providers.insert(provider.to_owned(), settings);
        Ok(())
    }

    /// Join a path to the loaded TOML file's parent. Absolute paths take precedence. With a
    /// default configuration the base is empty.
    pub fn resolve_path(&self, path: &Path) -> PathBuf {
        self.directory.join(path)
    }

    /// Iterate configured provider keys so the harness can reject unregistered tables.
    pub(crate) fn provider_ids(&self) -> impl Iterator<Item = &str> {
        self.providers.keys().map(String::as_str)
    }

    /// Borrow the raw table used to compare effective workflow provider settings.
    pub(crate) fn provider_table(&self, id: &str) -> Option<&toml::Table> {
        self.providers.get(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::llama_cpp::{LlamaCPP, LlamaCPPConfig};

    #[test]
    fn startup_config_merges_cli_overrides_before_selecting_the_model() -> Result<()> {
        let file = format!("{}/config.example.toml", env!("CARGO_MANIFEST_DIR"));
        let startup = Args::try_parse_from([
            "arta",
            "--config",
            &file,
            "--prompt",
            "Hello",
            "--model-id",
            "one-off",
            "--model",
            "replacement.gguf",
            "--max-tokens",
            "32",
            "--verbose",
        ])?
        .into_config()?;

        assert_eq!(startup.prompt.as_deref(), Some("Hello"));
        assert_eq!(startup.settings.app.model, "one-off");
        let provider: LlamaCPPConfig = startup.settings.provider_settings(LlamaCPP::ID)?;
        assert_eq!(provider.max_tokens.get(), 32);
        assert!(provider.verbose);
        assert_eq!(provider.models[0].id, "local");
        assert_eq!(provider.models[1].id, "one-off");
        assert_eq!(provider.models[1].path, path::absolute("replacement.gguf")?);
        Ok(())
    }
}

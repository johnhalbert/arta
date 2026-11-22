use super::Config;
use crate::{agent::AgentOptions, front_matter, provider::llama_cpp::LlamaCPPOverrides};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::{
    env, fs,
    path::{self, Path, PathBuf},
};

/// A named agent loaded from a Markdown file in the user's Arta directory.
pub struct AgentProfile {
    name: String,
    workspace: PathBuf,
    system_prompt: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FrontMatter {
    provider: String,
    model: String,
    workspace: PathBuf,
    #[serde(default)]
    config: LlamaCPPOverrides,
}

impl AgentProfile {
    pub(super) fn load(name: &str, settings: &mut Config) -> Result<Self> {
        ensure!(
            !name.is_empty()
                && name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')),
            "invalid agent name '{name}': use letters, digits, underscores, or hyphens"
        );
        let home =
            env::home_dir().context("cannot determine the home directory for Arta agents")?;
        let directory = home.join(".arta").join("agents");
        let file = directory.join(format!("{name}.md"));
        let source = fs::read_to_string(&file)
            .with_context(|| format!("failed to read agent definition '{}'", file.display()))?;
        Self::parse(&home, &directory, name, &source, settings)
            .with_context(|| format!("invalid agent definition '{}'", file.display()))
    }

    fn parse(
        home: &Path,
        directory: &Path,
        name: &str,
        source: &str,
        settings: &mut Config,
    ) -> Result<Self> {
        let (mut metadata, body): (FrontMatter, _) = front_matter::parse(source)?;
        ensure!(
            !metadata.provider.trim().is_empty(),
            "provider cannot be empty"
        );
        ensure!(!metadata.model.trim().is_empty(), "model cannot be empty");
        ensure!(
            !body.contains('\0'),
            "system prompt contains a NUL character"
        );

        let workspace =
            resolve_path(home, directory, &metadata.workspace).context("invalid workspace path")?;
        ensure!(
            workspace.is_dir(),
            "workspace '{}' must be an existing directory",
            workspace.display()
        );
        if let Some(model_path) = &mut metadata.config.model_path {
            *model_path =
                resolve_path(home, directory, model_path).context("invalid model path")?;
        }

        settings.app.provider = metadata.provider;
        settings.app.model = metadata.model;
        metadata.config.apply(settings)?;
        Ok(Self {
            name: name.to_owned(),
            workspace,
            system_prompt: body.to_owned(),
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn options(&self) -> AgentOptions<'_> {
        AgentOptions {
            workspace: Some(&self.workspace),
            system_prompt: (!self.system_prompt.trim().is_empty()).then_some(&self.system_prompt),
        }
    }
}

fn resolve_path(home: &Path, directory: &Path, path: &Path) -> Result<PathBuf> {
    ensure!(!path.as_os_str().is_empty(), "path cannot be empty");
    let resolved = match path.strip_prefix("~") {
        Ok(relative) => home.join(relative),
        Err(_) => directory.join(path),
    };
    Ok(path::absolute(resolved)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::llama_cpp::{LlamaCPP, LlamaCPPConfig};

    #[test]
    fn applies_agent_overrides_and_preserves_the_markdown_body() -> Result<()> {
        let repository = Path::new(env!("CARGO_MANIFEST_DIR"));
        let directory = repository.join("tests");
        let mut settings = Config::load(Some(&repository.join("config.example.toml")))?;
        let body = "\r\n# Instructions\r\n\r\nReview diffs.\r\n---\r\n";
        let source = format!(
            "---\r\nprovider: llama.cpp\r\nmodel: review\r\nworkspace: ~/tests\r\nconfig:\r\n  model_path: review.gguf\r\n  max_tokens: 48\r\n---\r\n{body}"
        );
        let profile =
            AgentProfile::parse(repository, &directory, "reviewer", &source, &mut settings)?;

        assert_eq!(profile.name(), "reviewer");
        assert_eq!(profile.options().workspace, Some(directory.as_path()));
        assert_eq!(profile.options().system_prompt, Some(body));
        assert_eq!(settings.app.model, "review");
        let provider: LlamaCPPConfig = settings.provider_settings(LlamaCPP::ID)?;
        assert_eq!(provider.max_tokens.get(), 48);
        assert_eq!(provider.context_length.unwrap().get(), 4096);
        assert_eq!(provider.models[0].id, "local");
        assert_eq!(provider.models[1].id, "review");
        assert_eq!(provider.models[1].path, directory.join("review.gguf"));
        Ok(())
    }
}

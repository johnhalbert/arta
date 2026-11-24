//! Discovers Markdown packages and exposes their catalog through one activation tool. A
//! caller can also read a skill body directly and remove it from the remaining catalog.

use crate::{
    front_matter,
    tool::{Tool, ToolArguments, ToolOutput, ToolSpec},
};
use anyhow::{Context, Result, bail, ensure};
use futures_util::future::BoxFuture;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, btree_map::Entry},
    fmt::Write,
    fs,
    io::ErrorKind,
    path::PathBuf,
    sync::Arc,
};

/// Search roots for immediate child directories containing `SKILL.md`.
#[derive(Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SkillConfig {
    /// Directories resolved relative to the TOML configuration before discovery.
    pub directories: Vec<PathBuf>,
}

/// Required metadata advertised to the model and used for completion.
#[derive(Deserialize)]
struct FrontMatter {
    /// Validated slug shared by the activation tool and REPL mention.
    name: String,
    /// Short description advertised without loading the skill body into a prompt.
    description: String,
}

/// Advertises available skills and returns their Markdown instructions on demand.
pub struct ActivateSkill {
    /// Function description and argument schema derived from this catalog's remaining names.
    spec: ToolSpec,
    /// Sorted name-to-body index. Filtered catalogs share their skill allocations.
    skills: BTreeMap<String, Arc<Skill>>,
}

/// Loaded description and verbatim body shared across filtered catalogs.
struct Skill {
    /// Catalog summary used in tool descriptions and REPL suggestions.
    description: String,
    /// Markdown following the closing YAML fence, returned verbatim on activation.
    instructions: String,
}

impl ActivateSkill {
    /// Read skill packages from the immediate subdirectories of each directory.
    pub fn discover(directories: impl IntoIterator<Item = PathBuf>) -> Result<Option<Self>> {
        let mut skills = BTreeMap::new();
        for directory in directories {
            let entries = fs::read_dir(&directory)
                .with_context(|| format!("failed to scan skills in '{}'", directory.display()))?;
            for entry in entries {
                let path = entry
                    .with_context(|| format!("failed to scan skills in '{}'", directory.display()))?
                    .path();
                if !path.is_dir() {
                    continue;
                }
                let path = path.join("SKILL.md");
                let source = match fs::read_to_string(&path) {
                    Err(error) if error.kind() == ErrorKind::NotFound => continue,
                    result => result
                        .with_context(|| format!("failed to read skill '{}'", path.display()))?,
                };
                let (metadata, body) = parse(&source)
                    .with_context(|| format!("invalid skill '{}'", path.display()))?;
                match skills.entry(metadata.name) {
                    Entry::Vacant(entry) => {
                        entry.insert(Arc::new(Skill {
                            description: metadata.description,
                            instructions: body.to_owned(),
                        }));
                    }
                    Entry::Occupied(entry) => {
                        bail!("duplicate skill '{}' in '{}'", entry.key(), path.display());
                    }
                }
            }
        }
        Self::with_skills(skills)
    }

    /// Iterate names and trimmed descriptions in name order.
    pub fn catalog(&self) -> impl Iterator<Item = (&str, &str)> {
        self.skills
            .iter()
            .map(|(name, skill)| (name.as_str(), skill.description.trim()))
    }

    /// Borrow the Markdown body for a catalog name.
    pub fn instructions(&self, name: &str) -> Option<&str> {
        self.skills
            .get(name)
            .map(|skill| skill.instructions.as_str())
    }

    /// Builds an activation tool for skills whose instructions are still needed.
    pub fn excluding(&self, active: impl Fn(&str) -> bool) -> Result<Option<Self>> {
        Self::with_skills(
            self.skills
                .iter()
                .filter(|(name, _)| !active(name))
                .map(|(name, skill)| (name.clone(), Arc::clone(skill)))
                .collect(),
        )
    }

    /// Build a function schema whose name argument enumerates the remaining skills. An empty
    /// catalog yields no tool.
    fn with_skills(skills: BTreeMap<String, Arc<Skill>>) -> Result<Option<Self>> {
        if skills.is_empty() {
            return Ok(None);
        }

        let mut description = String::from(
            "The activate_skill tool loads a skill's instructions by name. When asked which tools you can call, list activate_skill as a tool. When asked which skills are available, answer directly from the catalog below. Call activate_skill with the matching skill name before performing a task that needs its instructions.\n\nAvailable skills:\n",
        );
        for (name, skill) in &skills {
            writeln!(description, "- {name}: {}", skill.description.trim())?;
        }
        let spec = ToolSpec {
            name: "activate_skill".into(),
            description,
            input_schema: serde_json::from_value(json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "enum": skills.keys().collect::<Vec<_>>() }
                },
                "required": ["name"],
                "additionalProperties": false
            }))?,
        };
        Ok(Some(Self { spec, skills }))
    }
}

impl Tool for ActivateSkill {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn call(&self, arguments: ToolArguments) -> BoxFuture<'_, Result<ToolOutput>> {
        Box::pin(async move {
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return Ok(ToolOutput::error("Specify the skill name as a string."));
            };
            Ok(match self.instructions(name) {
                Some(body) => ToolOutput::success(body),
                None => ToolOutput::error(format!("Unknown skill '{name}'")),
            })
        })
    }
}

/// Enforce a lowercase ASCII slug of at most 64 bytes with single interior hyphens and a
/// nonblank description.
fn parse(source: &str) -> Result<(FrontMatter, &str)> {
    let (metadata, body): (FrontMatter, _) = front_matter::parse(source)?;
    let name = &metadata.name;
    ensure!(
        (1..=64).contains(&name.len())
            && name
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
            && !name.starts_with('-')
            && !name.ends_with('-')
            && !name.contains("--"),
        "invalid skill name '{name}'"
    );
    ensure!(
        !metadata.description.trim().is_empty(),
        "skill description is empty"
    );
    Ok((metadata, body))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_windows_front_matter_and_preserves_markdown() -> Result<()> {
        let body = "# Instructions\r\n\r\n---\r\n\r\n    Indented content\r\n";
        let source = format!(
            "\u{feff}---\r\nname: review-notes\r\ndescription: >-\r\n  Review notes\r\n  for accuracy.\r\nmetadata:\r\n  version: '1'\r\n---\r\n{body}"
        );

        let (metadata, instructions) = parse(&source)?;

        assert_eq!(metadata.name, "review-notes");
        assert_eq!(metadata.description, "Review notes for accuracy.");
        assert_eq!(instructions, body);
        Ok(())
    }

    #[tokio::test]
    async fn catalog_advertises_the_skill_and_activation_returns_its_body() -> Result<()> {
        let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/skills");
        let tool = ActivateSkill::discover([directory])?.context("expected a skill tool")?;
        let spec = tool.spec();
        assert_eq!(spec.name, "activate_skill");
        assert!(
            spec.description
                .contains("- review-notes: Check notes for accuracy.")
        );
        assert!(!spec.description.contains("Compare each claim"));
        assert_eq!(
            spec.input_schema["properties"]["name"]["enum"],
            json!(["review-notes"])
        );

        let arguments = serde_json::from_value(json!({"name": "review-notes"}))?;
        let output = tool.call(arguments).await?;
        assert!(!output.is_error);
        assert_eq!(
            output.content.as_str().map(str::trim_end),
            Some("Compare each claim with the source notes.")
        );
        Ok(())
    }
}

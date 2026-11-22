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
};

#[derive(Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SkillConfig {
    pub directories: Vec<PathBuf>,
}

#[derive(Deserialize)]
struct FrontMatter {
    name: String,
    description: String,
}

/// Advertises available skills and returns their Markdown instructions on demand.
pub struct ActivateSkill {
    spec: ToolSpec,
    instructions: BTreeMap<String, String>,
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
                        entry.insert((metadata.description, body.to_owned()));
                    }
                    Entry::Occupied(entry) => {
                        bail!("duplicate skill '{}' in '{}'", entry.key(), path.display());
                    }
                }
            }
        }
        if skills.is_empty() {
            return Ok(None);
        }

        let mut description = String::from(
            "Load a skill's instructions. When a task matches a skill below, call this tool with its name before proceeding.\n\nAvailable skills:\n",
        );
        for (name, (summary, _)) in &skills {
            writeln!(description, "- {name}: {}", summary.trim())?;
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
        let instructions = skills
            .into_iter()
            .map(|(name, (_, body))| (name, body))
            .collect();
        Ok(Some(Self { spec, instructions }))
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
            Ok(match self.instructions.get(name) {
                Some(body) => ToolOutput::success(body.clone()),
                None => ToolOutput::error(format!("Unknown skill '{name}'")),
            })
        })
    }
}

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

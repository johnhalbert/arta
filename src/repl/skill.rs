//! A dollar-prefixed skill mention makes its Markdown body part of the system instructions on
//! later chat turns. The activation tool continues to advertise other discovered skills.

use super::parser::{Parser, Rule};
use anyhow::Result;
use arta::{skill::ActivateSkill, tool::Tool};
use std::sync::Arc;

/// Prefix used when registering skill mention tokens.
const DELIMITER: char = '$';

/// Name-ordered discovery catalog and a separate list of active names in first-mention order.
#[derive(Default)]
pub(crate) struct Skills {
    /// Optional shared catalog used to retrieve both descriptions and full instructions.
    catalog: Option<Arc<ActivateSkill>>,
    /// Unique mentioned names in first-activation order, retained until reset.
    active: Vec<String>,
}

/// Register a completion and parser rule per catalog entry. Start with no directly mentioned
/// skills.
pub(crate) fn register(parser: &mut Parser, catalog: Option<Arc<ActivateSkill>>) -> Skills {
    if let Some(catalog) = &catalog {
        for (name, description) in catalog.catalog() {
            let name = name.to_owned();
            parser.register(Rule::named(
                DELIMITER,
                format!("{DELIMITER}{name}"),
                description.into(),
                move |context, _, prompt| {
                    context.activate_skill(&name);
                    Some(prompt)
                },
            ));
        }
    }
    Skills {
        catalog,
        active: Vec::new(),
    }
}

impl Skills {
    /// Remember the name once, preserving the order in which skills were first mentioned.
    pub(super) fn activate(&mut self, name: &str) {
        if !self.active.iter().any(|active| active == name) {
            self.active.push(name.into());
        }
    }

    /// Remove direct activations while retaining the startup catalog.
    pub(super) fn clear(&mut self) {
        self.active.clear();
    }

    /// Join the bodies of active skills under individually named Markdown headings.
    pub(super) fn instructions(&self) -> String {
        let Some(catalog) = &self.catalog else {
            return String::new();
        };
        self.active
            .iter()
            .filter_map(|name| {
                catalog
                    .instructions(name)
                    .map(|body| format!("## Skill {name}\n\n{body}"))
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Reuse the full catalog when no skills are active, otherwise rebuild the tool with only
    /// inactive names. All-active and empty catalogs yield no tool.
    pub(super) fn activation_tool(&self) -> Result<Option<Arc<dyn Tool>>> {
        let Some(catalog) = &self.catalog else {
            return Ok(None);
        };
        if self.active.is_empty() {
            return Ok(Some(Arc::clone(catalog) as Arc<dyn Tool>));
        }
        Ok(catalog
            .excluding(|name| self.active.iter().any(|active| active == name))?
            .map(|tool| Arc::new(tool) as Arc<dyn Tool>))
    }
}

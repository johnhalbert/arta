use super::parser::{Parser, Rule};
use anyhow::Result;
use arta::{skill::ActivateSkill, tool::Tool};
use std::sync::Arc;

const DELIMITER: char = '$';

#[derive(Default)]
pub(crate) struct Skills {
    catalog: Option<Arc<ActivateSkill>>,
    active: Vec<String>,
}

pub(crate) fn register(parser: &mut Parser, catalog: Option<Arc<ActivateSkill>>) -> Skills {
    if let Some(catalog) = &catalog {
        for (name, description) in catalog.catalog() {
            let name = name.to_owned();
            parser.register(Rule::named(
                DELIMITER,
                format!("{DELIMITER}{name}"),
                description.into(),
                move |context, prompt| {
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
    pub(super) fn activate(&mut self, name: &str) {
        if !self.active.iter().any(|active| active == name) {
            self.active.push(name.into());
        }
    }

    pub(super) fn clear(&mut self) {
        self.active.clear();
    }

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

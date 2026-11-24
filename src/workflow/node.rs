use anyhow::{Result, ensure};
use serde::Deserialize;
use std::collections::HashSet;

/// One step in a workflow. A loop executes its children in order on each iteration.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Node {
    Agent { id: String, agent: String },
    Loop { id: String, nodes: Vec<Node> },
}

impl Node {
    pub(crate) fn visit_sequence<'a>(
        nodes: &'a [Self],
        ids: &mut HashSet<&'a str>,
        visit: &mut impl FnMut(&str, &str) -> Result<()>,
    ) -> Result<()> {
        ensure!(!nodes.is_empty(), "a workflow or loop must contain nodes");
        for node in nodes {
            let id = match node {
                Self::Agent { id, .. } | Self::Loop { id, .. } => id,
            };
            ensure!(!id.trim().is_empty(), "node id cannot be empty");
            ensure!(ids.insert(id), "duplicate node id '{id}'");
            match node {
                Self::Agent { agent, .. } => visit(id, agent)?,
                Self::Loop { nodes, .. } => Self::visit_sequence(nodes, ids, visit)?,
            }
        }
        Ok(())
    }
}

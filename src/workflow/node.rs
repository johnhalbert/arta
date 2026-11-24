//! Structural validation of the workflow tree uses a shared ID set at every depth. This makes
//! node IDs suitable keys for conversation histories.

use anyhow::{Result, ensure};
use serde::Deserialize;
use std::collections::HashSet;

/// One step in a workflow. A loop executes its children in order on each iteration.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Node {
    /// Invoke the named Markdown profile. The node ID identifies this leaf's conversation
    /// across visits.
    Agent {
        /// Unique key used for this node's history and progress events.
        id: String,
        /// Filename stem of the Markdown profile in the workflow directory.
        agent: String,
    },
    /// Repeat an ordered nonempty sequence until its deciding agent calls the break tool.
    Loop {
        /// Unique key identifying the loop in iteration paths and control calls.
        id: String,
        /// Children in execution order. The deepest final leaf decides whether to repeat.
        nodes: Vec<Node>,
    },
}

impl Node {
    /// Walk depth first, rejecting empty sequences, blank IDs, and duplicate IDs throughout
    /// the tree. Call the visitor once per agent leaf to resolve its profile.
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

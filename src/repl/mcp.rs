//! Builds the REPL tool selection from startup discovery. Profile-selected servers are always
//! visible. An @ mention adds the named server's handles until reset.

use super::parser::{Parser, Rule};
use arta::{mcp::McpTools, tool::Tool};
use std::{collections::BTreeMap, sync::Arc};

/// Prefix used when registering server mention tokens.
const DELIMITER: char = '@';

/// Deduplicated tool handles keyed by the names exposed to the model.
#[derive(Default)]
pub(crate) struct ToolSelection {
    /// Visible tools indexed by exported name to deduplicate repeated mentions.
    entries: BTreeMap<String, VisibleTool>,
}

/// Pairs a callable handle with its reset policy.
struct VisibleTool {
    /// Discovered handle passed to the next inference step.
    tool: Arc<dyn Tool>,
    /// Set for profile-selected tools so reset retains them.
    always_visible: bool,
}

impl ToolSelection {
    /// Keep the first entry for each name. Profile tools are installed before mention
    /// handlers run.
    fn insert(&mut self, tools: &[Arc<dyn Tool>], always_visible: bool) {
        for tool in tools {
            self.entries
                .entry(tool.spec().name.clone())
                .or_insert_with(|| VisibleTool {
                    tool: Arc::clone(tool),
                    always_visible,
                });
        }
    }

    /// Add the mentioned server's handles with conversation-scoped visibility.
    pub(super) fn expose(&mut self, tools: &[Arc<dyn Tool>]) {
        self.insert(tools, false);
    }

    /// Retain only handles exposed by the original agent profile.
    pub(super) fn clear_mentions(&mut self) {
        self.entries.retain(|_, entry| entry.always_visible);
    }

    /// Yield shared handles in name order for the current chat request.
    pub(super) fn tools(&self) -> impl Iterator<Item = Arc<dyn Tool>> + '_ {
        self.entries.values().map(|entry| Arc::clone(&entry.tool))
    }
}

/// Register one completion and parser rule per discovered server, preselecting those listed
/// by the agent.
pub(crate) fn register(
    parser: &mut Parser,
    servers: McpTools,
    always_visible: &[String],
) -> ToolSelection {
    let mut selection = ToolSelection::default();
    for (name, tools) in servers {
        if always_visible.contains(&name) {
            selection.insert(&tools, true);
        }
        parser.register(Rule::named(
            DELIMITER,
            format!("{DELIMITER}{name}"),
            "Enable MCP server tools".into(),
            move |context, _, prompt| {
                context.expose_mcp_tools(&tools);
                Some(prompt)
            },
        ));
    }
    selection
}

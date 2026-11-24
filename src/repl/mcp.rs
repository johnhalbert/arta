use super::parser::{Parser, Rule};
use arta::{mcp::McpTools, tool::Tool};
use std::{collections::BTreeMap, sync::Arc};

const DELIMITER: char = '@';

#[derive(Default)]
pub(crate) struct ToolSelection {
    entries: BTreeMap<String, VisibleTool>,
}

struct VisibleTool {
    tool: Arc<dyn Tool>,
    always_visible: bool,
}

impl ToolSelection {
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

    pub(super) fn expose(&mut self, tools: &[Arc<dyn Tool>]) {
        self.insert(tools, false);
    }

    pub(super) fn clear_mentions(&mut self) {
        self.entries.retain(|_, entry| entry.always_visible);
    }

    pub(super) fn tools(&self) -> impl Iterator<Item = Arc<dyn Tool>> + '_ {
        self.entries.values().map(|entry| Arc::clone(&entry.tool))
    }
}

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

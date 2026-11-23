use super::parser::{Parser, Rule};
use arta::mcp::McpTools;

const DELIMITER: char = '@';

pub(crate) fn register(parser: &mut Parser, servers: McpTools) {
    for (name, tools) in servers {
        parser.register(Rule::named(
            DELIMITER,
            format!("{DELIMITER}{name}"),
            "Enable MCP server tools".into(),
            move |context, prompt| {
                context.expose_mcp_tools(&tools);
                Some(prompt)
            },
        ));
    }
}

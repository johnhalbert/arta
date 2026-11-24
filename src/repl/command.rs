//! Collects slash commands from linked modules using inventory and converts each into a named
//! parser rule. The editor and parser consequently see the same command names.

mod exit;
mod model;
mod reset;
mod workflow;

use super::parser::{Parser, Rule, RuleContext};

/// Marks a slash-command token at a whitespace boundary.
const DELIMITER: char = '/';

/// A slash command registered by its implementation with `inventory::submit!`.
struct Command {
    /// Complete token including its leading slash, used for parsing and completion.
    name: &'static str,
    /// Help text shown beside this candidate in the editor.
    description: &'static str,
    /// Whether the parser consumes the next eligible token or quoted value.
    takes_argument: bool,
    /// Handler receiving the argument and residual prompt during submission parsing.
    run: fn(&mut RuleContext<'_, '_>, Option<&str>, String) -> Option<String>,
}

inventory::collect!(Command);

/// Registers the discovered slash commands when the REPL starts.
pub(crate) fn register(parser: &mut Parser) {
    for command in inventory::iter::<Command> {
        let rule = Rule::named(
            DELIMITER,
            command.name.into(),
            command.description.into(),
            command.run,
        );
        parser.register(if command.takes_argument {
            rule.with_argument()
        } else {
            rule
        });
    }
}

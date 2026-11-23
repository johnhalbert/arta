mod exit;
mod reset;

use super::parser::{Parser, Rule, RuleContext};

const DELIMITER: char = '/';

/// A slash command registered by its implementation with `inventory::submit!`.
struct Command {
    name: &'static str,
    description: &'static str,
    run: fn(&mut RuleContext<'_, '_>, String) -> Option<String>,
}

inventory::collect!(Command);

/// Registers the discovered slash commands when the REPL starts.
pub(crate) fn register(parser: &mut Parser) {
    for command in inventory::iter::<Command> {
        parser.register(Rule::named(
            DELIMITER,
            command.name.into(),
            command.description.into(),
            command.run,
        ));
    }
}

mod exit;
mod reset;

use super::parser::{Handler, Parser, Rule, RuleContext};

pub(super) const DELIMITER: char = '/';

/// A slash command registered by its implementation with `inventory::submit!`.
pub(crate) struct Command {
    pub(crate) name: &'static str,
    pub(crate) description: &'static str,
    run: Handler,
}

inventory::collect!(Command);

pub(super) fn registered() -> impl Iterator<Item = &'static Command> {
    inventory::iter::<Command>.into_iter()
}

/// Registers the discovered slash commands when the REPL starts.
pub(crate) fn register(parser: &mut Parser) {
    for command in registered() {
        parser.register(Rule {
            delimiter: DELIMITER,
            matcher: Box::new(move |token| token == command.name),
            handler: command.run,
        });
    }
}

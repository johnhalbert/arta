//! Registers `/reset` as an immediate conversation-state change.

use super::{Command, RuleContext};

inventory::submit! {
    Command {
        name: "/reset",
        description: "Clear conversation history",
        takes_argument: false,
        run: reset,
    }
}

/// Clear history, pending selection and mentions during parsing, then pass remaining input to
/// later handlers.
fn reset(context: &mut RuleContext<'_, '_>, _: Option<&str>, prompt: String) -> Option<String> {
    context.clear_history();
    Some(prompt)
}

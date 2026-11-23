use super::{Command, RuleContext};

inventory::submit! {
    Command {
        name: "/reset",
        description: "Clear conversation history",
        run: reset,
    }
}

fn reset(context: &mut RuleContext<'_, '_>, prompt: String) -> Option<String> {
    context.clear_history();
    Some(prompt)
}

use super::{Command, RuleContext};

inventory::submit! {
    Command {
        name: "/reset",
        description: "Clear conversation history",
        takes_argument: false,
        run: reset,
    }
}

fn reset(context: &mut RuleContext<'_, '_>, _: Option<&str>, prompt: String) -> Option<String> {
    context.clear_history();
    Some(prompt)
}

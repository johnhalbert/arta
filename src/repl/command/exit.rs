use super::{Command, RuleContext};

inventory::submit! {
    Command {
        name: "/exit",
        description: "Exit the REPL",
        run: exit,
    }
}

fn exit(context: &mut RuleContext<'_, '_>, _: String) -> Option<String> {
    context.exit();
    None
}

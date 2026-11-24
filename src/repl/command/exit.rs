use super::{Command, RuleContext};

inventory::submit! {
    Command {
        name: "/exit",
        description: "Exit the REPL",
        takes_argument: false,
        run: exit,
    }
}

fn exit(context: &mut RuleContext<'_, '_>, _: Option<&str>, _: String) -> Option<String> {
    context.exit();
    None
}

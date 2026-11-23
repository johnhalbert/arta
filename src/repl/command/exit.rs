use super::{Command, CommandContext};

inventory::submit! {
    Command {
        name: "/exit",
        description: "Exit the REPL",
        run: exit,
    }
}

fn exit(context: &mut CommandContext<'_, '_>, _: String) -> Option<String> {
    context.exit();
    None
}

use super::{Command, RuleContext};

inventory::submit! {
    Command {
        name: "/workflow",
        description: "Select or run a workflow",
        takes_argument: true,
        run: workflow,
    }
}

fn workflow(
    context: &mut RuleContext<'_, '_>,
    name: Option<&str>,
    prompt: String,
) -> Option<String> {
    context.request_workflow(name);
    Some(prompt)
}

//! Registers `/workflow` to select a definition for the next prompt.

use super::{Command, RuleContext};

inventory::submit! {
    Command {
        name: "/workflow",
        description: "Select or run a workflow",
        takes_argument: true,
        run: workflow,
    }
}

/// Record the optional workflow name and keep residual text so selection and execution can
/// share one submission.
fn workflow(
    context: &mut RuleContext<'_, '_>,
    name: Option<&str>,
    prompt: String,
) -> Option<String> {
    context.request_workflow(name);
    Some(prompt)
}

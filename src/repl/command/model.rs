//! Registers `/model` for querying or switching the current agent's model.

use super::{Command, RuleContext};

inventory::submit! {
    Command {
        name: "/model",
        description: "Switch or show the model",
        takes_argument: true,
        run: model,
    }
}

/// Queue the optional model ID and preserve other prompt text. The frontend applies the
/// switch before sending that text.
fn model(context: &mut RuleContext<'_, '_>, id: Option<&str>, prompt: String) -> Option<String> {
    context.request_model(id);
    Some(prompt)
}

use super::{Command, RuleContext};

inventory::submit! {
    Command {
        name: "/model",
        description: "Switch or show the model",
        takes_argument: true,
        run: model,
    }
}

fn model(context: &mut RuleContext<'_, '_>, id: Option<&str>, prompt: String) -> Option<String> {
    context.request_model(id);
    Some(prompt)
}

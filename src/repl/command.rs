mod exit;
mod reset;

use super::{Repl, Submission};

/// A slash command registered by its implementation with `inventory::submit!`.
pub(crate) struct Command {
    pub(crate) name: &'static str,
    pub(crate) description: &'static str,
    /// Receives the prompt with command tokens removed and any earlier edits applied.
    /// Returning `None` consumes the prompt. Later commands receive an empty string.
    run: fn(&mut CommandContext<'_, '_>, String) -> Option<String>,
}

inventory::collect!(Command);

pub(super) fn registered() -> impl Iterator<Item = &'static Command> {
    inventory::iter::<Command>.into_iter()
}

struct CommandContext<'repl, 'agent> {
    repl: &'repl mut Repl<'agent>,
    submission: Submission,
}

impl CommandContext<'_, '_> {
    fn clear_history(&mut self) {
        self.repl.history.clear();
        self.submission.history_cleared = true;
    }

    fn exit(&mut self) {
        self.submission.exit = true;
    }
}

pub(super) fn execute<'a>(
    repl: &mut Repl<'_>,
    mut prompt: String,
    commands: impl IntoIterator<Item = &'a Command>,
) -> Submission {
    let mut context = CommandContext {
        repl,
        submission: Submission::default(),
    };
    for command in commands {
        prompt = (command.run)(&mut context, prompt).unwrap_or_default();
    }
    if !context.submission.exit && !prompt.trim().is_empty() {
        context.submission.prompt = Some(prompt);
    }
    context.submission
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repl::tests::test_harness;

    #[test]
    fn handlers_receive_each_others_edits_and_can_consume_the_prompt() -> anyhow::Result<()> {
        let harness = test_harness()?;
        let mut repl = Repl::new(harness.agent("cli", "reviewer")?);
        let append = Command {
            name: "/append",
            description: "",
            run: |_, mut prompt| {
                prompt.push('!');
                Some(prompt)
            },
        };
        let uppercase = Command {
            name: "/uppercase",
            description: "",
            run: |_, prompt| {
                assert_eq!(prompt, "Hello!");
                Some(prompt.to_uppercase())
            },
        };
        let consume = Command {
            name: "/consume",
            description: "",
            run: |_, prompt| {
                assert_eq!(prompt, "HELLO!");
                None
            },
        };
        let submission = execute(&mut repl, "Hello".into(), [&append, &uppercase]);
        assert_eq!(submission.prompt.as_deref(), Some("HELLO!"));
        let submission = execute(&mut repl, "Hello".into(), [&append, &uppercase, &consume]);
        assert!(submission.prompt.is_none());
        let submission = execute(
            &mut repl,
            "Hello".into(),
            [&append, &uppercase, &consume, &append],
        );
        assert_eq!(submission.prompt.as_deref(), Some("!"));
        Ok(())
    }
}

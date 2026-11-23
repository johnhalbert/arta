use super::{Repl, Submission};

/// Receives the prompt with matched tokens removed and any earlier edits applied.
/// Returning `None` consumes the prompt. Later handlers receive an empty string.
pub(super) type Handler = fn(&mut RuleContext<'_, '_>, String) -> Option<String>;

pub(super) struct Rule {
    pub(super) delimiter: char,
    pub(super) matcher: Box<dyn Fn(&str) -> bool + Send + Sync>,
    pub(super) handler: Handler,
}

#[derive(Default)]
pub(crate) struct Parser {
    rules: Vec<Rule>,
}

impl Parser {
    /// Registers a rule. If multiple rules match a token, the first registered rule wins.
    pub(super) fn register(&mut self, rule: Rule) {
        self.rules.push(rule);
    }

    /// Matches whitespace-delimited tokens and applies their handlers in prompt order.
    pub(super) fn parse(&self, repl: &mut Repl<'_>, input: String) -> Submission {
        let mut matches = Vec::new();
        let mut prompt = String::new();
        let mut offset = 0;
        let mut copied = 0;
        for segment in input.split_inclusive(char::is_whitespace) {
            let start = offset;
            offset += segment.len();
            let token = segment.trim_end_matches(char::is_whitespace);
            if let Some(rule) = self
                .rules
                .iter()
                .find(|rule| token.starts_with(rule.delimiter) && (rule.matcher)(token))
            {
                matches.push(rule.handler);
                prompt.push_str(&input[copied..start]);
                copied = offset;
            }
        }
        let mut prompt = if matches.is_empty() {
            input
        } else {
            prompt.push_str(&input[copied..]);
            prompt.trim().into()
        };

        let mut context = RuleContext {
            repl,
            submission: Submission::default(),
        };
        for handler in matches {
            prompt = handler(&mut context, prompt).unwrap_or_default();
        }
        if !context.submission.exit && !prompt.trim().is_empty() {
            context.submission.prompt = Some(prompt);
        }
        context.submission
    }
}

pub(super) struct RuleContext<'repl, 'agent> {
    repl: &'repl mut Repl<'agent>,
    submission: Submission,
}

impl RuleContext<'_, '_> {
    pub(super) fn clear_history(&mut self) {
        self.repl.history.clear();
        self.submission.history_cleared = true;
    }

    pub(super) fn exit(&mut self) {
        self.submission.exit = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repl::{command, tests::test_harness};

    #[test]
    fn preserves_prompt_text_outside_matched_tokens() -> anyhow::Result<()> {
        let harness = test_harness()?;
        let mut parser = Parser::default();
        command::register(&mut parser);
        let mut repl = Repl::new(harness.agent("cli", "reviewer")?, &parser);

        let ordinary = "  /resetting path/exit /exit/path /reset. /unknown  ";
        let submission = repl.submit(ordinary.into());
        assert_eq!(submission.prompt.as_deref(), Some(ordinary));
        assert!(!submission.history_cleared);

        let submission = repl.submit("/reset café /reset 👋 /unknown".into());
        assert_eq!(submission.prompt.as_deref(), Some("café 👋 /unknown"));
        assert!(submission.history_cleared);
        Ok(())
    }

    #[test]
    fn handlers_for_multiple_delimiters_run_in_source_order() -> anyhow::Result<()> {
        let harness = test_harness()?;
        let mut parser = Parser::default();
        parser.register(Rule {
            delimiter: '@',
            matcher: Box::new(|token| token == "@uppercase"),
            handler: |_, prompt| {
                assert_eq!(prompt, "Hello!");
                Some(prompt.to_uppercase())
            },
        });
        parser.register(Rule {
            delimiter: '!',
            matcher: Box::new(|token| token == "!append"),
            handler: |_, mut prompt| {
                prompt.push('!');
                Some(prompt)
            },
        });
        parser.register(Rule {
            delimiter: '!',
            matcher: Box::new(|token| token == "!consume"),
            handler: |_, prompt| {
                assert_eq!(prompt, "HELLO!");
                None
            },
        });
        let mut repl = Repl::new(harness.agent("cli", "reviewer")?, &parser);

        let submission = repl.submit("!append Hello @uppercase".into());
        assert_eq!(submission.prompt.as_deref(), Some("HELLO!"));
        let submission = repl.submit("!append Hello @uppercase !consume".into());
        assert!(submission.prompt.is_none());
        let submission = repl.submit("!append Hello @uppercase !consume !append".into());
        assert_eq!(submission.prompt.as_deref(), Some("!"));
        Ok(())
    }
}

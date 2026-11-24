use super::suggestion::Candidate;
use super::{Repl, Submission};
use arta::tool::Tool;
use std::sync::Arc;

/// Receives the prompt with matched tokens removed and any earlier edits applied.
/// Returning `None` consumes the prompt. Later handlers receive an empty string.
type Handler =
    dyn Fn(&mut RuleContext<'_, '_>, Option<&str>, String) -> Option<String> + Send + Sync;

pub(super) struct Rule {
    delimiter: char,
    matcher: Box<dyn Fn(&str) -> bool + Send + Sync>,
    takes_argument: bool,
    handler: Box<Handler>,
    suggestion: Option<Candidate>,
}

impl Rule {
    pub(super) fn new(
        delimiter: char,
        matcher: impl Fn(&str) -> bool + Send + Sync + 'static,
        handler: impl Fn(&mut RuleContext<'_, '_>, Option<&str>, String) -> Option<String>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self {
            delimiter,
            matcher: Box::new(matcher),
            takes_argument: false,
            handler: Box::new(handler),
            suggestion: None,
        }
    }

    pub(super) fn named(
        delimiter: char,
        name: String,
        description: String,
        handler: impl Fn(&mut RuleContext<'_, '_>, Option<&str>, String) -> Option<String>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        let token = name.clone();
        Self {
            suggestion: Some(Candidate { name, description }),
            ..Self::new(delimiter, move |input| input == token, handler)
        }
    }

    pub(super) fn with_argument(mut self) -> Self {
        self.takes_argument = true;
        self
    }
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

    pub(crate) fn suggestions(&self) -> impl Iterator<Item = &Candidate> {
        self.rules
            .iter()
            .filter_map(|rule| rule.suggestion.as_ref())
    }

    fn rule_for(&self, token: &str) -> Option<&Rule> {
        self.rules
            .iter()
            .find(|rule| token.starts_with(rule.delimiter) && (rule.matcher)(token))
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
            if start < copied {
                continue;
            }
            let token = segment.trim_end_matches(char::is_whitespace);
            if let Some(rule) = self.rule_for(token) {
                prompt.push_str(&input[copied..start]);
                copied = offset;
                let mut argument = None;
                if rule.takes_argument {
                    let tail = input[offset..].trim_start();
                    if let Some((value, rest)) = tail
                        .strip_prefix('"')
                        .and_then(|quoted| quoted.split_once('"'))
                        .filter(|(_, rest)| {
                            rest.is_empty() || rest.starts_with(char::is_whitespace)
                        })
                    {
                        argument = Some(value);
                        copied = input.len() - rest.trim_start().len();
                    } else if let Some(token) = tail.split_whitespace().next()
                        && self.rule_for(token).is_none()
                    {
                        argument = Some(token);
                        copied = input.len() - tail[token.len()..].trim_start().len();
                    }
                }
                matches.push((rule.handler.as_ref(), argument));
            }
        }
        if matches.is_empty() {
            return Submission {
                prompt: (!input.trim().is_empty()).then_some(input),
                ..Submission::default()
            };
        }
        prompt.push_str(&input[copied..]);
        let mut prompt = prompt.trim().into();

        let mut context = RuleContext {
            repl,
            submission: Submission::default(),
        };
        for (handler, argument) in matches {
            prompt = handler(&mut context, argument, prompt).unwrap_or_default();
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
    pub(super) fn request_model(&mut self, id: Option<&str>) {
        self.submission.model_requests.push(id.map(str::to_owned));
    }

    pub(super) fn clear_history(&mut self) {
        self.repl.history.clear();
        self.repl.skills.clear();
        self.repl.mcp_tools.clear_mentions();
        self.submission.history_cleared = true;
    }

    pub(super) fn activate_skill(&mut self, name: &str) {
        self.repl.skills.activate(name);
    }

    pub(super) fn expose_mcp_tools(&mut self, tools: &[Arc<dyn Tool>]) {
        self.repl.mcp_tools.expose(tools);
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
        let mut harness = test_harness()?;
        let mut parser = Parser::default();
        command::register(&mut parser);
        let mut repl = Repl::new(
            harness.agent_mut("cli", "reviewer")?,
            &parser,
            Default::default(),
            Default::default(),
        );

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
        let mut harness = test_harness()?;
        let mut parser = Parser::default();
        parser.register(Rule::new(
            '@',
            |token| token == "@uppercase",
            |_, _, prompt| {
                assert_eq!(prompt, "Hello!");
                Some(prompt.to_uppercase())
            },
        ));
        parser.register(Rule::new(
            '!',
            |token| token == "!append",
            |_, _, mut prompt| {
                prompt.push('!');
                Some(prompt)
            },
        ));
        parser.register(Rule::new(
            '!',
            |token| token == "!consume",
            |_, _, prompt| {
                assert_eq!(prompt, "HELLO!");
                None
            },
        ));
        let mut repl = Repl::new(
            harness.agent_mut("cli", "reviewer")?,
            &parser,
            Default::default(),
            Default::default(),
        );

        let submission = repl.submit("!append Hello @uppercase".into());
        assert_eq!(submission.prompt.as_deref(), Some("HELLO!"));
        let submission = repl.submit("!append Hello @uppercase !consume".into());
        assert!(submission.prompt.is_none());
        let submission = repl.submit("!append Hello @uppercase !consume !append".into());
        assert_eq!(submission.prompt.as_deref(), Some("!"));
        Ok(())
    }
}

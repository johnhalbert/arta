pub(super) mod command;
pub(super) mod parser;
pub(super) mod suggestion;

use super::markdown::TerminalOutput;
use anyhow::{Context, Result};
use arta::{
    agent::Agent,
    provider::Turn,
    stream::{Message, Stream},
};
use futures_util::{TryStreamExt, stream};
use parser::Parser;
use std::io::IsTerminal;
use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt, BufReader};

#[derive(Default)]
pub(super) struct Submission {
    pub(super) prompt: Option<String>,
    pub(super) exit: bool,
    pub(super) history_cleared: bool,
}

/// Owns the conversation shared by the terminal and line-based interfaces.
pub(super) struct Repl<'a> {
    agent: &'a Agent,
    parser: &'a Parser,
    history: Vec<Turn>,
}

impl<'a> Repl<'a> {
    pub(super) fn new(agent: &'a Agent, parser: &'a Parser) -> Self {
        Self {
            agent,
            parser,
            history: Vec::new(),
        }
    }

    pub(super) fn submit(&mut self, input: String) -> Submission {
        self.parser.parse(self, input)
    }

    pub(super) fn history(&self) -> &[Turn] {
        &self.history
    }

    pub(super) fn reply<'b>(&'b self, prompt: &'b str) -> Stream<'b> {
        Stream::new(
            stream::once(async {
                self.agent
                    .chat(&self.history, prompt)
                    .await
                    .map(Message::into_stream)
            })
            .try_flatten(),
        )
    }

    pub(super) fn complete(&mut self, turn: Turn) {
        self.history.push(turn);
    }

    pub(super) async fn run_lines(&mut self) -> Result<()> {
        let interactive = std::io::stdin().is_terminal();
        let mut input = BufReader::new(io::stdin()).lines();
        let mut stdout = io::stdout();
        let mut stderr = io::stderr();
        if interactive {
            stderr
                .write_all(b"Enter a prompt, /reset to clear the conversation, or /exit to quit.\n")
                .await?;
        }
        loop {
            if interactive {
                stderr.write_all(b"> ").await?;
                stderr.flush().await?;
            }
            let Some(line) = input
                .next_line()
                .await
                .context("failed to read UTF-8 prompt from stdin")?
            else {
                break;
            };
            let submission = self.submit(line);
            if submission.exit {
                break;
            }
            if let Some(prompt) = submission.prompt {
                let mut reply = String::new();
                let terminal = TerminalOutput::detect();
                let result =
                    super::write_reply(self.reply(&prompt), &mut stdout, terminal, |chunk| {
                        reply.push_str(chunk);
                    })
                    .await;
                match result {
                    Ok(()) => self.complete(Turn { prompt, reply }),
                    Err(error) if !interactive => return Err(error),
                    Err(error) => {
                        if terminal.is_none() && !reply.is_empty() && !reply.ends_with('\n') {
                            stdout.write_all(b"\n").await?;
                            stdout.flush().await?;
                        }
                        stderr
                            .write_all(format!("error: {error:#}\n").as_bytes())
                            .await?;
                        stderr.flush().await?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arta::{
        agent::AgentOptions,
        config::Config,
        harness::Harness,
        model::{Model, ModelId},
        provider::Provider,
        stream::Frame,
    };
    use futures_util::future::BoxFuture;
    use std::sync::Arc;

    struct TestProvider;

    impl Model for TestProvider {
        fn id(&self) -> &str {
            "test"
        }

        fn trained_context_length(&self) -> u32 {
            4096
        }
    }

    impl Provider for TestProvider {
        fn model(&self, _: &str) -> Result<Arc<dyn Model>> {
            Ok(Arc::new(Self))
        }

        fn send<'a>(&'a self, _: &'a str, _: &'a str) -> BoxFuture<'a, Result<Message>> {
            Box::pin(async { anyhow::bail!("expected the agent's system prompt") })
        }

        fn chat_with_system<'a>(
            &'a self,
            _: &'a str,
            system_prompt: Option<&'a str>,
            history: &'a [Turn],
            _: &'a str,
        ) -> BoxFuture<'a, Result<Message>> {
            assert_eq!(system_prompt, Some("Use short answers."));
            let reply = history.len().to_string();
            Box::pin(async { Ok(Message::new(stream::iter([Ok(reply)]))) })
        }
    }

    pub(super) fn test_harness() -> Result<Harness> {
        let mut harness = Harness::new(&Config::default())?;
        harness.register_provider("test", || Ok(Arc::new(TestProvider)))?;
        harness.create_session_with_agents(
            "cli",
            [(
                "reviewer",
                ModelId {
                    provider: "test",
                    model: "test",
                },
                AgentOptions {
                    system_prompt: Some("Use short answers."),
                    ..Default::default()
                },
            )],
        )?;
        Ok(harness)
    }

    #[tokio::test]
    async fn reset_clears_history_and_keeps_the_system_prompt() -> Result<()> {
        let harness = test_harness()?;
        let mut parser = Parser::default();
        command::register(&mut parser);
        let mut repl = Repl::new(harness.agent("cli", "reviewer")?, &parser);
        repl.complete(Turn {
            prompt: "Hello".into(),
            reply: "Hi".into(),
        });
        let reset = repl.submit("/reset".into());
        assert!(reset.history_cleared);
        assert!(reset.prompt.is_none());
        assert!(!reset.exit);
        repl.complete(Turn {
            prompt: "A new turn".into(),
            reply: "Hi again".into(),
        });
        let reset_and_prompt = repl.submit("Hello /reset again /reset".into());
        assert!(reset_and_prompt.history_cleared);
        let prompt = reset_and_prompt.prompt.as_deref().unwrap();
        assert_eq!(prompt, "Hello again");
        assert_eq!(
            repl.reply(prompt).try_collect::<Vec<_>>().await?,
            vec![Frame::Begin, Frame::Chunk("0".into()), Frame::End]
        );
        repl.complete(Turn {
            prompt: prompt.into(),
            reply: "0".into(),
        });
        let exit = repl.submit("Text /exit /reset".into());
        assert!(exit.exit);
        assert!(exit.prompt.is_none());
        assert!(exit.history_cleared);
        assert!(repl.history().is_empty());
        Ok(())
    }
}

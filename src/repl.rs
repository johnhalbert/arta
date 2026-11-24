pub(super) mod command;
pub(super) mod mcp;
pub(super) mod parser;
pub(super) mod skill;
pub(super) mod suggestion;

use super::markdown::TerminalOutput;
use anyhow::{Context, Result};
use arta::{
    agent::Agent,
    provider::{PromptObserver, Turn},
    stream::{Message, Stream},
    tool::Tool,
};
use futures_util::{TryStreamExt, stream};
use parser::Parser;
use std::{collections::BTreeMap, io::IsTerminal, sync::Arc};
use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt, BufReader};

#[derive(Default)]
pub(super) struct Submission {
    pub(super) prompt: Option<String>,
    pub(super) exit: bool,
    pub(super) history_cleared: bool,
    pub(super) model_requests: Vec<Option<String>>,
}

/// Owns the conversation shared by the terminal and line-based interfaces.
pub(super) struct Repl<'a> {
    agent: &'a mut Agent,
    parser: &'a Parser,
    history: Vec<Turn>,
    skills: skill::Skills,
    mcp_tools: BTreeMap<String, Arc<dyn Tool>>,
}

impl<'a> Repl<'a> {
    pub(super) fn new(agent: &'a mut Agent, parser: &'a Parser, skills: skill::Skills) -> Self {
        Self {
            agent,
            parser,
            history: Vec::new(),
            skills,
            mcp_tools: BTreeMap::new(),
        }
    }

    pub(super) fn submit(&mut self, input: String) -> Submission {
        self.parser.parse(self, input)
    }

    pub(super) fn parser(&self) -> &'a Parser {
        self.parser
    }

    pub(super) async fn apply_model_requests(
        &mut self,
        requests: Vec<Option<String>>,
    ) -> Result<Option<String>> {
        let mut status = None;
        for request in requests {
            let label = if let Some(id) = request {
                self.agent
                    .switch_model(&id)
                    .await
                    .with_context(|| format!("failed to switch to model '{id}'"))?;
                "Switched to model"
            } else {
                "Current model is"
            };
            status = Some(format!("{label} '{}'", self.agent.model().id()));
        }
        Ok(status)
    }

    pub(super) fn history(&self) -> &[Turn] {
        &self.history
    }

    pub(super) fn reply<'b>(
        &'b self,
        prompt: &'b str,
        observer: Option<PromptObserver>,
    ) -> Stream<'b> {
        Stream::new(
            stream::once(async {
                let instructions = self.skills.instructions();
                let tools = self
                    .skills
                    .activation_tool()?
                    .into_iter()
                    .chain(self.mcp_tools.values().cloned());
                self.agent
                    .chat_with_context_observed(
                        &self.history,
                        prompt,
                        &instructions,
                        tools,
                        observer,
                    )
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
                .write_all(b"Enter a prompt, /model <id> to switch models, /reset to clear the conversation, or /exit to quit.\n")
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
            match self.apply_model_requests(submission.model_requests).await {
                Ok(Some(status)) => {
                    stderr.write_all(format!("{status}\n").as_bytes()).await?;
                    stderr.flush().await?;
                }
                Ok(None) => {}
                Err(error) if !interactive => return Err(error),
                Err(error) => {
                    stderr
                        .write_all(format!("error: {error:#}\n").as_bytes())
                        .await?;
                    stderr.flush().await?;
                    continue;
                }
            }
            if let Some(prompt) = submission.prompt {
                let mut reply = String::new();
                let terminal = TerminalOutput::detect();
                let result =
                    super::write_reply(self.reply(&prompt, None), &mut stdout, terminal, |chunk| {
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
        provider::{
            Provider,
            tool::{ToolAdapter, ToolRequest, ToolResponse},
        },
        skill::ActivateSkill,
        stream::Frame,
        tool::{ToolArguments, ToolOutput, ToolSpec},
    };
    use futures_util::future::BoxFuture;
    use std::path::PathBuf;

    struct TestProvider(&'static str);

    impl Model for TestProvider {
        fn id(&self) -> &str {
            self.0
        }

        fn trained_context_length(&self) -> u32 {
            4096
        }
    }

    impl Provider for TestProvider {
        fn model(&self, id: &str) -> Result<Arc<dyn Model>> {
            match id {
                "test" => Ok(Arc::new(Self("test"))),
                "other model" => Ok(Arc::new(Self("other model"))),
                _ => anyhow::bail!("unknown model '{id}'"),
            }
        }

        fn switch_model(&self, _: &str, id: &str) -> Result<Arc<dyn Model>> {
            self.model(id)
        }

        fn send<'a>(&'a self, _: &'a str, _: &'a str) -> BoxFuture<'a, Result<Message>> {
            Box::pin(async { anyhow::bail!("expected the agent's system prompt") })
        }

        fn tool_adapter(&self) -> Option<&dyn ToolAdapter> {
            Some(self)
        }

        fn chat_with_system<'a>(
            &'a self,
            model_id: &'a str,
            system_prompt: Option<&'a str>,
            history: &'a [Turn],
            _: &'a str,
        ) -> BoxFuture<'a, Result<Message>> {
            assert_eq!(system_prompt, Some("Use short answers."));
            let reply = if model_id == "test" {
                history.len().to_string()
            } else {
                format!("{model_id}: {}", history.len())
            };
            Box::pin(async { Ok(Message::new(stream::iter([Ok(reply)]))) })
        }
    }

    impl ToolAdapter for TestProvider {
        fn infer<'a>(&'a self, request: ToolRequest<'a>) -> BoxFuture<'a, Result<ToolResponse>> {
            Box::pin(async move {
                let tools: Vec<_> = request
                    .tools
                    .iter()
                    .map(|tool| tool.name.as_str())
                    .collect();
                Ok(ToolResponse::Reply(format!(
                    "{}\n{}\n{}",
                    request.prompt,
                    request.system_prompt.unwrap_or_default().trim_end(),
                    tools.join(","),
                )))
            })
        }
    }

    struct TestMcpTool(ToolSpec);

    impl Tool for TestMcpTool {
        fn spec(&self) -> &ToolSpec {
            &self.0
        }

        fn call(&self, _: ToolArguments) -> BoxFuture<'_, Result<ToolOutput>> {
            Box::pin(async { anyhow::bail!("the test only checks tool exposure") })
        }
    }

    pub(super) fn test_harness() -> Result<Harness> {
        let mut harness = Harness::new(&Config::default())?;
        harness.register_provider("test", || Ok(Arc::new(TestProvider("test"))))?;
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
    async fn model_command_switches_before_the_prompt_and_keeps_history() -> Result<()> {
        let mut harness = test_harness()?;
        let mut parser = Parser::default();
        command::register(&mut parser);
        let mut repl = Repl::new(
            harness.agent_mut("cli", "reviewer")?,
            &parser,
            Default::default(),
        );
        repl.complete(Turn {
            prompt: "First prompt".into(),
            reply: "First reply".into(),
        });

        let submission = repl.submit("/model \"other model\" Continue".into());
        assert_eq!(submission.prompt.as_deref(), Some("Continue"));
        assert_eq!(
            repl.apply_model_requests(submission.model_requests)
                .await?
                .as_deref(),
            Some("Switched to model 'other model'")
        );
        assert_eq!(repl.agent.model().id(), "other model");
        assert_eq!(
            repl.reply("Continue", None).try_collect::<Vec<_>>().await?,
            vec![
                Frame::Begin,
                Frame::Chunk("other model: 1".into()),
                Frame::End
            ]
        );

        let invalid = repl.submit("/model missing".into());
        assert!(
            repl.apply_model_requests(invalid.model_requests)
                .await
                .is_err()
        );
        let query = repl.submit("/model".into());
        assert_eq!(
            repl.apply_model_requests(query.model_requests)
                .await?
                .as_deref(),
            Some("Current model is 'other model'")
        );
        Ok(())
    }

    #[tokio::test]
    async fn reset_clears_history_and_keeps_the_system_prompt() -> Result<()> {
        let mut harness = test_harness()?;
        let mut parser = Parser::default();
        command::register(&mut parser);
        let mut repl = Repl::new(
            harness.agent_mut("cli", "reviewer")?,
            &parser,
            Default::default(),
        );
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
            repl.reply(prompt, None).try_collect::<Vec<_>>().await?,
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

    #[tokio::test]
    async fn mentions_complete_and_add_conversation_context_once() -> Result<()> {
        let mut harness = test_harness()?;
        let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/skills");
        let mut parser = Parser::default();
        command::register(&mut parser);
        let skills = skill::register(
            &mut parser,
            ActivateSkill::discover([directory])?.map(Arc::new),
        );
        let server = |name: &str| -> Vec<Arc<dyn Tool>> {
            vec![Arc::new(TestMcpTool(ToolSpec {
                name: format!("{name}__lookup"),
                description: "Look up notes".into(),
                input_schema: Default::default(),
            }))]
        };
        mcp::register(
            &mut parser,
            BTreeMap::from([
                ("notes".into(), server("notes")),
                ("other".into(), server("other")),
            ]),
        );

        let candidates: Vec<_> = parser.suggestions().collect();
        for (prefix, expected) in [("$rev", "Use $review-notes "), ("@no", "Use @notes ")] {
            let mut input = format!("Use {prefix}");
            let cursor = input.len();
            assert_eq!(
                suggestion::complete(&candidates, &mut input, cursor, 0),
                Some(expected.len())
            );
            assert_eq!(input, expected);
        }

        let mut repl = Repl::new(harness.agent_mut("cli", "reviewer")?, &parser, skills);
        let plain = repl.reply("Hello", None).try_collect::<Vec<_>>().await?;
        assert_eq!(
            plain,
            vec![
                Frame::Begin,
                Frame::Chunk("Hello\nUse short answers.\nactivate_skill".into()),
                Frame::End
            ]
        );

        assert!(repl.submit("$review-notes @notes".into()).prompt.is_none());
        let active_prompt = "Use short answers.\n\n## Skill review-notes\n\nCompare each claim with the source notes.";
        for input in [
            "Check notes",
            "$review-notes Check notes @notes @notes $review-notes",
            "Check notes $review-notes @notes",
        ] {
            let submission = repl.submit(input.into());
            let prompt = submission.prompt.as_deref().unwrap();
            assert_eq!(prompt, "Check notes");
            let reply = repl.reply(prompt, None).try_collect::<Vec<_>>().await?;
            assert_eq!(
                reply,
                vec![
                    Frame::Begin,
                    Frame::Chunk(format!("Check notes\n{active_prompt}\nnotes__lookup")),
                    Frame::End,
                ]
            );
        }

        assert!(repl.submit("/reset".into()).history_cleared);
        assert_eq!(
            repl.reply("Hello", None).try_collect::<Vec<_>>().await?,
            plain
        );
        Ok(())
    }
}

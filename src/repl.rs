use anyhow::{Context, Result};
use arta::{
    agent::Agent,
    provider::Turn,
    stream::{Message, Stream},
};
use futures_util::{TryStreamExt, stream};
use std::io::IsTerminal;
use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt, BufReader};

pub(super) enum Action {
    Ignore,
    Exit,
    Reset,
    Prompt(String),
}

/// Owns the conversation shared by the terminal and line-based interfaces.
pub(super) struct Repl<'a> {
    agent: &'a Agent,
    history: Vec<Turn>,
}

impl<'a> Repl<'a> {
    pub(super) fn new(agent: &'a Agent) -> Self {
        Self {
            agent,
            history: Vec::new(),
        }
    }

    pub(super) fn submit(&mut self, input: String) -> Action {
        match input.trim() {
            "" => Action::Ignore,
            "/exit" => Action::Exit,
            "/reset" => {
                self.history.clear();
                Action::Reset
            }
            _ => Action::Prompt(input),
        }
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
            match self.submit(line) {
                Action::Ignore | Action::Reset => {}
                Action::Exit => break,
                Action::Prompt(prompt) => {
                    let mut reply = String::new();
                    let result = super::write_reply(self.reply(&prompt), &mut stdout, |chunk| {
                        reply.push_str(chunk)
                    })
                    .await;
                    match result {
                        Ok(()) => self.complete(Turn { prompt, reply }),
                        Err(error) if !interactive => return Err(error),
                        Err(error) => {
                            if !reply.is_empty() && !reply.ends_with('\n') {
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
        }
        Ok(())
    }
}

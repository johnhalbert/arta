mod markdown;
mod repl;
mod tui;

use anyhow::{Context, Result, ensure};
use arta::{
    config::{self, AgentProfile, StartupConfig},
    harness::Harness,
    model::ModelId,
    provider::llama_cpp::{LlamaCPP, LlamaCPPConfig},
    skill::ActivateSkill,
    stream::{Frame, Message, Stream},
};
use futures_util::{StreamExt, stream};
use std::{io::IsTerminal, process::ExitCode, sync::Arc};
use tokio::{
    io::{self, AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader},
    task::spawn_blocking,
};

async fn run_once(startup: &'static StartupConfig, prompt: &str) -> Result<()> {
    let input = if prompt == "-" {
        stdin_message(BufReader::new(io::stdin())).await?
    } else {
        ensure!(!prompt.trim().is_empty(), "prompt cannot be empty");
        Message::new(stream::iter([Ok(prompt.to_owned())]))
    };

    let harness = cli_harness(startup, false).await?;
    let replies = harness
        .session("cli")?
        .stream(startup.agent_name(), input.into_stream())?;
    write_reply(
        replies,
        &mut io::stdout(),
        markdown::TerminalOutput::detect(),
        |_| {},
    )
    .await
}

async fn cli_harness(startup: &'static StartupConfig, fullscreen: bool) -> Result<Harness> {
    let config = &startup.settings;
    let mut harness = spawn_blocking(move || -> Result<Harness> {
        let mut quiet_config;
        let config = if fullscreen {
            // Native diagnostics would write over the alternate screen during inference.
            let mut llama: LlamaCPPConfig = config.provider_settings(LlamaCPP::ID)?;
            llama.verbose = false;
            quiet_config = config.clone();
            quiet_config.set_provider_settings(LlamaCPP::ID, llama)?;
            &quiet_config
        } else {
            config
        };
        let mut harness = Harness::new(config)?;
        let directories = config
            .skills
            .directories
            .iter()
            .map(|path| config.resolve_path(path));
        if let Some(tool) = ActivateSkill::discover(directories)? {
            harness.register_tool(Arc::new(tool))?;
        }
        Ok(harness)
    })
    .await
    .context("failed to start the CLI harness")??;
    let options = startup
        .agent
        .as_ref()
        .map(AgentProfile::options)
        .unwrap_or_default();
    harness.connect_mcp_in(config, options.workspace).await?;
    let model_id = ModelId {
        provider: &config.app.provider,
        model: &config.app.model,
    };
    harness.create_session_with_agents("cli", [(startup.agent_name(), model_id, options)])?;
    Ok(harness)
}

async fn run_conversation(startup: &'static StartupConfig) -> Result<()> {
    let fullscreen = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
    let harness = cli_harness(startup, fullscreen)
        .await
        .context("failed to start the REPL")?;
    let mut repl = repl::Repl::new(harness.agent("cli", startup.agent_name())?);
    if fullscreen {
        tui::run(&mut repl).await
    } else {
        repl.run_lines().await
    }
}

async fn write_reply(
    mut replies: Stream<'_>,
    stdout: &mut (impl AsyncWrite + Unpin),
    terminal: Option<markdown::TerminalOutput>,
    mut capture: impl FnMut(&str),
) -> Result<()> {
    let mut needs_newline = false;
    // Render a whole message so later lines can resolve earlier Markdown syntax.
    let mut markdown = String::new();
    while let Some(frame) = replies.next().await {
        match frame {
            Ok(Frame::Begin) => needs_newline = false,
            Ok(Frame::Chunk(text)) if !text.is_empty() => {
                if terminal.is_some() {
                    markdown.push_str(&text);
                } else {
                    stdout.write_all(text.as_bytes()).await?;
                    stdout.flush().await?;
                    needs_newline = !text.ends_with('\n');
                }
                capture(&text);
            }
            Ok(Frame::End) | Err(_) => {
                if let Some(terminal) = terminal {
                    stdout
                        .write_all(terminal.format(&markdown).as_bytes())
                        .await?;
                    markdown.clear();
                } else if frame.is_ok() && needs_newline {
                    stdout.write_all(b"\n").await?;
                }
                stdout.flush().await?;
                frame?;
            }
            _ => {}
        }
    }
    stdout.flush().await?;
    Ok(())
}

async fn stdin_message(mut input: impl AsyncBufRead + Unpin + Send + 'static) -> Result<Message> {
    // Check for a non-blank prompt before opening the model.
    let mut prefix = String::new();
    loop {
        let start = prefix.len();
        let bytes = input
            .read_line(&mut prefix)
            .await
            .context("failed to read UTF-8 prompt from stdin")?;
        ensure!(bytes != 0, "prompt cannot be empty");
        if !prefix[start..].trim().is_empty() {
            break;
        }
    }

    let rest = stream::try_unfold(input, |mut input| async move {
        let mut line = String::new();
        let bytes = input
            .read_line(&mut line)
            .await
            .context("failed to read UTF-8 prompt from stdin")?;
        Ok((bytes != 0).then_some((line, input)))
    });
    Ok(Message::new(stream::iter([Ok(prefix)]).chain(rest)))
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    let config = match config::get() {
        Ok(config) => config,
        Err(error) => return report_error(error),
    };
    let result = match config.prompt.as_deref() {
        Some(prompt) => run_once(config, prompt).await,
        None => run_conversation(config).await,
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => report_error(&error),
    }
}

fn report_error(error: &anyhow::Error) -> ExitCode {
    eprintln!("error: {error:#}");
    ExitCode::FAILURE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn terminal_reply_renders_markdown_across_chunk_boundaries() -> Result<()> {
        let chunks = [
            Ok("# Heading\n\n**bo".into()),
            Ok("ld** and [docs][ref]\n\n[ref]: https://example.com".into()),
        ];
        let mut output = Vec::new();
        let mut captured = String::new();
        write_reply(
            Message::new(stream::iter(chunks)).into_stream(),
            &mut output,
            Some(markdown::TerminalOutput {
                width: 80,
                ansi: false,
            }),
            |chunk| captured.push_str(chunk),
        )
        .await?;

        let output = String::from_utf8(output)?;
        assert_eq!(output, "Heading\n\nbold and docs (https://example.com)\n");
        assert_eq!(
            captured,
            "# Heading\n\n**bold** and [docs][ref]\n\n[ref]: https://example.com"
        );
        Ok(())
    }

    #[tokio::test]
    async fn redirected_reply_writes_original_markdown() -> Result<()> {
        let chunks = [Ok("# Heading\n\n".into()), Ok("**bold**".into())];
        let mut output = Vec::new();
        write_reply(
            Message::new(stream::iter(chunks)).into_stream(),
            &mut output,
            None,
            |_| {},
        )
        .await?;
        assert_eq!(output, b"# Heading\n\n**bold**\n");
        Ok(())
    }
}

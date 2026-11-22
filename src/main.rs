mod repl;
mod tui;

use anyhow::{Context, Result, ensure};
use arta::{
    config::{self, Config},
    harness::Harness,
    model::ModelId,
    provider::llama_cpp::{LlamaCPP, LlamaCPPConfig},
    stream::{Frame, Message, Stream},
};
use futures_util::{StreamExt, stream};
use std::{io::IsTerminal, process::ExitCode};
use tokio::{
    io::{self, AsyncBufRead, AsyncBufReadExt, AsyncWriteExt, BufReader},
    task::spawn_blocking,
};

async fn run_once(config: &'static Config, prompt: &str) -> Result<()> {
    let input = if prompt == "-" {
        stdin_message(BufReader::new(io::stdin())).await?
    } else {
        ensure!(!prompt.trim().is_empty(), "prompt cannot be empty");
        Message::new(stream::iter([Ok(prompt.to_owned())]))
    };

    let harness = cli_harness(config, false).await?;
    let replies = harness
        .session("cli")?
        .stream("default", input.into_stream())?;
    write_reply(replies, &mut io::stdout(), |_| {}).await
}

async fn cli_harness(config: &'static Config, fullscreen: bool) -> Result<Harness> {
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
        Harness::new(config)
    })
    .await
    .context("failed to start the CLI harness")??;
    harness.connect_mcp(config).await?;
    let model_id = ModelId {
        provider: &config.app.provider,
        model: &config.app.model,
    };
    harness.create_session("cli", &[("default", model_id)])?;
    Ok(harness)
}

async fn run_conversation(config: &'static Config) -> Result<()> {
    let fullscreen = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
    let harness = cli_harness(config, fullscreen)
        .await
        .context("failed to start the REPL with the model selected in config.toml")?;
    let mut repl = repl::Repl::new(harness.agent("cli", "default")?);
    if fullscreen {
        tui::run(&mut repl).await
    } else {
        repl.run_lines().await
    }
}

async fn write_reply(
    mut replies: Stream<'_>,
    stdout: &mut io::Stdout,
    mut capture: impl FnMut(&str),
) -> Result<()> {
    let mut needs_newline = false;
    while let Some(frame) = replies.next().await {
        match frame? {
            Frame::Begin => needs_newline = false,
            Frame::Chunk(text) if !text.is_empty() => {
                stdout.write_all(text.as_bytes()).await?;
                stdout.flush().await?;
                capture(&text);
                needs_newline = !text.ends_with('\n');
            }
            Frame::End if needs_newline => stdout.write_all(b"\n").await?,
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
        Some(prompt) => run_once(&config.settings, prompt).await,
        None => run_conversation(&config.settings).await,
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

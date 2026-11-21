use anyhow::{Context, Result, ensure};
use arta::{
    config::{self, StartupConfig},
    harness::Harness,
    model::ModelId,
    stream::{Frame, Message},
};
use futures_util::{StreamExt, stream};
use std::process::ExitCode;
use tokio::{
    io::{self, AsyncBufRead, AsyncBufReadExt, AsyncWriteExt, BufReader},
    task::spawn_blocking,
};

async fn run(config: &'static StartupConfig) -> Result<()> {
    let input = if config.prompt == "-" {
        stdin_message(BufReader::new(io::stdin())).await?
    } else {
        ensure!(!config.prompt.trim().is_empty(), "prompt cannot be empty");
        Message::new(stream::iter([Ok(config.prompt.clone())]))
    };

    let harness = spawn_blocking(move || -> Result<Harness> {
        let mut harness = Harness::new(&config.settings)?;
        let model_id = ModelId {
            provider: &config.settings.app.provider,
            model: &config.settings.app.model,
        };
        harness.create_session("cli", &[("default", model_id)])?;
        Ok(harness)
    })
    .await
    .context("failed to start the CLI session")??;
    let mut replies = harness
        .session("cli")?
        .stream("default", input.into_stream())?;

    let mut stdout = io::stdout();
    let mut needs_newline = false;
    while let Some(frame) = replies.next().await {
        match frame? {
            Frame::Begin => needs_newline = false,
            Frame::Chunk(text) if !text.is_empty() => {
                stdout.write_all(text.as_bytes()).await?;
                stdout.flush().await?;
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
    match run(config).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => report_error(&error),
    }
}

fn report_error(error: &anyhow::Error) -> ExitCode {
    eprintln!("error: {error:#}");
    ExitCode::FAILURE
}

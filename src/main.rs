//! Command-line entry point. Startup selects one framed prompt, a named workflow, or a
//! conversation. Both terminal interfaces use the library harness on a current-thread Tokio
//! runtime. Native setup and token generation can use blocking workers.

mod markdown;
mod repl;
mod tui;
mod workflow_output;

use anyhow::{Context, Result, ensure};
use arta::{
    config::{self, AgentProfile, StartupConfig},
    harness::Harness,
    mcp::McpTools,
    model::ModelId,
    provider::llama_cpp::{LlamaCPP, LlamaCPPConfig},
    skill::ActivateSkill,
    stream::{Frame, Message, Stream},
    workflow::Workflow,
};
use futures_util::{StreamExt, stream};
use std::{io::IsTerminal, process::ExitCode, sync::Arc};
use tokio::{
    io::{self, AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader},
    task::spawn_blocking,
};

/// Resources shared by CLI turns, including the discovered catalogs used to build mention
/// rules.
struct Cli {
    /// Initialized providers and CLI session bound after discovery.
    harness: Harness,
    /// Optional activation catalog also used to register skill mentions.
    skills: Option<Arc<ActivateSkill>>,
    /// Already connected tool handles grouped for per-server visibility in the REPL.
    mcp: McpTools,
}

/// Send one framed message to the startup agent with the full set of registered tools.
async fn run_once(startup: &'static StartupConfig, prompt: &str) -> Result<()> {
    let input = prompt_message(prompt).await?;
    let cli = cli_harness(startup, false).await?;
    let replies = cli
        .harness
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

/// Wrap a nonblank literal in one chunk, or read a multiline UTF-8 message for the sentinel
/// `-`.
async fn prompt_message(prompt: &str) -> Result<Message> {
    if prompt == "-" {
        stdin_message(BufReader::new(io::stdin())).await
    } else {
        ensure!(!prompt.trim().is_empty(), "prompt cannot be empty");
        Ok(Message::new(stream::iter([Ok(prompt.to_owned())])))
    }
}

/// Collect the request, register skills, and drive the workflow by polling its events. Node
/// profiles select their own MCP connections.
async fn run_workflow(
    startup: &'static StartupConfig,
    workflow: &Workflow,
    prompt: &str,
) -> Result<()> {
    let prompt = prompt_message(prompt).await?.into_text().await?;
    let mut harness = spawn_blocking(|| {
        let mut harness = Harness::new(&startup.settings)?;
        register_skills(&mut harness, &startup.settings)?;
        Ok::<_, anyhow::Error>(harness)
    })
    .await
    .context("failed to start the workflow harness")??;
    workflow_output::write(
        workflow.events(&mut harness, &prompt, false),
        &mut io::stdout(),
        &mut io::stderr(),
        std::io::stdout().is_terminal(),
    )
    .await
}

/// Resolve TOML search roots and register one activation tool if discovery finds any skills.
/// Retain the same catalog for REPL completions.
fn register_skills(
    harness: &mut Harness,
    config: &config::Config,
) -> Result<Option<Arc<ActivateSkill>>> {
    let directories = config
        .skills
        .directories
        .iter()
        .map(|path| config.resolve_path(path));
    let skills = ActivateSkill::discover(directories)?.map(Arc::new);
    if let Some(tool) = &skills {
        harness.register_tool(tool.clone())?;
    }
    Ok(skills)
}

/// Load the selected model on a blocking worker before connecting MCP servers. Create the CLI
/// session last so its agent captures the populated tool registry.
async fn cli_harness(startup: &'static StartupConfig, fullscreen: bool) -> Result<Cli> {
    let config = &startup.settings;
    let model_id = ModelId {
        provider: &config.app.provider,
        model: &config.app.model,
    };
    let (mut harness, skills) = spawn_blocking(move || {
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
        // Load the startup model before connecting tools.
        harness.model(model_id).with_context(|| {
            format!(
                "failed to load model '{}' for agent '{}'",
                model_id.model,
                startup.agent_name()
            )
        })?;
        let skills = register_skills(&mut harness, config)?;
        Ok::<_, anyhow::Error>((harness, skills))
    })
    .await
    .context("failed to start the CLI harness")??;
    let options = startup
        .agent
        .as_ref()
        .map(AgentProfile::options)
        .unwrap_or_default();
    let mcp = harness
        .connect_mcp_tools_in(config, options.workspace)
        .await?;
    harness.create_session_with_agents("cli", [(startup.agent_name(), model_id, options)])?;
    Ok(Cli {
        harness,
        skills,
        mcp,
    })
}

/// Register command and mention rules, then share one conversation controller with the chosen
/// interface. Fullscreen requires both stdin and stdout to be terminals.
async fn run_conversation(startup: &'static StartupConfig) -> Result<()> {
    let fullscreen = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
    let mut cli = cli_harness(startup, fullscreen)
        .await
        .context("failed to start the REPL")?;
    let mut parser = repl::parser::Parser::default();
    repl::command::register(&mut parser);
    let skills = repl::skill::register(&mut parser, cli.skills);
    let mcp_servers = startup
        .agent
        .as_ref()
        .map_or(&[][..], AgentProfile::mcp_servers);
    let mcp_tools = repl::mcp::register(&mut parser, cli.mcp, mcp_servers);
    let mut repl = repl::Repl::new(
        &mut cli.harness,
        startup.agent_name(),
        &parser,
        skills,
        mcp_tools,
    );
    if fullscreen {
        tui::run(&mut repl, &startup.settings).await
    } else {
        repl.run_lines(&startup.settings).await
    }
}

/// Capture original chunks for history while writing to stdout. Buffer terminal Markdown
/// until the message ends or fails. Flush redirected chunks immediately and terminate a
/// successful nonempty reply with a newline when needed.
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

/// Read through the first nonblank line before model setup. Preserve leading whitespace and
/// defer reading the rest of stdin until the message is polled.
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

/// Select the execution mode from the presence of a workflow and prompt. Report chained setup
/// and runtime errors on stderr.
#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    let config = match config::get() {
        Ok(config) => config,
        Err(error) => return report_error(error),
    };
    let result = match (config.workflow.as_ref(), config.prompt.as_deref()) {
        (Some(workflow), Some(prompt)) => run_workflow(config, workflow, prompt).await,
        (Some(_), None) => Err(anyhow::anyhow!("a workflow requires --prompt")),
        (None, Some(prompt)) => run_once(config, prompt).await,
        (None, None) => run_conversation(config).await,
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => report_error(&error),
    }
}

/// Print the error and its causes and return a failing process exit code.
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

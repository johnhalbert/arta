//! Line-oriented projection of workflow events. Replies go to the output writer and progress
//! goes to a separate writer. Label helpers also supply text for the fullscreen transcript.

use crate::markdown::clean_terminal_text;
use anyhow::Result;
use arta::{
    tool::ToolArguments,
    workflow::{LoopIteration, WorkflowEvent, WorkflowEvents},
};
use futures_util::StreamExt;
use serde_json::Value;
use std::fmt::Write;
use tokio::io::{AsyncWrite, AsyncWriteExt};

/// Stream every node's reply, separating completed nodes with a newline when needed. On
/// error, flush partial output before propagating the failure.
pub(super) async fn write(
    mut events: WorkflowEvents<'_>,
    output: &mut (impl AsyncWrite + Unpin),
    progress: &mut (impl AsyncWrite + Unpin),
    terminal: bool,
) -> Result<()> {
    let mut needs_newline = false;
    while let Some(event) = events.next().await {
        match event {
            Ok(WorkflowEvent::NodeStarted { id, agent, loops }) => {
                let label = node_label(&id, &agent, &loops);
                write_progress(progress, &format!("[{label}]")).await?;
            }
            Ok(WorkflowEvent::ImageStarted {
                id,
                provider,
                model,
                loops,
            }) => {
                let label = artifact_node_label(&id, "image", &provider, &model, &loops);
                write_progress(progress, &format!("[{label}]")).await?;
            }
            Ok(WorkflowEvent::AudioStarted {
                id,
                provider,
                model,
                loops,
            }) => {
                let label = artifact_node_label(&id, "audio", &provider, &model, &loops);
                write_progress(progress, &format!("[{label}]")).await?;
            }
            Ok(WorkflowEvent::ModelLoading { previous, next }) => {
                let message = WorkflowEvent::model_loading_message(previous.as_deref(), &next);
                write_progress(progress, &message).await?;
            }
            Ok(WorkflowEvent::LoopToolCall {
                iteration,
                name,
                arguments,
                error,
            }) => {
                let label = loop_tool_label(&iteration, &name, arguments, error.as_deref());
                write_progress(progress, &label).await?;
            }
            Ok(WorkflowEvent::Output(text)) if !text.is_empty() => {
                if terminal {
                    for line in text.split_inclusive('\n') {
                        output
                            .write_all(clean_terminal_text(line).as_bytes())
                            .await?;
                        if line.ends_with('\n') {
                            output.write_all(b"\n").await?;
                        }
                    }
                } else {
                    output.write_all(text.as_bytes()).await?;
                }
                output.flush().await?;
                needs_newline = !text.ends_with('\n');
            }
            Ok(WorkflowEvent::NodeFinished) if needs_newline => {
                output.write_all(b"\n").await?;
                output.flush().await?;
                needs_newline = false;
            }
            Err(error) => {
                if needs_newline {
                    output.write_all(b"\n").await?;
                }
                output.flush().await?;
                return Err(error);
            }
            _ => {}
        }
    }
    Ok(())
}

/// Strip control characters, terminate the status line, and flush it immediately.
async fn write_progress(progress: &mut (impl AsyncWrite + Unpin), message: &str) -> Result<()> {
    progress
        .write_all(format!("{}\n", clean_terminal_text(message)).as_bytes())
        .await?;
    progress.flush().await?;
    Ok(())
}

/// Format the control name, original JSON arguments, loop iteration, and optional rejection.
pub(super) fn loop_tool_label(
    iteration: &LoopIteration,
    name: &str,
    arguments: ToolArguments,
    error: Option<&str>,
) -> String {
    let mut label = format!(
        "Tool call {name}({}) · {} iteration {}",
        Value::Object(arguments),
        iteration.id,
        iteration.number
    );
    if let Some(error) = error {
        let _ = write!(label, " · rejected ({error})");
    }
    label
}

/// Disambiguate a node from its profile when their names differ and append all enclosing
/// iteration numbers.
pub(super) fn node_label(id: &str, agent: &str, loops: &[LoopIteration]) -> String {
    let mut label = if id == agent {
        id.to_owned()
    } else {
        format!("{id} ({agent})")
    };
    for iteration in loops {
        let _ = write!(label, " · {} iteration {}", iteration.id, iteration.number);
    }
    label
}

pub(super) fn artifact_node_label(
    id: &str,
    kind: &str,
    provider: &str,
    model: &str,
    loops: &[LoopIteration],
) -> String {
    node_label(id, &format!("{kind} · {provider}/{model}"), loops)
}

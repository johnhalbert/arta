//! Progress channel shared by the executor, blocking prompt observers, and event consumers.
//! Direct runs use a disabled emitter.

use super::LoopIteration;
use crate::{provider::PromptObserver, tool::ToolArguments};
use anyhow::Result;
use futures_util::stream::BoxStream;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Borrowing stream of progress records or a terminal execution error. Polling also drives
/// the run.
pub type WorkflowEvents<'a> = BoxStream<'a, Result<WorkflowEvent>>;

/// Ordered milestones for one workflow. Node-scoped records refer to the most recent node
/// start.
#[derive(Debug, PartialEq, Eq)]
pub enum WorkflowEvent {
    /// An agent visit began, before tool resolution or model selection.
    NodeStarted {
        /// Unique node ID in the execution tree.
        id: String,
        /// Name of the Markdown profile used for this visit.
        agent: String,
        /// Snapshot of the active loop stack from outermost to innermost.
        loops: Vec<LoopIteration>,
    },
    /// An image node began generation with the selected provider and model.
    ImageStarted {
        id: String,
        provider: String,
        model: String,
        loops: Vec<LoopIteration>,
    },
    /// An audio node began generation with the selected provider and model.
    AudioStarted {
        id: String,
        provider: String,
        model: String,
        loops: Vec<LoopIteration>,
    },
    /// A new binding is needed, including when settings change for the same model ID.
    ModelLoading {
        /// Previous provider/model label, absent on first load.
        previous: Option<String>,
        /// Requested provider/model label.
        next: String,
    },
    /// The new binding is ready for this visit.
    ModelLoaded {
        /// Provider/model label that finished loading.
        model: String,
    },
    /// Rendered prompt for one inference step when observation is enabled.
    ModelInput(String),
    /// A loop control tool was invoked. An error means the call was rejected.
    LoopToolCall {
        /// Innermost loop whose finishing tool was invoked.
        iteration: LoopIteration,
        /// Invoked control name, either `continue` or `break`.
        name: String,
        /// Original arguments, retained for the CLI and TUI transcript.
        arguments: ToolArguments,
        /// Reason this control call was rejected, when present.
        error: Option<String>,
    },
    /// Chunk of the node's reply or the path to a generated artifact.
    Output(String),
    /// The node visit completed successfully, including any required decision.
    NodeFinished,
}

impl WorkflowEvent {
    /// Distinguish first load, switch, and reconfiguration of the same displayed model.
    pub fn model_loading_message(previous: Option<&str>, next: &str) -> String {
        match previous {
            Some(previous) if previous == next => format!("Reloading model {next}"),
            Some(previous) => format!("Switching model from {previous} to {next}"),
            None => format!("Loading model {next}"),
        }
    }
}

/// Optional emitter. Full rendered prompts are captured only when a consumer requests them.
#[derive(Clone, Default)]
pub(super) struct Progress {
    /// Absent for direct runs. Attached streams receive both progress and terminal errors.
    sender: Option<mpsc::UnboundedSender<Result<WorkflowEvent>>>,
    /// Allows creation of an observer that copies rendered prompts into the event stream.
    include_model_input: bool,
}

impl Progress {
    /// Attach progress to the stream's channel and select whether prompt observation is
    /// enabled.
    pub(super) fn new(
        sender: mpsc::UnboundedSender<Result<WorkflowEvent>>,
        include_model_input: bool,
    ) -> Self {
        Self {
            sender: Some(sender),
            include_model_input,
        }
    }

    /// Construct the event only for an attached consumer, then yield so the consumer can draw
    /// or flush.
    pub(super) async fn emit(&self, event: impl FnOnce() -> WorkflowEvent) {
        if let Some(sender) = &self.sender {
            let _ = sender.send(Ok(event()));
            // Let the consumer draw or flush before the workflow advances.
            tokio::task::yield_now().await;
        }
    }

    /// Create a thread-safe callback that enqueues rendered input on the same progress
    /// channel.
    pub(super) fn prompt_observer(&self) -> Option<PromptObserver> {
        let sender = self
            .sender
            .as_ref()
            .filter(|_| self.include_model_input)?
            .clone();
        Some(Arc::new(move |prompt| {
            let _ = sender.send(Ok(WorkflowEvent::ModelInput(prompt.to_owned())));
        }))
    }
}

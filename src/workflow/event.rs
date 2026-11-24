use crate::{provider::PromptObserver, tool::ToolArguments};
use anyhow::Result;
use futures_util::stream::BoxStream;
use std::sync::Arc;
use tokio::sync::mpsc;

pub type WorkflowEvents<'a> = BoxStream<'a, Result<WorkflowEvent>>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoopIteration {
    pub id: String,
    pub number: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub enum WorkflowEvent {
    NodeStarted {
        id: String,
        agent: String,
        loops: Vec<LoopIteration>,
    },
    ModelLoading {
        previous: Option<String>,
        next: String,
    },
    ModelLoaded {
        model: String,
    },
    ModelInput(String),
    /// A loop control tool was invoked. An error means the call was rejected.
    LoopToolCall {
        iteration: LoopIteration,
        name: String,
        arguments: ToolArguments,
        error: Option<String>,
    },
    Output(String),
    NodeFinished,
}

impl WorkflowEvent {
    pub fn model_loading_message(previous: Option<&str>, next: &str) -> String {
        match previous {
            Some(previous) if previous == next => format!("Reloading model {next}"),
            Some(previous) => format!("Switching model from {previous} to {next}"),
            None => format!("Loading model {next}"),
        }
    }
}

#[derive(Clone, Default)]
pub(super) struct Progress {
    sender: Option<mpsc::UnboundedSender<Result<WorkflowEvent>>>,
    include_model_input: bool,
}

impl Progress {
    pub(super) fn new(
        sender: mpsc::UnboundedSender<Result<WorkflowEvent>>,
        include_model_input: bool,
    ) -> Self {
        Self {
            sender: Some(sender),
            include_model_input,
        }
    }

    pub(super) async fn emit(&self, event: impl FnOnce() -> WorkflowEvent) {
        if let Some(sender) = &self.sender {
            let _ = sender.send(Ok(event()));
            // Let the consumer draw or flush before the workflow advances.
            tokio::task::yield_now().await;
        }
    }

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

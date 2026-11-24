use anyhow::{Result, bail, ensure};
use futures_util::future::BoxFuture;
use serde_json::{Map, Value};
use std::{
    collections::{BTreeMap, btree_map::Entry},
    sync::Arc,
};

pub type ToolArguments = Map<String, Value>;

/// A callable tool's name, description, and JSON Schema for its arguments.
#[derive(Clone, Debug)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: Map<String, Value>,
}

/// Data returned to the model. A tool can report a failure the model can act on.
#[derive(Debug)]
pub struct ToolOutput {
    pub content: Value,
    pub is_error: bool,
}

impl ToolOutput {
    pub fn success(content: impl Into<Value>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }

    pub fn error(content: impl Into<Value>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
        }
    }
}

/// An executable tool independent of the provider and the transport.
pub trait Tool: Send + Sync {
    fn spec(&self) -> &ToolSpec;

    /// Finish the turn with this tool's output when the call succeeds.
    fn finishes_turn(&self) -> bool {
        false
    }

    /// Return execution errors in the output when the model can use them to recover.
    fn call(&self, arguments: ToolArguments) -> BoxFuture<'_, Result<ToolOutput>>;
}

#[derive(Clone, Default)]
pub(crate) struct Tools {
    entries: BTreeMap<String, Arc<dyn Tool>>,
}

impl Tools {
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(crate) fn register(&mut self, tool: Arc<dyn Tool>) -> Result<()> {
        let name = &tool.spec().name;
        ensure!(
            !name.is_empty()
                && name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-')),
            "invalid tool name '{name}': use letters, digits, underscores, or hyphens"
        );
        match self.entries.entry(name.clone()) {
            Entry::Vacant(entry) => {
                entry.insert(tool);
                Ok(())
            }
            Entry::Occupied(_) => bail!("tool '{name}' is already registered"),
        }
    }

    pub(crate) fn specs(&self) -> Vec<&ToolSpec> {
        self.entries.values().map(|tool| tool.spec()).collect()
    }

    pub(crate) fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.entries.get(name).map(Arc::as_ref)
    }

    pub(crate) fn remove(&mut self, name: &str) {
        self.entries.remove(name);
    }
}

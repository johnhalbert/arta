//! In-process contract shared by built-in tools and MCP clients. The registry enforces unique
//! names and presents schemas in sorted order.

use anyhow::{Result, bail, ensure};
use futures_util::future::BoxFuture;
use serde_json::{Map, Value};
use std::{
    collections::{BTreeMap, btree_map::Entry},
    sync::Arc,
};

/// JSON object parsed from the model's function arguments.
pub type ToolArguments = Map<String, Value>;

/// A callable tool's name, description, and JSON Schema for its arguments.
#[derive(Clone, Debug)]
pub struct ToolSpec {
    /// Exact key used for dispatch and result correlation.
    pub name: String,
    /// Text advertised alongside the function schema.
    pub description: String,
    /// JSON Schema object describing accepted arguments. Each implementation validates its
    /// own arguments.
    pub input_schema: Map<String, Value>,
}

/// Data returned to the model. A tool can report a failure the model can act on.
#[derive(Debug)]
pub struct ToolOutput {
    /// Text or structured JSON returned to the adapter.
    pub content: Value,
    /// Marks a result as a recoverable tool failure during inference.
    pub is_error: bool,
}

impl ToolOutput {
    /// Construct a successful result from text or structured JSON.
    pub fn success(content: impl Into<Value>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }

    /// Construct a failure that the model can inspect on its next step.
    pub fn error(content: impl Into<Value>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
        }
    }
}

/// An executable tool independent of the provider and the transport.
pub trait Tool: Send + Sync {
    /// Borrow the specification used for registration and prompt rendering.
    fn spec(&self) -> &ToolSpec;

    /// Allow an automatically advertised tool to be omitted for text-only models.
    fn omit_if_unsupported(&self) -> bool {
        false
    }

    /// Finish the turn with this tool's output when the call succeeds.
    fn finishes_turn(&self) -> bool {
        false
    }

    /// Return execution errors in the output when the model can use them to recover.
    fn call(&self, arguments: ToolArguments) -> BoxFuture<'_, Result<ToolOutput>>;
}

/// Name-indexed shared implementations. Cloning copies the index and shares the tool
/// instances.
#[derive(Clone, Default)]
pub(crate) struct Tools {
    /// Ordered index of executable handles keyed by their advertised names.
    entries: BTreeMap<String, Arc<dyn Tool>>,
}

impl Tools {
    pub(crate) fn has_optional(&self) -> bool {
        self.entries.values().any(|tool| tool.omit_if_unsupported())
    }

    pub(crate) fn selected(&self, keep: impl Fn(&dyn Tool) -> bool) -> Self {
        Self {
            entries: self
                .entries
                .iter()
                .filter(|(_, tool)| keep(tool.as_ref()))
                .map(|(name, tool)| (name.clone(), Arc::clone(tool)))
                .collect(),
        }
    }

    /// Check whether the caller can take the plain-inference path without requesting an
    /// adapter.
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Accept a unique nonempty ASCII name made of letters, digits, underscores, or hyphens.
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

    /// Return borrowed schemas in stable name order.
    pub(crate) fn specs(&self) -> Vec<&ToolSpec> {
        self.entries.values().map(|tool| tool.spec()).collect()
    }

    /// Borrow a callable implementation by the exact name parsed from a model response.
    pub(crate) fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.entries.get(name).map(Arc::as_ref)
    }

    /// Drop a name from this registry copy, used to filter MCP handles for workflow nodes.
    pub(crate) fn remove(&mut self, name: &str) {
        self.entries.remove(name);
    }
}

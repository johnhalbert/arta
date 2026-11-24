//! Model identity shared by all providers. A model handle exposes metadata while inference
//! and resource management go through its owning provider.

/// Identifies a model within a provider registered with the harness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelId<'a> {
    /// Harness registry key identifying the provider.
    pub provider: &'a str,
    /// Model key within the selected provider.
    pub model: &'a str,
}

/// The identity and metadata of a model that a provider can use for inference.
pub trait Model: Send + Sync {
    /// The identifier that the owning provider accepts for inference.
    fn id(&self) -> &str;

    /// The maximum context length, in tokens, from model training.
    fn trained_context_length(&self) -> u32;
}

/// Identifies a model within a provider registered with the harness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelId<'a> {
    pub provider: &'a str,
    pub model: &'a str,
}

/// A loaded model that a provider can use for inference.
pub trait Model: Send + Sync {
    /// The identifier that the owning provider accepts for inference.
    fn id(&self) -> &str;

    /// The maximum context length, in tokens, from model training.
    fn trained_context_length(&self) -> u32;
}

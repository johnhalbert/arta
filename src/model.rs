/// A loaded model that a provider can use for inference.
pub trait Model {
    /// The maximum context length, in tokens, from model training.
    fn trained_context_length(&self) -> u32;
}

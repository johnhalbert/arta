pub mod llama_cpp;

/// Runs a single prompt without retaining conversation state.
pub trait Provider {
    fn send(&self, message: &str) -> anyhow::Result<String>;
}

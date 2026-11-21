use crate::provider::{
    Provider,
    llama_cpp::{LlamaCPP, LlamaCPPConfig, LlamaCPPModel},
};
use anyhow::Result;
use std::path::Path;

/// Owns the provider that loads models and runs prompts for agents.
pub struct Harness<P = LlamaCPP> {
    provider: P,
}

impl Harness<LlamaCPP> {
    /// Create a harness with a llama.cpp provider.
    pub fn new(config: LlamaCPPConfig) -> Result<Self> {
        Ok(Self::with_provider(LlamaCPP::new(config)?))
    }

    pub fn load_model(&self, path: impl AsRef<Path>, gpu_layers: u32) -> Result<LlamaCPPModel<'_>> {
        self.provider.load_model(path, gpu_layers)
    }
}

impl<P: Provider> Harness<P> {
    /// Give the harness ownership of a provider supplied by the caller.
    pub fn with_provider(provider: P) -> Self {
        Self { provider }
    }

    pub fn send<'harness>(
        &'harness self,
        model: &P::Model<'harness>,
        prompt: &str,
    ) -> Result<String> {
        self.provider.send(model, prompt)
    }
}

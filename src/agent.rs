use crate::harness::Harness;
use crate::provider::Provider;
use anyhow::Result;

pub struct Agent<'harness, P> {
    harness: &'harness Harness<P>,
}

impl<'harness, P: Provider> Agent<'harness, P> {
    pub fn new(harness: &'harness Harness<P>) -> Self {
        Self { harness }
    }

    /// Run one independent prompt and return the provider's response.
    pub fn prompt(&self, model: &P::Model<'harness>, prompt: &str) -> Result<String> {
        self.harness.send(model, prompt)
    }
}

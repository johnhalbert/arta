use crate::provider::Provider;
use anyhow::Result;

pub struct Agent<'provider, P> {
    provider: &'provider P,
}

impl<'provider, P: Provider> Agent<'provider, P> {
    pub fn new(provider: &'provider P) -> Self {
        Self { provider }
    }

    /// Run one independent prompt and return the provider's response.
    pub fn prompt(&self, model: &P::Model<'provider>, prompt: &str) -> Result<String> {
        self.provider.send(model, prompt)
    }
}

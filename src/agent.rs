use crate::provider::Provider;
use anyhow::Result;

pub struct Agent<P> {
    provider: P,
}

impl<P: Provider> Agent<P> {
    pub fn new(provider: P) -> Self {
        Self { provider }
    }

    /// Run one independent prompt and return the provider's response.
    pub fn prompt(&self, message: &str) -> Result<String> {
        self.provider.send(message)
    }
}

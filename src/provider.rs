pub mod llama_cpp;

use crate::model::Model;
use anyhow::Result;

/// Runs a single prompt without retaining conversation state.
pub trait Provider {
    type Model<'provider>: Model
    where
        Self: 'provider;

    fn send<'provider>(
        &'provider self,
        model: &Self::Model<'provider>,
        prompt: &str,
    ) -> Result<String>;
}

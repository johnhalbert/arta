//! Local models served by the native Cactus Compute engine in a separate process.

mod config;
mod needle;
mod runtime;

pub use config::{CactusConfig, CactusModelConfig, RuntimeConfig};
pub use needle::NeedleConfig;

use self::needle::Needle;
use super::{
    PromptObserver, Provider, Turn,
    registry::ProviderRegistration,
    tool::{ToolAdapter, ToolExchange, ToolRequest, ToolResponse, ToolTurn},
};
use crate::{config::Config, model::Model, stream::Message};
use anyhow::{Context, Result, ensure};
use futures_util::{future::BoxFuture, stream};
use std::{collections::HashMap, sync::Arc};

/// Run the internal worker mode, or record the executable for later workers.
#[doc(hidden)]
pub fn worker_mode(mut args: impl Iterator<Item = std::ffi::OsString>) -> Option<Result<()>> {
    if args.next().as_deref() != Some(std::ffi::OsStr::new(runtime::WORKER_FLAG)) {
        runtime::register_cli();
        return None;
    }
    Some(
        match (
            args.next().as_deref().and_then(std::ffi::OsStr::to_str),
            args.next(),
        ) {
            (Some("needle"), None) => needle::worker::run(),
            _ => Err(anyhow::anyhow!("expected a supported Cactus worker model")),
        },
    )
}

/// The `cactus` provider for locally executed Cactus Compute models.
pub struct CactusCompute {
    runtime: Arc<RuntimeConfig>,
    models: HashMap<String, CactusModel>,
}

enum CactusModel {
    Needle(Arc<Needle>),
}

inventory::submit! {
    ProviderRegistration { id: CactusCompute::ID, create: CactusCompute::from_config }
}

impl CactusCompute {
    /// Registry and TOML identifier.
    pub const ID: &str = "cactus";

    fn from_config(config: &Config) -> Result<Arc<dyn Provider>> {
        let mut settings: CactusConfig = config.provider_settings(Self::ID)?;
        settings.resolve_paths(config)?;
        Ok(Arc::new(Self::new(settings)?))
    }

    /// Register model settings. The SDK starts on the first inference request.
    pub fn new(config: CactusConfig) -> Result<Self> {
        let mut models = HashMap::with_capacity(config.models.len());
        for model in config.models {
            let (id, model) = match model {
                CactusModelConfig::Needle(settings) => {
                    let id = settings.id.clone();
                    let needle = Needle::new(settings)
                        .with_context(|| format!("invalid Cactus model '{id}'"))?;
                    (id, CactusModel::Needle(Arc::new(needle)))
                }
            };
            ensure!(
                models.insert(id.clone(), model).is_none(),
                "Cactus model '{id}' is already registered"
            );
        }
        Ok(Self {
            runtime: Arc::new(config.runtime),
            models,
        })
    }

    fn registered(&self, id: &str) -> Result<&CactusModel> {
        self.models
            .get(id)
            .with_context(|| format!("unknown Cactus model '{id}'"))
    }
}

impl Provider for CactusCompute {
    fn model(&self, model_id: &str) -> Result<Arc<dyn Model>> {
        match self.registered(model_id)? {
            CactusModel::Needle(needle) => needle.model(),
        }
    }

    fn switch_model(&self, _current_id: &str, model_id: &str) -> Result<Arc<dyn Model>> {
        self.model(model_id)
    }

    fn configured(&self, config: &Config) -> Result<Arc<dyn Provider>> {
        Self::from_config(config)
    }

    fn send<'a>(&'a self, model_id: &'a str, prompt: &'a str) -> BoxFuture<'a, Result<Message>> {
        self.chat_observed(model_id, None, &[], prompt, None)
    }

    fn chat<'a>(
        &'a self,
        model_id: &'a str,
        history: &'a [Turn],
        prompt: &'a str,
    ) -> BoxFuture<'a, Result<Message>> {
        self.chat_observed(model_id, None, history, prompt, None)
    }

    fn chat_with_system<'a>(
        &'a self,
        model_id: &'a str,
        system_prompt: Option<&'a str>,
        history: &'a [Turn],
        prompt: &'a str,
    ) -> BoxFuture<'a, Result<Message>> {
        self.chat_observed(model_id, system_prompt, history, prompt, None)
    }

    fn chat_observed<'a>(
        &'a self,
        model_id: &'a str,
        system_prompt: Option<&'a str>,
        _history: &'a [Turn],
        prompt: &'a str,
        observer: Option<PromptObserver>,
    ) -> BoxFuture<'a, Result<Message>> {
        Box::pin(async move {
            let reply = match self.registered(model_id)? {
                CactusModel::Needle(needle) => {
                    needle
                        .extract(&self.runtime, system_prompt, prompt, observer.as_ref())
                        .await?
                }
            };
            Ok(Message::new(stream::iter([Ok(reply)])))
        })
    }

    fn tool_adapter(&self) -> Option<&dyn ToolAdapter> {
        Some(self)
    }
}

impl ToolAdapter for CactusCompute {
    fn begin_turn(&self, model_id: &str) -> Result<Option<Box<dyn ToolTurn>>> {
        match self.registered(model_id)? {
            CactusModel::Needle(needle) => Ok(Some(Box::new(needle.turn(self.runtime.clone())))),
        }
    }

    fn infer<'a>(&'a self, request: ToolRequest<'a>) -> BoxFuture<'a, Result<ToolResponse>> {
        self.infer_recorded(request, &[], None)
    }

    fn infer_observed<'a>(
        &'a self,
        request: ToolRequest<'a>,
        observer: Option<PromptObserver>,
    ) -> BoxFuture<'a, Result<ToolResponse>> {
        self.infer_recorded(request, &[], observer)
    }

    fn infer_recorded<'a>(
        &'a self,
        request: ToolRequest<'a>,
        past_exchanges: &'a [Vec<ToolExchange>],
        observer: Option<PromptObserver>,
    ) -> BoxFuture<'a, Result<ToolResponse>> {
        Box::pin(async move {
            match self.registered(request.model_id)? {
                CactusModel::Needle(needle) => {
                    needle
                        .turn(self.runtime.clone())
                        .infer(request, past_exchanges, observer)
                        .await
                }
            }
        })
    }
}

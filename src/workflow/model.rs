use super::{WorkflowEvent, event::Progress};
use crate::{config::Config, harness::Harness, model::Model, provider::Provider};
use anyhow::{Context, Result};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(PartialEq)]
struct ProviderSettings {
    id: String,
    table: Option<toml::Table>,
    directory: PathBuf,
}

impl ProviderSettings {
    fn new(settings: &Config, id: &str) -> Self {
        Self {
            id: id.to_owned(),
            table: settings.provider_table(id).cloned(),
            directory: settings.resolve_path(Path::new(".")),
        }
    }
}

struct Binding {
    id: String,
    model: Arc<dyn Model>,
}

/// The model selected by the workflow and responsible for releasing its resources.
pub(super) struct ActiveModel {
    settings: ProviderSettings,
    pub(super) provider: Arc<dyn Provider>,
    binding: Option<Binding>,
}

impl ActiveModel {
    pub(super) async fn select<'a>(
        active: &'a mut Option<Self>,
        harness: &Harness,
        settings: &Config,
        progress: &Progress,
    ) -> Result<&'a Self> {
        let provider_id = &settings.app.provider;
        let provider_settings = ProviderSettings::new(settings, provider_id);
        let id = &settings.app.model;
        if !active.as_ref().is_some_and(|current| {
            current.settings == provider_settings
                && current
                    .binding
                    .as_ref()
                    .is_some_and(|model| model.id == *id)
        }) {
            progress
                .emit(|| WorkflowEvent::ModelLoading {
                    previous: active.as_ref().and_then(|current| {
                        current
                            .binding
                            .as_ref()
                            .map(|binding| format!("{}/{}", current.settings.id, binding.id))
                    }),
                    next: format!("{provider_id}/{id}"),
                })
                .await;
            let base = Arc::clone(harness.provider_handle(provider_id)?);
            let use_base =
                ProviderSettings::new(harness.workflow_config(), provider_id) == provider_settings;
            let settings = settings.clone();
            let target = id.clone();
            let previous = active.take();
            *active = Some(
                tokio::task::spawn_blocking(move || {
                    if let Some(mut current) = previous {
                        if current.settings == provider_settings {
                            let binding =
                                current.binding.take().context("no active workflow model")?;
                            drop(binding.model);
                            let model = current
                                .provider
                                .switch_model(&binding.id, &target)
                                .inspect_err(|_| {
                                    let _ = current.provider.unload_model(&binding.id);
                                })?;
                            current.binding = Some(Binding { id: target, model });
                            return Ok(current);
                        }
                        current.unload()?;
                    }
                    let provider = if use_base {
                        base
                    } else {
                        base.configured(&settings)?
                    };
                    let model = provider.model(&target)?;
                    Ok::<_, anyhow::Error>(Self {
                        settings: provider_settings,
                        provider,
                        binding: Some(Binding { id: target, model }),
                    })
                })
                .await
                .context("workflow model load worker failed")?
                .with_context(|| {
                    format!("failed to select model '{id}' for provider '{provider_id}'")
                })?,
            );
            progress
                .emit(|| WorkflowEvent::ModelLoaded {
                    model: format!("{provider_id}/{id}"),
                })
                .await;
        }
        Ok(active.as_ref().expect("model selection succeeded"))
    }

    pub(super) fn model(&self) -> Arc<dyn Model> {
        Arc::clone(
            &self
                .binding
                .as_ref()
                .expect("workflow model is active")
                .model,
        )
    }

    pub(super) fn unload(&mut self) -> Result<()> {
        if let Some(binding) = self.binding.take() {
            drop(binding.model);
            self.provider.unload_model(&binding.id)?;
        }
        Ok(())
    }
}

impl Drop for ActiveModel {
    fn drop(&mut self) {
        let _ = self.unload();
    }
}

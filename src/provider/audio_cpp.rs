//! Local audio generation through the audio.cpp shared library.

mod native;

use super::{
    Provider,
    audio::{AudioGenerator, AudioOptions},
    registry::ProviderRegistration,
};
use crate::{config::Config, model::Model, stream::Message};
use anyhow::{Context, Result, bail, ensure};
use futures_util::future::BoxFuture;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, btree_map::Entry},
    env::{
        self,
        consts::{DLL_PREFIX, DLL_SUFFIX},
    },
    num::NonZeroU16,
    path::{self, Path, PathBuf},
    sync::{Arc, Mutex},
    thread,
};
use tokio::sync::{mpsc, oneshot};

/// Runtime settings and model registrations for audio.cpp.
#[derive(Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct AudioCppConfig {
    /// Config-relative library path. Defaults to the library beside the executable.
    pub library_path: Option<PathBuf>,
    /// Config-relative artifact directory. Defaults to `audio`.
    pub output_dir: Option<PathBuf>,
    /// Defaults to the first configured model.
    pub default_model: Option<String>,
    /// Defaults to the backend selected by Cargo features, or CPU on a CPU build.
    pub backend: Option<AudioBackend>,
    pub threads: Option<NonZeroU16>,
    pub models: Vec<AudioModelConfig>,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioBackend {
    Cpu,
    Cuda,
    Hip,
    Vulkan,
    Metal,
}

impl Default for AudioBackend {
    fn default() -> Self {
        if cfg!(feature = "cuda") {
            Self::Cuda
        } else if cfg!(feature = "hip") {
            Self::Hip
        } else if cfg!(feature = "vulkan") {
            Self::Vulkan
        } else {
            Self::Cpu
        }
    }
}

/// The text-to-audio task supported by the registered model.
#[derive(Clone, Copy, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioTask {
    #[default]
    Tts,
    Gen,
}

/// One local audio.cpp model file or package directory and its request defaults.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AudioModelConfig {
    pub id: String,
    pub path: PathBuf,
    pub family: Option<String>,
    #[serde(default)]
    pub task: AudioTask,
    #[serde(default)]
    pub options: AudioOptions,
}

/// Registered `audio.cpp` provider with one in-process inference thread.
pub struct AudioCpp {
    library_path: PathBuf,
    output_dir: PathBuf,
    backend: AudioBackend,
    threads: i32,
    default_model: Option<String>,
    models: BTreeMap<String, Arc<AudioModelConfig>>,
    worker: Mutex<Option<mpsc::UnboundedSender<native::Request>>>,
}

inventory::submit! {
    ProviderRegistration { id: AudioCpp::ID, create: AudioCpp::from_config }
}

impl AudioCpp {
    pub const ID: &str = super::audio::DEFAULT_PROVIDER_ID;

    fn from_config(config: &Config) -> Result<Arc<dyn Provider>> {
        let settings: AudioCppConfig = config.provider_settings(Self::ID)?;
        let default_model = settings
            .default_model
            .or_else(|| settings.models.first().map(|model| model.id.clone()));
        let mut models = BTreeMap::new();
        for mut model in settings.models {
            ensure!(
                !model.id.trim().is_empty(),
                "audio model ID cannot be empty"
            );
            model.path = absolute(config, &model.path).context("invalid audio model path")?;
            if let Some(family) = &model.family {
                ensure!(
                    !family.trim().is_empty() && !family.contains('\0'),
                    "invalid family for audio model '{}'",
                    model.id
                );
            }
            validate_options(&model.options)
                .with_context(|| format!("invalid settings for audio model '{}'", model.id))?;
            match models.entry(model.id.clone()) {
                Entry::Vacant(entry) => {
                    entry.insert(Arc::new(model));
                }
                Entry::Occupied(entry) => {
                    bail!("audio model '{}' is already registered", entry.key());
                }
            }
        }
        if let Some(id) = &default_model {
            ensure!(
                models.contains_key(id),
                "unknown default audio model '{id}'"
            );
        }
        let library_path = match settings.library_path {
            Some(path) => absolute(config, &path).context("invalid audio.cpp library path")?,
            None => env::current_exe()
                .context("could not locate the bundled audio.cpp library")?
                .with_file_name(format!("{DLL_PREFIX}audiocpp{DLL_SUFFIX}")),
        };
        let output_dir = absolute(
            config,
            settings.output_dir.as_deref().unwrap_or(Path::new("audio")),
        )
        .context("invalid audio output directory")?;
        let threads = settings.threads.map_or_else(
            || {
                thread::available_parallelism()
                    .map_or(1, |n| i32::try_from(n.get()).unwrap_or(i32::MAX))
            },
            |n| i32::from(n.get()),
        );
        Ok(Arc::new(Self {
            library_path,
            output_dir,
            backend: settings.backend.unwrap_or_default(),
            threads,
            default_model,
            models,
            worker: Mutex::new(None),
        }))
    }

    fn requests(&self) -> Result<mpsc::UnboundedSender<native::Request>> {
        let mut worker = self
            .worker
            .lock()
            .map_err(|_| anyhow::anyhow!("audio worker lock was poisoned"))?;
        if let Some(requests) = worker.as_ref().filter(|requests| !requests.is_closed()) {
            return Ok(requests.clone());
        }
        let (requests, receiver) = mpsc::unbounded_channel();
        let library = self.library_path.clone();
        let output = self.output_dir.clone();
        let backend = self.backend;
        let threads = self.threads;
        thread::Builder::new()
            .name(Self::ID.into())
            .spawn(move || native::serve(&library, &output, backend, threads, receiver))
            .context("failed to start the audio.cpp inference thread")?;
        *worker = Some(requests.clone());
        Ok(requests)
    }

    async fn render(
        &self,
        model: &Arc<AudioModelConfig>,
        prompt: &str,
        options: &AudioOptions,
    ) -> Result<PathBuf> {
        ensure!(!prompt.trim().is_empty(), "audio prompt cannot be empty");
        ensure!(
            !prompt.contains('\0'),
            "audio prompt contains a NUL character"
        );
        validate_options(options)?;
        let mut merged = model.options.clone();
        merged.extend(options.clone());
        let (reply, result) = oneshot::channel();
        self.requests()?
            .send(native::Request {
                model: Arc::clone(model),
                prompt: prompt.to_owned(),
                options: merged,
                reply,
            })
            .map_err(|_| anyhow::anyhow!("audio.cpp inference thread stopped"))?;
        let artifact = result
            .await
            .context("audio.cpp inference thread stopped")??;
        Ok(artifact.keep().join(native::AUDIO_FILENAME))
    }
}

fn absolute(config: &Config, value: &Path) -> Result<PathBuf> {
    ensure!(!value.as_os_str().is_empty(), "path cannot be empty");
    Ok(path::absolute(config.resolve_path(value))?)
}

fn validate_options(options: &AudioOptions) -> Result<()> {
    for (key, value) in options {
        ensure!(
            !key.trim().is_empty() && !key.contains('\0'),
            "invalid audio option name"
        );
        ensure!(
            !value.contains('\0'),
            "audio option '{key}' contains a NUL character"
        );
    }
    Ok(())
}

impl Provider for AudioCpp {
    fn model(&self, _model_id: &str) -> Result<Arc<dyn Model>> {
        bail!(
            "{} provides audio models. Use an audio workflow node or the generate_audio tool",
            Self::ID
        )
    }

    fn send<'a>(&'a self, _model_id: &'a str, _prompt: &'a str) -> BoxFuture<'a, Result<Message>> {
        Box::pin(async { bail!("{} does not support text inference", Self::ID) })
    }

    fn audio_generator(self: Arc<Self>) -> Option<Arc<dyn AudioGenerator>> {
        Some(self)
    }
}

impl AudioGenerator for AudioCpp {
    fn default_model(&self) -> Option<&str> {
        self.default_model.as_deref()
    }

    fn model_ids(&self) -> Vec<&str> {
        self.models.keys().map(String::as_str).collect()
    }

    fn generate<'a>(
        &'a self,
        model: &'a str,
        prompt: &'a str,
        options: &'a AudioOptions,
    ) -> BoxFuture<'a, Result<PathBuf>> {
        Box::pin(async move {
            let settings = self
                .models
                .get(model)
                .with_context(|| format!("unknown audio model '{model}'"))?;
            self.render(settings, prompt, options)
                .await
                .with_context(|| format!("audio generation with model '{model}' failed"))
        })
    }
}

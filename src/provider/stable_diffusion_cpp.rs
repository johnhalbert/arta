//! Local image generation through the stable-diffusion.cpp shared library.

mod native;

use super::{
    Provider,
    image::{ImageGenerator, ImageOptions},
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
    path::{self, Path, PathBuf},
    sync::{Arc, Mutex},
    thread,
};
use tokio::sync::{mpsc, oneshot};

/// Runtime settings and per-checkpoint registrations for stable-diffusion.cpp.
#[derive(Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct StableDiffusionCppConfig {
    /// Config-relative path to the native library. Defaults to the library beside the executable.
    pub library_path: Option<PathBuf>,
    /// Config-relative directory for generated images. Defaults to `images`.
    pub output_dir: Option<PathBuf>,
    /// Model used when no ID is supplied. Defaults to the first configured model.
    pub default_model: Option<String>,
    pub models: Vec<StableDiffusionModelConfig>,
}

/// One local checkpoint and its generation defaults.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StableDiffusionModelConfig {
    pub id: String,
    pub path: PathBuf,
    pub vae: Option<PathBuf>,
    #[serde(default)]
    pub options: ImageOptions,
}

/// Registered `stable-diffusion.cpp` image provider with one in-process inference thread.
pub struct StableDiffusionCpp {
    library_path: PathBuf,
    output_dir: PathBuf,
    default_model: Option<String>,
    models: BTreeMap<String, Arc<StableDiffusionModelConfig>>,
    /// Starts lazily and keeps native handles on the inference thread.
    worker: Mutex<Option<mpsc::UnboundedSender<native::Request>>>,
}

inventory::submit! {
    ProviderRegistration { id: StableDiffusionCpp::ID, create: StableDiffusionCpp::from_config }
}

impl StableDiffusionCpp {
    pub const ID: &str = "stable-diffusion.cpp";

    fn from_config(config: &Config) -> Result<Arc<dyn Provider>> {
        let settings: StableDiffusionCppConfig = config.provider_settings(Self::ID)?;
        let default_model = settings
            .default_model
            .or_else(|| settings.models.first().map(|model| model.id.clone()));
        let mut models = BTreeMap::new();
        for mut model in settings.models {
            ensure!(
                !model.id.trim().is_empty(),
                "image model ID cannot be empty"
            );
            model.path = absolute(config, &model.path).context("invalid image model path")?;
            if let Some(vae) = &mut model.vae {
                *vae = absolute(config, vae).context("invalid image model VAE path")?;
            }
            validate_options(&model.options)
                .with_context(|| format!("invalid settings for image model '{}'", model.id))?;
            match models.entry(model.id.clone()) {
                Entry::Vacant(entry) => {
                    entry.insert(Arc::new(model));
                }
                Entry::Occupied(entry) => {
                    bail!("image model '{}' is already registered", entry.key())
                }
            }
        }
        if let Some(id) = &default_model {
            ensure!(
                models.contains_key(id),
                "unknown default image model '{id}'"
            );
        }
        let library_path = match settings.library_path {
            Some(path) => {
                absolute(config, &path).context("invalid stable-diffusion.cpp library path")?
            }
            None => env::current_exe()
                .context("could not locate the bundled stable-diffusion.cpp library")?
                .with_file_name(format!("{DLL_PREFIX}stable-diffusion{DLL_SUFFIX}")),
        };
        let output_dir = absolute(
            config,
            settings
                .output_dir
                .as_deref()
                .unwrap_or(Path::new("images")),
        )
        .context("invalid image output directory")?;
        Ok(Arc::new(Self {
            library_path,
            output_dir,
            default_model,
            models,
            worker: Mutex::new(None),
        }))
    }

    fn requests(&self) -> Result<mpsc::UnboundedSender<native::Request>> {
        let mut worker = self
            .worker
            .lock()
            .map_err(|_| anyhow::anyhow!("image worker lock was poisoned"))?;
        if let Some(requests) = worker.as_ref().filter(|requests| !requests.is_closed()) {
            return Ok(requests.clone());
        }
        let (requests, receiver) = mpsc::unbounded_channel();
        let library = self.library_path.clone();
        let output_dir = self.output_dir.clone();
        thread::Builder::new()
            .name("stable-diffusion.cpp".into())
            .spawn(move || native::serve(&library, &output_dir, receiver))
            .context("failed to start the stable-diffusion.cpp inference thread")?;
        *worker = Some(requests.clone());
        Ok(requests)
    }

    async fn render(
        &self,
        model: &Arc<StableDiffusionModelConfig>,
        prompt: &str,
        options: &ImageOptions,
    ) -> Result<PathBuf> {
        ensure!(!prompt.trim().is_empty(), "image prompt cannot be empty");
        ensure!(
            !prompt.contains('\0'),
            "image prompt contains a NUL character"
        );
        validate_options(options)?;
        let defaults = &model.options;
        let options = ImageOptions {
            width: options.width.or(defaults.width),
            height: options.height.or(defaults.height),
            steps: options.steps.or(defaults.steps),
            seed: options.seed.or(defaults.seed),
            negative_prompt: options
                .negative_prompt
                .as_ref()
                .or(defaults.negative_prompt.as_ref())
                .cloned(),
        };
        let (reply, result) = oneshot::channel();
        self.requests()?
            .send(native::Request {
                model: Arc::clone(model),
                prompt: prompt.to_owned(),
                options,
                reply,
            })
            .map_err(|_| anyhow::anyhow!("stable-diffusion.cpp inference thread stopped"))?;
        let artifact = result
            .await
            .context("stable-diffusion.cpp inference thread stopped")??;
        Ok(artifact.keep().join(native::IMAGE_FILENAME))
    }
}

fn absolute(config: &Config, value: &Path) -> Result<PathBuf> {
    ensure!(!value.as_os_str().is_empty(), "path cannot be empty");
    Ok(path::absolute(config.resolve_path(value))?)
}

fn validate_options(options: &ImageOptions) -> Result<()> {
    ensure!(
        options.seed.is_none_or(|seed| seed >= -1),
        "image seed must be -1 or greater"
    );
    ensure!(
        options
            .negative_prompt
            .as_deref()
            .is_none_or(|prompt| !prompt.contains('\0')),
        "negative image prompt contains a NUL character"
    );
    for (name, value) in [
        ("width", options.width),
        ("height", options.height),
        ("steps", options.steps),
    ] {
        ensure!(
            value.is_none_or(|value| value.get() <= i32::MAX as u32),
            "image {name} is too large"
        );
    }
    Ok(())
}

impl Provider for StableDiffusionCpp {
    fn model(&self, _model_id: &str) -> Result<Arc<dyn Model>> {
        bail!(
            "{} provides image models; use an image workflow node or the generate_image tool",
            Self::ID
        )
    }

    fn send<'a>(&'a self, _model_id: &'a str, _prompt: &'a str) -> BoxFuture<'a, Result<Message>> {
        Box::pin(async { bail!("{} does not support text inference", Self::ID) })
    }

    fn image_generator(self: Arc<Self>) -> Option<Arc<dyn ImageGenerator>> {
        Some(self)
    }
}

impl ImageGenerator for StableDiffusionCpp {
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
        options: &'a ImageOptions,
    ) -> BoxFuture<'a, Result<PathBuf>> {
        Box::pin(async move {
            let settings = self
                .models
                .get(model)
                .with_context(|| format!("unknown image model '{model}'"))?;
            self.render(settings, prompt, options)
                .await
                .with_context(|| format!("image generation with model '{model}' failed"))
        })
    }
}

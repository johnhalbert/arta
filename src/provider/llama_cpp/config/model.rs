use crate::config::Config;
use anyhow::{Context, Result, ensure};
use llama_cpp_4::{
    model::params::{LlamaLazyMode, LlamaLoadMode},
    prelude::{
        LlamaBackend, LlamaModelParams, LlamaSplitMode, ParamOverrideValue, supports_gpu_offload,
    },
};
use serde::{Deserialize, Serialize};
use std::{ffi::CString, path::PathBuf, pin::Pin};

#[derive(Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LlamaCPPModelConfig {
    pub id: String,
    pub path: PathBuf,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub additional_shards: Vec<PathBuf>,
    #[serde(default)]
    pub gpu_layers: u32,
    pub main_gpu: Option<u32>,
    pub devices: Option<Vec<String>>,
    pub split_mode: Option<ModelSplitMode>,
    pub tensor_split: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tensor_buffer_overrides: Vec<TensorBufferOverride>,
    pub check_tensors: Option<bool>,
    pub use_extra_bufts: Option<bool>,
    pub no_host: Option<bool>,
    pub load_mode: Option<ModelLoadMode>,
    pub lazy_mode: Option<ModelLazyMode>,
    pub load_mtp: Option<bool>,
    pub kv_override: Option<MetadataOverride>,
}

impl LlamaCPPModelConfig {
    pub(super) fn resolve_paths(&mut self, config: &Config) {
        self.path = config.resolve_path(&self.path);
        for path in &mut self.additional_shards {
            *path = config.resolve_path(path);
        }
    }

    pub(in crate::provider::llama_cpp) fn params(
        &self,
        backend: &LlamaBackend,
    ) -> Result<LlamaModelParams> {
        let gpu_layers = if self.devices.as_ref().is_some_and(Vec::is_empty) {
            0
        } else {
            self.gpu_layers
        };
        ensure!(
            gpu_layers == 0 || supports_gpu_offload(),
            "GPU offload is unavailable. Use a build with a working CUDA, HIP, or Vulkan backend, or set devices = [] for CPU-only inference"
        );
        let mut params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers);
        if let Some(main_gpu) = self.main_gpu {
            params =
                params.with_main_gpu(i32::try_from(main_gpu).context("main_gpu is too large")?);
        }
        if let Some(mode) = self.split_mode {
            params = params.with_split_mode(mode.into());
        }
        if let Some(devices) = &self.devices {
            if let Some(weights) = &self.tensor_split {
                ensure!(
                    weights.len() <= devices.len(),
                    "tensor_split has more weights than the explicitly selected devices"
                );
            }
            ensure!(
                !devices.is_empty() || params.split_mode() != LlamaSplitMode::Tensor,
                "split_mode = \"tensor\" requires at least one device"
            );
            params = params
                .with_devices(backend, devices)
                .context("invalid devices")?;
        }
        if let Some(weights) = &self.tensor_split {
            params = params
                .with_tensor_split(weights)
                .context("invalid tensor_split")?;
        }
        if !self.tensor_buffer_overrides.is_empty() {
            params = params
                .with_tensor_buffer_overrides(
                    backend,
                    self.tensor_buffer_overrides
                        .iter()
                        .map(|entry| (entry.pattern.as_str(), entry.buffer_type.as_str())),
                )
                .context("invalid tensor_buffer_overrides")?;
        }
        if let Some(enabled) = self.check_tensors {
            params = params.with_check_tensors(enabled);
        }
        if let Some(enabled) = self.use_extra_bufts {
            params = params.with_use_extra_bufts(enabled);
        }
        if let Some(enabled) = self.no_host {
            params = params.with_no_host(enabled);
        }
        if let Some(mode) = self.load_mode {
            params = params.with_load_mode(mode.into());
        }
        if let Some(mode) = self.lazy_mode {
            params = params.with_lazy_mode(mode.into());
        }
        if let Some(load_mtp) = self.load_mtp {
            params = params.with_load_mtp(load_mtp);
        }
        if let Some(item) = &self.kv_override {
            ensure!(
                !item.key.is_empty() && item.key.len() < 128 && item.key.is_ascii(),
                "kv_override.key must be between 1 and 127 ASCII bytes"
            );
            let key = CString::new(item.key.as_str()).context("invalid kv_override.key")?;
            Pin::new(&mut params).append_kv_override(&key, item.value.native()?);
        }
        Ok(params)
    }
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSplitMode {
    None,
    Layer,
    Row,
    Tensor,
}

impl From<ModelSplitMode> for LlamaSplitMode {
    fn from(value: ModelSplitMode) -> Self {
        match value {
            ModelSplitMode::None => Self::None,
            ModelSplitMode::Layer => Self::Layer,
            ModelSplitMode::Row => Self::Row,
            ModelSplitMode::Tensor => Self::Tensor,
        }
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TensorBufferOverride {
    pub pattern: String,
    pub buffer_type: String,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelLoadMode {
    Auto,
    None,
    Mmap,
    Mlock,
    #[serde(rename = "mmap+mlock")]
    MmapMlock,
    #[serde(rename = "dio")]
    DirectIo,
}

impl From<ModelLoadMode> for LlamaLoadMode {
    fn from(value: ModelLoadMode) -> Self {
        match value {
            ModelLoadMode::Auto => Self::Auto,
            ModelLoadMode::None => Self::None,
            ModelLoadMode::Mmap => Self::Mmap,
            ModelLoadMode::Mlock => Self::Mlock,
            ModelLoadMode::MmapMlock => Self::MmapMlock,
            ModelLoadMode::DirectIo => Self::DirectIo,
        }
    }
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelLazyMode {
    Off,
    Auto,
    On,
}

impl From<ModelLazyMode> for LlamaLazyMode {
    fn from(value: ModelLazyMode) -> Self {
        match value {
            ModelLazyMode::Off => Self::Off,
            ModelLazyMode::Auto => Self::Auto,
            ModelLazyMode::On => Self::On,
        }
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MetadataOverride {
    pub key: String,
    pub value: MetadataValue,
}

#[derive(Deserialize, Serialize)]
#[serde(untagged)]
pub enum MetadataValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
}

impl MetadataValue {
    fn native(&self) -> Result<ParamOverrideValue> {
        Ok(match self {
            Self::Bool(value) => ParamOverrideValue::Bool(*value),
            Self::Int(value) => ParamOverrideValue::Int(*value),
            Self::Float(value) => {
                ensure!(value.is_finite(), "kv_override.value must be finite");
                ParamOverrideValue::Float(*value)
            }
            Self::String(value) => {
                ensure!(
                    value.len() < 128 && !value.contains('\0'),
                    "kv_override.value must fit in 127 bytes and contain no NUL"
                );
                let mut bytes = [0; 128];
                for (target, byte) in bytes.iter_mut().zip(value.bytes()) {
                    *target = byte as std::ffi::c_char;
                }
                ParamOverrideValue::Str(bytes)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placement_settings_reach_the_native_parameters() -> Result<()> {
        let backend = LlamaBackend::init()?;
        let default = LlamaCPPModelConfig::default().params(&backend)?;
        assert_eq!(default.split_mode(), LlamaSplitMode::Layer);
        assert!(!default.check_tensors());
        assert!(default.use_extra_bufts());
        assert!(!default.no_host());

        let mut config: LlamaCPPModelConfig = toml::from_str(
            r#"
id = "local"
path = "model.gguf"
gpu_layers = 99
devices = []
split_mode = "none"
check_tensors = true
use_extra_bufts = false
no_host = true
tensor_buffer_overrides = [{ pattern = '^blk\.0\.', buffer_type = "CPU" }]
"#,
        )?;
        let params = config.params(&backend)?;
        assert_eq!(params.n_gpu_layers(), 0);
        assert_eq!(params.split_mode(), LlamaSplitMode::None);
        assert!(params.check_tensors());
        assert!(!params.use_extra_bufts());
        assert!(params.no_host());

        config.gpu_layers = 0;
        config.devices = Some(vec!["missing_device".into()]);
        let error = config.params(&backend).unwrap_err();
        assert!(format!("{error:#}").contains("unknown offload device"));

        config.devices = None;
        config.tensor_split = Some(vec![-1.0]);
        let error = config.params(&backend).unwrap_err();
        assert!(format!("{error:#}").contains("weights must be nonnegative"));

        config.tensor_split = None;
        config.tensor_buffer_overrides[0].buffer_type = "missing_buffer".into();
        let error = config.params(&backend).unwrap_err();
        assert!(format!("{error:#}").contains("unknown tensor buffer type"));
        Ok(())
    }
}

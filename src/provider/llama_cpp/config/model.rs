//! Translates one model's TOML entry into native weight-loading parameters. Resolves file
//! paths before loading and checks placement against the initialized backend.

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

/// Per-model identity, files, placement, and metadata settings. Optional fields preserve
/// their native defaults.
#[derive(Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LlamaCPPModelConfig {
    /// Nonblank provider-local name used by the harness and agent selections.
    pub id: String,
    /// Primary GGUF file, also the first path passed for a split model.
    pub path: PathBuf,
    /// Remaining split files in loader order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub additional_shards: Vec<PathBuf>,
    /// Number of layers requested for GPU offload. Defaults to zero.
    #[serde(default)]
    pub gpu_layers: u32,
    /// Zero-based device index passed to the native loader as a signed integer.
    pub main_gpu: Option<u32>,
    /// Ordered offload device names. An empty list forces zero GPU layers. Omission leaves
    /// device selection to the backend.
    pub devices: Option<Vec<String>>,
    /// How weights are distributed over the selected devices.
    pub split_mode: Option<ModelSplitMode>,
    /// Relative offload weights in device order. When devices are explicit the number of
    /// weights must fit that list.
    pub tensor_split: Option<Vec<f32>>,
    /// Ordered regex-to-buffer assignments. The loader uses the first matching pattern.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tensor_buffer_overrides: Vec<TensorBufferOverride>,
    /// Validate tensor data during model loading.
    pub check_tensors: Option<bool>,
    /// Allow extra buffer types for repacking weights.
    pub use_extra_bufts: Option<bool>,
    /// Bypass the host buffer during native buffer selection.
    pub no_host: Option<bool>,
    /// File mapping, locking, or direct I/O strategy.
    pub load_mode: Option<ModelLoadMode>,
    /// Policy for loading rows of architecture-marked tensors on demand.
    pub lazy_mode: Option<ModelLazyMode>,
    /// Load multi-token prediction weights when available. Arta's generation loop still
    /// samples one token at a time.
    pub load_mtp: Option<bool>,
    /// Single GGUF metadata entry to override while loading this model.
    pub kv_override: Option<MetadataOverride>,
}

impl LlamaCPPModelConfig {
    /// Resolve the primary model and every named shard relative to the TOML file.
    pub(super) fn resolve_paths(&mut self, config: &Config) {
        self.path = config.resolve_path(&self.path);
        for path in &mut self.additional_shards {
            *path = config.resolve_path(path);
        }
    }

    /// Build native parameters after checking offload availability, split dimensions, and C
    /// string bounds. The wrapper owns backing storage for placement arrays and metadata
    /// values.
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

/// Device distribution strategy forwarded to llama.cpp.
#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSplitMode {
    /// Use the main device for offloaded weights.
    None,
    /// Distribute layers and KV cache among devices.
    Layer,
    /// Distribute weights by row on compatible backends.
    Row,
    /// Request experimental tensor parallelism.
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

/// One loader rule assigning matching tensors to a named backend buffer type.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TensorBufferOverride {
    /// Regular expression matched against tensor names by llama.cpp.
    pub pattern: String,
    /// Name of a registered backend buffer type.
    pub buffer_type: String,
}

/// File access strategy for loading GGUF weights.
#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelLoadMode {
    /// Let the loader choose mapping or ordinary reads from backend capabilities.
    Auto,
    /// Read files normally without memory mapping or locking.
    None,
    /// Request memory-mapped files.
    Mmap,
    /// Read files normally and lock loaded pages in memory.
    Mlock,
    /// Map model files and lock their pages.
    #[serde(rename = "mmap+mlock")]
    MmapMlock,
    /// Request direct I/O when supported.
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

/// On-demand loading for tensors marked by the model architecture.
#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelLazyMode {
    /// Load marked tensors up front.
    Off,
    /// Use the loader's size and device capability heuristics.
    Auto,
    /// Request on-demand reads for all marked tensors.
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

/// GGUF header override encoded in the native parameter block.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MetadataOverride {
    /// One to 127 ASCII bytes without an embedded NUL.
    pub key: String,
    /// Replacement value whose TOML type selects its native representation.
    pub value: MetadataValue,
}

/// Scalar TOML values supported by the GGUF metadata override.
#[derive(Deserialize, Serialize)]
#[serde(untagged)]
pub enum MetadataValue {
    /// Native boolean value.
    Bool(bool),
    /// Signed 64-bit integer.
    Int(i64),
    /// Finite double-precision value.
    Float(f64),
    /// Up to 127 UTF-8 bytes stored in a zero-padded 128-byte native buffer.
    String(String),
}

impl MetadataValue {
    /// Validate the scalar and convert it to the tagged native override representation.
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

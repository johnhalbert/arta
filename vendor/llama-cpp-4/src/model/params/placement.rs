use super::LlamaModelParams;
use crate::{llama_backend::LlamaBackend, max_devices, max_tensor_buft_overrides};
use llama_cpp_sys_4 as sys;
use std::collections::{BTreeMap, HashSet};
use std::ffi::{CStr, CString};
use std::ptr::{null, null_mut};
use thiserror::Error;

/// How llama.cpp distributes a model across offload devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaSplitMode {
    /// Use only the main GPU.
    None,
    /// Distribute layers and KV cache across devices.
    Layer,
    /// Distribute weights by rows when the backend supports it.
    Row,
    /// Use experimental tensor parallelism.
    Tensor,
}

impl From<LlamaSplitMode> for sys::llama_split_mode {
    fn from(mode: LlamaSplitMode) -> Self {
        match mode {
            LlamaSplitMode::None => sys::LLAMA_SPLIT_MODE_NONE,
            LlamaSplitMode::Layer => sys::LLAMA_SPLIT_MODE_LAYER,
            LlamaSplitMode::Row => sys::LLAMA_SPLIT_MODE_ROW,
            LlamaSplitMode::Tensor => sys::LLAMA_SPLIT_MODE_TENSOR,
        }
    }
}

/// Invalid explicit model placement settings.
#[derive(Debug, Error)]
pub enum ModelPlacementError {
    /// The device list exceeds llama.cpp's limit.
    #[error("at most {0} offload devices can be selected")]
    TooManyDevices(usize),
    /// The backend did not register an offload device by this name.
    #[error("unknown offload device {0:?}. Use a registered non-CPU backend device name")]
    UnknownDevice(String),
    /// The same device was selected more than once.
    #[error("offload device {0:?} is listed more than once")]
    DuplicateDevice(String),
    /// The weight array exceeds llama.cpp's device limit.
    #[error("at most {0} tensor split weights can be supplied")]
    TooManySplitWeights(usize),
    /// The weights cannot be used as proportions.
    #[error("tensor split weights must be nonnegative and have a finite sum")]
    InvalidSplitWeights,
    /// The number of buffer overrides exceeds llama.cpp's limit.
    #[error("at most {0} tensor buffer overrides can be supplied")]
    TooManyBufferOverrides(usize),
    /// The pattern cannot be passed to llama.cpp as a C string.
    #[error("tensor buffer override pattern {0:?} must be nonempty and contain no NUL")]
    InvalidPattern(String),
    /// The named buffer type is unavailable.
    #[error("unknown tensor buffer type {name:?}. Available types are {available}")]
    UnknownBufferType {
        /// Requested buffer type.
        name: String,
        /// Buffer types offered by the registered devices.
        available: String,
    },
}

#[derive(Default)]
pub(super) struct PlacementStorage {
    devices: Box<[sys::ggml_backend_dev_t]>,
    tensor_split: Box<[f32]>,
    _patterns: Vec<CString>,
    buffer_overrides: Box<[sys::llama_model_tensor_buft_override]>,
}

impl LlamaModelParams {
    /// Return the model's requested split mode.
    #[must_use]
    pub fn split_mode(&self) -> LlamaSplitMode {
        match self.params.split_mode {
            sys::LLAMA_SPLIT_MODE_NONE => LlamaSplitMode::None,
            sys::LLAMA_SPLIT_MODE_ROW => LlamaSplitMode::Row,
            sys::LLAMA_SPLIT_MODE_TENSOR => LlamaSplitMode::Tensor,
            _ => LlamaSplitMode::Layer,
        }
    }

    /// Set how to distribute the model across offload devices.
    #[must_use]
    pub fn with_split_mode(mut self, mode: LlamaSplitMode) -> Self {
        self.params.split_mode = mode.into();
        self
    }

    /// Whether to validate tensor data while loading.
    #[must_use]
    pub fn check_tensors(&self) -> bool {
        self.params.check_tensors
    }

    /// Enable or disable validation of tensor data while loading.
    #[must_use]
    pub fn with_check_tensors(mut self, enabled: bool) -> Self {
        self.params.check_tensors = enabled;
        self
    }

    /// Whether to consider extra buffer types for weight repacking.
    #[must_use]
    pub fn use_extra_bufts(&self) -> bool {
        self.params.use_extra_bufts
    }

    /// Enable or disable extra buffer types for weight repacking.
    #[must_use]
    pub fn with_use_extra_bufts(mut self, enabled: bool) -> Self {
        self.params.use_extra_bufts = enabled;
        self
    }

    /// Whether to bypass the host buffer.
    #[must_use]
    pub fn no_host(&self) -> bool {
        self.params.no_host
    }

    /// Bypass the host buffer to allow extra buffer types to be used.
    #[must_use]
    pub fn with_no_host(mut self, enabled: bool) -> Self {
        self.params.no_host = enabled;
        self
    }

    /// Select offload devices by backend name in the requested order.
    /// An empty list disables offload. The default uses automatic device selection.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown, CPU, duplicate, or excess devices.
    pub fn with_devices(
        mut self,
        _backend: &LlamaBackend,
        names: &[impl AsRef<str>],
    ) -> Result<Self, ModelPlacementError> {
        let limit = max_devices();
        if names.len() > limit {
            return Err(ModelPlacementError::TooManyDevices(limit));
        }

        let mut devices = Vec::with_capacity(names.len() + 1);
        let mut seen = HashSet::with_capacity(names.len());
        for name in names {
            let name = name.as_ref();
            let c_name =
                CString::new(name).map_err(|_| ModelPlacementError::UnknownDevice(name.into()))?;
            // SAFETY: the backend is initialized and the name is a live C string.
            let device = unsafe { sys::ggml_backend_dev_by_name(c_name.as_ptr()) };
            // SAFETY: any non-null device returned by the registry is valid here.
            if device.is_null()
                || unsafe { sys::ggml_backend_dev_type(device) }
                    == sys::GGML_BACKEND_DEVICE_TYPE_CPU
            {
                return Err(ModelPlacementError::UnknownDevice(name.into()));
            }
            if !seen.insert(device) {
                return Err(ModelPlacementError::DuplicateDevice(name.into()));
            }
            devices.push(device);
        }
        devices.push(null_mut());
        self.placement.devices = devices.into_boxed_slice();
        self.params.devices = self.placement.devices.as_mut_ptr();
        Ok(self)
    }

    /// Set relative offload weights in the same order as the selected devices.
    /// Missing weights are padded with zeros. An empty list restores automatic allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if the weights have a negative element, a non-finite sum,
    /// or exceed the maximum device count.
    pub fn with_tensor_split(mut self, weights: &[f32]) -> Result<Self, ModelPlacementError> {
        let limit = max_devices();
        if weights.len() > limit {
            return Err(ModelPlacementError::TooManySplitWeights(limit));
        }
        if weights.iter().any(|&weight| weight < 0.0) || !weights.iter().sum::<f32>().is_finite() {
            return Err(ModelPlacementError::InvalidSplitWeights);
        }
        self.params.tensor_split = if weights.is_empty() {
            self.placement.tensor_split = Box::default();
            null()
        } else {
            let mut padded = vec![0.0; limit].into_boxed_slice();
            padded[..weights.len()].copy_from_slice(weights);
            self.placement.tensor_split = padded;
            self.placement.tensor_split.as_ptr()
        };
        Ok(self)
    }

    /// Set ordered pairs of tensor name regexes and backend buffer type names.
    /// llama.cpp applies the first matching entry. An empty list clears all overrides.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown buffer type, invalid pattern string, or
    /// too many overrides. llama.cpp checks regex syntax when it loads the model.
    pub fn with_tensor_buffer_overrides<'a>(
        mut self,
        _backend: &LlamaBackend,
        overrides: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Result<Self, ModelPlacementError> {
        let limit = max_tensor_buft_overrides();
        let buffer_types = registered_buffer_types();
        let mut patterns = Vec::new();
        let mut entries = Vec::new();
        for (pattern, name) in overrides {
            if entries.len() == limit {
                return Err(ModelPlacementError::TooManyBufferOverrides(limit));
            }
            let pattern = CString::new(pattern)
                .ok()
                .filter(|value| !value.as_bytes().is_empty())
                .ok_or_else(|| ModelPlacementError::InvalidPattern(pattern.into()))?;
            let &buft =
                buffer_types
                    .get(name)
                    .ok_or_else(|| ModelPlacementError::UnknownBufferType {
                        name: name.into(),
                        available: buffer_types.keys().cloned().collect::<Vec<_>>().join(", "),
                    })?;
            entries.push(sys::llama_model_tensor_buft_override {
                pattern: pattern.as_ptr(),
                buft,
            });
            patterns.push(pattern);
        }

        self.params.tensor_buft_overrides = if entries.is_empty() {
            self.placement.buffer_overrides = Box::default();
            null()
        } else {
            entries.push(sys::llama_model_tensor_buft_override {
                pattern: null(),
                buft: null_mut(),
            });
            self.placement.buffer_overrides = entries.into_boxed_slice();
            self.placement.buffer_overrides.as_ptr()
        };
        // The raw entries borrow each CString's heap allocation through model loading.
        self.placement._patterns = patterns;
        Ok(self)
    }
}

fn registered_buffer_types() -> BTreeMap<String, sys::ggml_backend_buffer_type_t> {
    let mut buffer_types = BTreeMap::new();
    // SAFETY: the initialized backend registry owns its devices, default buffer
    // types and NUL-terminated names. The loop stays within the device count.
    unsafe {
        for index in 0..sys::ggml_backend_dev_count() {
            let device = sys::ggml_backend_dev_get(index);
            let buffer_type = sys::ggml_backend_dev_buffer_type(device);
            if buffer_type.is_null() {
                continue;
            }
            let name = CStr::from_ptr(sys::ggml_backend_buft_name(buffer_type));
            buffer_types.insert(name.to_string_lossy().into_owned(), buffer_type);
        }
    }
    buffer_types
}

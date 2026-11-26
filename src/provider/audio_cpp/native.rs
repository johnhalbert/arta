//! Binding to audio.cpp's C ABI 0.4. All native handles stay on the inference thread.

use super::{AudioBackend, AudioModelConfig, AudioOptions, AudioTask};
use crate::provider::native::load_library;
use anyhow::{Context, Result, bail, ensure};
use libloading::Library;
use std::{
    ffi::{CStr, CString, c_char, c_int},
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
    ptr::{self, NonNull},
    slice,
    sync::Arc,
};
use tempfile::TempDir;
use tokio::sync::{mpsc, oneshot};

pub(super) const AUDIO_FILENAME: &str = "audio.wav";

pub(super) struct Request {
    pub model: Arc<AudioModelConfig>,
    pub prompt: String,
    pub options: AudioOptions,
    pub reply: oneshot::Sender<Result<TempDir>>,
}

pub(super) fn serve(
    library: &Path,
    output: &Path,
    backend: AudioBackend,
    threads: c_int,
    mut requests: mpsc::UnboundedReceiver<Request>,
) {
    let mut runtime = None;
    while let Some(request) = requests.blocking_recv() {
        if request.reply.is_closed() {
            continue;
        }
        let result = (|| {
            if runtime.is_none() {
                runtime = Some(Runtime::load(library, backend, threads)?);
            }
            runtime.as_mut().unwrap().render(output, &request)
        })();
        // An abandoned request drops its temporary directory, including any partial WAV.
        let _ = request.reply.send(result);
    }
}

struct Runtime {
    // Drop the session and registry before unloading their library.
    active: Option<(String, Owned<Session>)>,
    registry: Owned<Registry>,
    api: Api,
    backend: BackendConfig,
}

impl Runtime {
    fn load(library: &Path, backend: AudioBackend, threads: c_int) -> Result<Self> {
        let api = Api::load(library)
            .with_context(|| format!("failed to load audio.cpp from '{}'", library.display()))?;
        let version = unsafe { (api.audiocpp_abi_version)() };
        ensure!(
            version >> 16 == 0 && version >= 0x0400,
            "expected audio.cpp C ABI 0.4 or newer within major version 0, found {version:#x}"
        );
        let mut registry = ptr::null_mut();
        let status = unsafe { (api.audiocpp_registry_create)(ptr::null(), &mut registry) };
        api.check(status)
            .context("failed to create audio.cpp registry")?;
        let registry = Owned::new(registry, api.audiocpp_registry_free)
            .context("audio.cpp returned no registry")?;
        let backend = match backend {
            AudioBackend::Cpu => c"cpu",
            AudioBackend::Cuda => c"cuda",
            AudioBackend::Hip => c"hip",
            AudioBackend::Vulkan => c"vulkan",
            AudioBackend::Metal => c"metal",
        };
        Ok(Self {
            active: None,
            registry,
            api,
            backend: BackendConfig {
                backend: backend.as_ptr(),
                device: 0,
                threads,
            },
        })
    }

    fn render(&mut self, output: &Path, request: &Request) -> Result<TempDir> {
        let model = &request.model;
        if !self.active.as_ref().is_some_and(|(id, _)| *id == model.id) {
            drop(self.active.take());
            let session = self
                .api
                .session(self.registry.as_ptr(), model, &self.backend)
                .with_context(|| format!("failed to open audio model '{}'", model.id))?;
            self.active = Some((model.id.clone(), session));
        }
        let (_, session) = self.active.as_ref().unwrap();
        ensure!(!request.reply.is_closed(), "audio generation cancelled");
        fs::create_dir_all(output)
            .with_context(|| format!("failed to create audio directory '{}'", output.display()))?;
        let artifact = tempfile::Builder::new()
            .prefix("audio-")
            .tempdir_in(output)?;
        let result = self.api.generate(session.as_ptr(), request)?;
        ensure!(!request.reply.is_closed(), "audio generation cancelled");
        self.api
            .write_wav(&result, &artifact.path().join(AUDIO_FILENAME))?;
        Ok(artifact)
    }
}

#[repr(C)]
struct Registry {
    _opaque: [u8; 0],
}
#[repr(C)]
struct Model {
    _opaque: [u8; 0],
}
#[repr(C)]
struct Options {
    _opaque: [u8; 0],
}
#[repr(C)]
struct Session {
    _opaque: [u8; 0],
}
#[repr(C)]
struct NativeRequest {
    _opaque: [u8; 0],
}
#[repr(C)]
struct NativeResult {
    _opaque: [u8; 0],
}

#[repr(C)]
struct ModelConfig {
    family_hint: *const c_char,
    config_id: *const c_char,
    weight_id: *const c_char,
    model_spec_override: *const c_char,
}

#[repr(C)]
struct BackendConfig {
    backend: *const c_char,
    device: c_int,
    threads: c_int,
}

struct Owned<T> {
    ptr: NonNull<T>,
    free: unsafe extern "C" fn(*mut T),
}

impl<T> Owned<T> {
    fn new(ptr: *mut T, free: unsafe extern "C" fn(*mut T)) -> Option<Self> {
        Some(Self {
            ptr: NonNull::new(ptr)?,
            free,
        })
    }

    fn as_ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }
}

impl<T> Drop for Owned<T> {
    fn drop(&mut self) {
        unsafe { (self.free)(self.as_ptr()) }
    }
}

// Declare and load each symbol with its matching C signature.
macro_rules! api {
    ($(fn $name:ident($($param:ty),*) $(-> $ret:ty)?;)*) => {
        struct Api {
            $($name: unsafe extern "C" fn($($param),*) $(-> $ret)?,)*
            _library: Library,
        }

        impl Api {
            fn load(path: &Path) -> Result<Self> {
                let library = load_library(path)?;
                Ok(Self {
                    $($name: *unsafe { library.get(concat!(stringify!($name), "\0").as_bytes()) }
                        .with_context(|| concat!("missing audio.cpp C API symbol ", stringify!($name)))?,)*
                    _library: library,
                })
            }
        }
    };
}

api! {
    fn audiocpp_abi_version() -> u32;
    fn audiocpp_last_error() -> *const c_char;
    fn audiocpp_registry_create(*const c_char, *mut *mut Registry) -> c_int;
    fn audiocpp_registry_free(*mut Registry);
    fn audiocpp_model_load(*mut Registry, *const c_char, *const ModelConfig, *const Options, *mut *mut Model) -> c_int;
    fn audiocpp_model_free(*mut Model);
    fn audiocpp_session_create(*const Model, *const c_char, *const c_char, *const BackendConfig, *const Options, *mut *mut Session) -> c_int;
    fn audiocpp_session_free(*mut Session);
    fn audiocpp_request_create() -> *mut NativeRequest;
    fn audiocpp_request_free(*mut NativeRequest);
    fn audiocpp_request_set_text(*mut NativeRequest, *const c_char, *const c_char) -> c_int;
    fn audiocpp_request_set_option(*mut NativeRequest, *const c_char, *const c_char) -> c_int;
    fn audiocpp_session_run(*mut Session, *const NativeRequest, *mut *mut NativeResult) -> c_int;
    fn audiocpp_result_free(*mut NativeResult);
    fn audiocpp_result_audio(*const NativeResult, *mut *const f32, *mut usize, *mut c_int, *mut c_int) -> c_int;
}

impl Api {
    fn check(&self, status: c_int) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        let error = unsafe { CStr::from_ptr((self.audiocpp_last_error)()) }.to_string_lossy();
        bail!("audio.cpp error {status}: {error}");
    }

    fn session(
        &self,
        registry: *mut Registry,
        model: &AudioModelConfig,
        backend: &BackendConfig,
    ) -> Result<Owned<Session>> {
        let metadata = model.path.metadata().with_context(|| {
            format!(
                "could not access audio model path '{}'",
                model.path.display()
            )
        })?;
        ensure!(
            metadata.is_file() || metadata.is_dir(),
            "audio model path must be a file or directory"
        );
        let path = CString::new(
            model
                .path
                .to_str()
                .context("audio model path is not UTF-8")?,
        )?;
        let family = model.family.as_deref().map(CString::new).transpose()?;
        let config = ModelConfig {
            family_hint: family.as_ref().map_or(ptr::null(), |value| value.as_ptr()),
            config_id: ptr::null(),
            weight_id: ptr::null(),
            model_spec_override: ptr::null(),
        };
        let mut loaded = ptr::null_mut();
        let status = unsafe {
            (self.audiocpp_model_load)(registry, path.as_ptr(), &config, ptr::null(), &mut loaded)
        };
        self.check(status)?;
        let loaded =
            Owned::new(loaded, self.audiocpp_model_free).context("audio.cpp returned no model")?;
        let task = match model.task {
            AudioTask::Tts => c"tts",
            AudioTask::Gen => c"gen",
        };
        let mut session = ptr::null_mut();
        let status = unsafe {
            (self.audiocpp_session_create)(
                loaded.as_ptr(),
                task.as_ptr(),
                c"offline".as_ptr(),
                backend,
                ptr::null(),
                &mut session,
            )
        };
        self.check(status).with_context(|| {
            format!(
                "failed to create offline {} session",
                task.to_string_lossy()
            )
        })?;
        // The session retains the loaded model internally.
        Owned::new(session, self.audiocpp_session_free).context("audio.cpp returned no session")
    }

    fn generate(&self, session: *mut Session, request: &Request) -> Result<Owned<NativeResult>> {
        let native_request = Owned::new(
            unsafe { (self.audiocpp_request_create)() },
            self.audiocpp_request_free,
        )
        .context("audio.cpp could not allocate a request")?;
        let prompt = CString::new(request.prompt.as_str())?;
        let language = request
            .options
            .get("language")
            .map(|value| CString::new(value.as_str()))
            .transpose()?;
        let status = unsafe {
            (self.audiocpp_request_set_text)(
                native_request.as_ptr(),
                prompt.as_ptr(),
                language
                    .as_ref()
                    .map_or(ptr::null(), |value| value.as_ptr()),
            )
        };
        self.check(status).context("failed to set audio prompt")?;
        for (key, value) in &request.options {
            let name = CString::new(key.as_str())?;
            let value = CString::new(value.as_str())?;
            let status = unsafe {
                (self.audiocpp_request_set_option)(
                    native_request.as_ptr(),
                    name.as_ptr(),
                    value.as_ptr(),
                )
            };
            self.check(status)
                .with_context(|| format!("failed to set audio option '{key}'"))?;
        }
        let mut result = ptr::null_mut();
        let status =
            unsafe { (self.audiocpp_session_run)(session, native_request.as_ptr(), &mut result) };
        self.check(status).context("audio generation failed")?;
        Owned::new(result, self.audiocpp_result_free).context("audio.cpp returned no result")
    }

    fn write_wav(&self, result: &Owned<NativeResult>, path: &Path) -> Result<()> {
        let (mut samples, mut frames, mut rate, mut channels) = (ptr::null(), 0, 0, 0);
        let status = unsafe {
            (self.audiocpp_result_audio)(
                result.as_ptr(),
                &mut samples,
                &mut frames,
                &mut rate,
                &mut channels,
            )
        };
        self.check(status)
            .context("could not read generated audio")?;
        ensure!(
            !samples.is_null() && frames > 0 && rate > 0 && channels > 0,
            "audio.cpp returned empty or invalid PCM"
        );
        let rate = rate as u32;
        let channels = u16::try_from(channels).context("too many WAV channels")?;
        let block = channels.checked_mul(2).context("WAV frame is too large")?;
        let byte_rate = rate
            .checked_mul(u32::from(block))
            .context("WAV byte rate is too large")?;
        let data_len = u32::try_from(frames)
            .ok()
            .and_then(|frames| frames.checked_mul(u32::from(block)))
            .context("audio is too long for a WAV file")?;
        let riff_len = data_len
            .checked_add(36)
            .context("audio is too long for a WAV file")?;
        let count = (data_len / 2) as usize;
        ensure!(
            count <= isize::MAX as usize / size_of::<f32>(),
            "audio sample buffer is too large"
        );
        // The C API returns interleaved f32 samples borrowed from the result handle.
        let pcm = unsafe { slice::from_raw_parts(samples, count) };
        let mut wav = BufWriter::new(File::create(path)?);
        wav.write_all(b"RIFF")?;
        wav.write_all(&riff_len.to_le_bytes())?;
        wav.write_all(b"WAVEfmt ")?;
        wav.write_all(&16u32.to_le_bytes())?;
        wav.write_all(&1u16.to_le_bytes())?; // PCM
        wav.write_all(&channels.to_le_bytes())?;
        wav.write_all(&rate.to_le_bytes())?;
        wav.write_all(&byte_rate.to_le_bytes())?;
        wav.write_all(&block.to_le_bytes())?;
        wav.write_all(&16u16.to_le_bytes())?;
        wav.write_all(b"data")?;
        wav.write_all(&data_len.to_le_bytes())?;
        for &sample in pcm {
            let value = (sample * 32768.0).round() as i16;
            wav.write_all(&value.to_le_bytes())?;
        }
        Ok(wav.flush()?)
    }
}

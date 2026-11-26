//! Minimal binding to the stable-diffusion.cpp opaque C API in master-1133-52491bd.
//! One thread owns the library and its context. Native handles never cross threads.

use super::{ImageOptions, StableDiffusionModelConfig};
use crate::provider::native::load_library;
use anyhow::{Context, Result, bail, ensure};
use libloading::Library;
use std::{
    ffi::{CStr, CString, c_char, c_int, c_void},
    fs::{self, File},
    io::BufWriter,
    path::Path,
    ptr::{self, NonNull},
    slice,
    sync::{Arc, Mutex, PoisonError},
};
use tempfile::TempDir;
use tokio::sync::{mpsc, oneshot};

pub(super) const IMAGE_FILENAME: &str = "image.png";

// The library has process-wide progress callbacks and shared backend state.
static NATIVE_CALLS: Mutex<()> = Mutex::new(());

pub(super) struct Request {
    pub model: Arc<StableDiffusionModelConfig>,
    pub prompt: String,
    pub options: ImageOptions,
    pub reply: oneshot::Sender<Result<TempDir>>,
}

pub(super) fn serve(library: &Path, output: &Path, mut requests: mpsc::UnboundedReceiver<Request>) {
    let mut runtime = None;
    while let Some(request) = requests.blocking_recv() {
        let _native = NATIVE_CALLS.lock().unwrap_or_else(PoisonError::into_inner);
        if request.reply.is_closed() {
            continue;
        }
        let result = (|| {
            if runtime.is_none() {
                runtime = Some(Runtime::load(library)?);
            }
            runtime.as_mut().unwrap().render(output, &request)
        })();
        // The receiver keeps the directory only after accepting a complete image. A failed
        // job or a dropped receiver removes any partial or abandoned artifact.
        let _ = request.reply.send(result);
    }
    let _native = NATIVE_CALLS.lock().unwrap_or_else(PoisonError::into_inner);
    drop(runtime);
}

struct Runtime {
    // Drop the native context before unloading its library.
    active: Option<(String, Owned<SdContext>)>,
    api: Api,
}

impl Runtime {
    fn load(library: &Path) -> Result<Self> {
        let api = Api::load(library).with_context(|| {
            format!(
                "failed to load stable-diffusion.cpp from '{}'",
                library.display()
            )
        })?;
        // The pinned runtime writes legacy loading and sampling bars to stdout unless a
        // callback is registered. Arta reports its own progress and keeps stdout for paths.
        unsafe { (api.sd_set_progress_callback)(Some(ignore_console_progress), ptr::null_mut()) };
        Ok(Self { active: None, api })
    }

    fn render(&mut self, output: &Path, request: &Request) -> Result<TempDir> {
        let model = &request.model;
        if !self.active.as_ref().is_some_and(|(id, _)| *id == model.id) {
            // Release the previous model's memory before loading another checkpoint.
            drop(self.active.take());
            self.active = Some((model.id.clone(), self.api.context(model)?));
        }
        let (_, context) = self.active.as_ref().unwrap();
        ensure!(!request.reply.is_closed(), "image generation cancelled");
        fs::create_dir_all(output)
            .with_context(|| format!("failed to create image directory '{}'", output.display()))?;
        let artifact = tempfile::Builder::new()
            .prefix("image-")
            .tempdir_in(output)?;
        let result = self.api.generate(context.as_ptr(), request)?;
        ensure!(!request.reply.is_closed(), "image generation cancelled");
        self.api
            .write_png(&result, &artifact.path().join(IMAGE_FILENAME))?;
        Ok(artifact)
    }
}

#[repr(C)]
struct SdOpt {
    _opaque: [u8; 0],
}

#[repr(C)]
struct SdContext {
    _opaque: [u8; 0],
}

#[repr(C)]
struct SdResult {
    _opaque: [u8; 0],
}

#[repr(C)]
struct SdTensor {
    _opaque: [u8; 0],
}

type SdStatus = c_int;
type ProgressCallback = unsafe extern "C" fn(*const c_void, *mut c_void);
type LegacyProgressCallback = unsafe extern "C" fn(c_int, c_int, f32, *mut c_void);
const SD_STATUS_OK: SdStatus = 0;
const SD_CANCEL_ALL: c_int = 0;
const SD_TENSOR_U8: i32 = 1;

// Keeping each symbol's declaration and lookup together makes the required ABI explicit.
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
                        .with_context(|| concat!("missing stable-diffusion.cpp C API symbol ", stringify!($name)))?,)*
                    _library: library,
                })
            }
        }
    };
}

api! {
    fn sd_get_last_error() -> *const c_char;
    fn sd_ctx_opt_create() -> *mut SdOpt;
    fn sd_img_gen_opt_create() -> *mut SdOpt;
    fn sd_opt_set_string(*mut SdOpt, *const c_char, *const c_char) -> SdStatus;
    fn sd_opt_set_int(*mut SdOpt, *const c_char, i64) -> SdStatus;
    fn sd_opt_free(*mut SdOpt);
    fn sd_ctx_create(*const SdOpt, *mut *mut SdContext) -> SdStatus;
    fn sd_ctx_free(*mut SdContext);
    fn sd_set_progress_callback(Option<LegacyProgressCallback>, *mut c_void);
    fn sd_set_generation_progress_callback(*mut SdContext, Option<ProgressCallback>, *mut c_void);
    fn sd_cancel_generation(*mut SdContext, c_int);
    fn sd_generate_image(*mut SdContext, *const SdOpt, *mut *mut SdResult) -> SdStatus;
    fn sd_result_get_image(*const SdResult, usize) -> *const SdTensor;
    fn sd_result_free(*mut SdResult);
    fn sd_tensor_type(*const SdTensor) -> i32;
    fn sd_tensor_rank(*const SdTensor) -> usize;
    fn sd_tensor_dim(*const SdTensor, usize) -> i64;
    fn sd_tensor_byte_size(*const SdTensor) -> usize;
    fn sd_tensor_data(*const SdTensor) -> *const c_void;
}

/// An owned C handle. These stay within the runtime's library lifetime.
struct Owned<T> {
    ptr: NonNull<T>,
    free: unsafe extern "C" fn(*mut T),
}

impl<T> Owned<T> {
    fn new(ptr: *mut T, free: unsafe extern "C" fn(*mut T)) -> Option<Self> {
        NonNull::new(ptr).map(|ptr| Self { ptr, free })
    }

    fn as_ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }
}

impl<T> Drop for Owned<T> {
    fn drop(&mut self) {
        // SAFETY: This is the matching destructor for a live, uniquely owned handle.
        unsafe { (self.free)(self.as_ptr()) };
    }
}

impl Api {
    fn last_error(&self) -> String {
        // SAFETY: The library returns a thread-local C string, valid until the next call.
        let message = unsafe { (self.sd_get_last_error)() };
        if message.is_null() {
            return "no native diagnostic".into();
        }
        let message = unsafe { CStr::from_ptr(message) }.to_string_lossy();
        if message.is_empty() {
            "no native diagnostic".into()
        } else {
            message.into_owned()
        }
    }

    fn check(&self, status: SdStatus, operation: &str) -> Result<()> {
        ensure!(
            status == SD_STATUS_OK,
            "{operation} failed (status {status}): {}",
            self.last_error()
        );
        Ok(())
    }

    fn context(&self, model: &StableDiffusionModelConfig) -> Result<Owned<SdContext>> {
        // SAFETY: The option constructor transfers ownership to the caller.
        let options = Owned::new(unsafe { (self.sd_ctx_opt_create)() }, self.sd_opt_free)
            .with_context(|| format!("failed to allocate model options: {}", self.last_error()))?;
        self.set_path(&options, c"model_path", &model.path)?;
        if let Some(vae) = &model.vae {
            self.set_path(&options, c"vae_path", vae)?;
        }
        let mut context = ptr::null_mut();
        // SAFETY: The options are live and the output slot is writable. The API copies the options.
        let status = unsafe { (self.sd_ctx_create)(options.as_ptr(), &mut context) };
        self.check(status, "creating the image model context")?;
        Owned::new(context, self.sd_ctx_free)
            .context("stable-diffusion.cpp returned no model context")
    }

    fn set_path(&self, options: &Owned<SdOpt>, key: &CStr, path: &Path) -> Result<()> {
        ensure!(
            path.is_file(),
            "image model file '{}' is missing or unreadable",
            path.display()
        );
        let path = path
            .to_str()
            .context("image model path must be valid UTF-8")?;
        self.set_string(options, key, path)
    }

    fn set_string(&self, options: &Owned<SdOpt>, key: &CStr, value: &str) -> Result<()> {
        let value =
            CString::new(value).context("stable-diffusion.cpp option contains a NUL character")?;
        // SAFETY: Both strings and the option handle are valid for the call; the setter copies the value.
        let status =
            unsafe { (self.sd_opt_set_string)(options.as_ptr(), key.as_ptr(), value.as_ptr()) };
        self.check(status, &format!("setting {}", key.to_string_lossy()))
    }

    fn set_int(&self, options: &Owned<SdOpt>, key: &CStr, value: i64) -> Result<()> {
        // SAFETY: The key and option handle are live for the call.
        let status = unsafe { (self.sd_opt_set_int)(options.as_ptr(), key.as_ptr(), value) };
        self.check(status, &format!("setting {}", key.to_string_lossy()))
    }

    fn generate(&self, context: *mut SdContext, request: &Request) -> Result<Owned<SdResult>> {
        // SAFETY: The option constructor transfers ownership to the caller.
        let options = Owned::new(unsafe { (self.sd_img_gen_opt_create)() }, self.sd_opt_free)
            .with_context(|| format!("failed to allocate image options: {}", self.last_error()))?;
        self.set_string(&options, c"prompt", &request.prompt)?;
        if let Some(negative) = &request.options.negative_prompt {
            self.set_string(&options, c"negative_prompt", negative)?;
        }
        for (key, value) in [
            (c"width", request.options.width),
            (c"height", request.options.height),
            (c"sample.steps", request.options.steps),
        ] {
            if let Some(value) = value {
                self.set_int(&options, key, i64::from(value.get()))?;
            }
        }
        self.set_int(&options, c"seed", request.options.seed.unwrap_or(-1))?;

        let cancellation = Cancellation {
            context,
            cancel: self.sd_cancel_generation,
            reply: &request.reply,
        };
        let mut result = ptr::null_mut();
        // SAFETY: Handles, callback data and the output slot remain valid throughout the
        // synchronous generation. Clear the borrowed callback before returning.
        let outcome = unsafe {
            (self.sd_set_generation_progress_callback)(
                context,
                Some(cancel_if_abandoned),
                ptr::from_ref(&cancellation).cast_mut().cast(),
            );
            let status = (self.sd_generate_image)(context, options.as_ptr(), &mut result);
            let outcome = self.check(status, "generating the image");
            (self.sd_set_generation_progress_callback)(context, None, ptr::null_mut());
            outcome
        };
        outcome?;
        Owned::new(result, self.sd_result_free)
            .context("stable-diffusion.cpp returned no image result")
    }

    fn write_png(&self, result: &Owned<SdResult>, path: &Path) -> Result<()> {
        // SAFETY: The result owns the borrowed image tensor and its pixels until this method returns.
        let tensor = unsafe { (self.sd_result_get_image)(result.as_ptr(), 0) };
        ensure!(!tensor.is_null(), "stable-diffusion.cpp returned no image");
        let (kind, rank) =
            unsafe { ((self.sd_tensor_type)(tensor), (self.sd_tensor_rank)(tensor)) };
        ensure!(
            kind == SD_TENSOR_U8 && rank == 3,
            "expected a [channels, width, height] byte tensor"
        );
        let dimension = |axis| -> Result<u32> {
            let value = u32::try_from(unsafe { (self.sd_tensor_dim)(tensor, axis) })?;
            ensure!(value > 0, "image dimension cannot be zero");
            Ok(value)
        };
        let (channels, width, height) = (dimension(0)?, dimension(1)?, dimension(2)?);
        let color = match channels {
            1 => png::ColorType::Grayscale,
            2 => png::ColorType::GrayscaleAlpha,
            3 => png::ColorType::Rgb,
            4 => png::ColorType::Rgba,
            _ => bail!("unsupported image channel count {channels}"),
        };
        let size = (width as usize)
            .checked_mul(height as usize)
            .and_then(|size| size.checked_mul(channels as usize))
            .filter(|size| *size <= isize::MAX as usize)
            .context("image tensor is too large")?;
        ensure!(
            unsafe { (self.sd_tensor_byte_size)(tensor) } == size,
            "unexpected image tensor size"
        );
        let pixels = unsafe { (self.sd_tensor_data)(tensor) }.cast::<u8>();
        ensure!(
            !pixels.is_null(),
            "stable-diffusion.cpp returned no image pixels"
        );
        // SAFETY: The API supplies contiguous, interleaved bytes scoped to the result.
        let pixels = unsafe { slice::from_raw_parts(pixels, size) };
        let mut encoder = png::Encoder::new(BufWriter::new(File::create(path)?), width, height);
        encoder.set_color(color);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(pixels)?;
        writer.finish()?;
        Ok(())
    }
}

struct Cancellation<'a> {
    context: *mut SdContext,
    cancel: unsafe extern "C" fn(*mut SdContext, c_int),
    reply: &'a oneshot::Sender<Result<TempDir>>,
}

unsafe extern "C" fn ignore_console_progress(
    _step: c_int,
    _steps: c_int,
    _time: f32,
    _data: *mut c_void,
) {
}

unsafe extern "C" fn cancel_if_abandoned(_progress: *const c_void, data: *mut c_void) {
    // SAFETY: Registered data points to a live Cancellation for the duration of the call.
    let cancellation = unsafe { &*data.cast::<Cancellation<'_>>() };
    if cancellation.reply.is_closed() {
        unsafe { (cancellation.cancel)(cancellation.context, SD_CANCEL_ALL) };
    }
}

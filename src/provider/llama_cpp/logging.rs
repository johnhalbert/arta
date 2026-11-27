//! Routes llama.cpp diagnostics through a shared callback with thread-local verbosity.

use std::{cell::Cell, os::raw::c_char, ptr};
#[cfg(not(target_os = "android"))]
use std::{ffi::CStr, io::Write};

thread_local! {
    static VERBOSE: Cell<bool> = const { Cell::new(false) };
}

/// llama.cpp has a process-wide callback. Read verbosity from the model worker's scope.
pub(super) fn install() {
    // The callback ignores user data and is valid for the lifetime of the backend.
    unsafe { llama_cpp_sys_4::llama_log_set(Some(log), ptr::null_mut()) };
}

/// The native callback supplies a NUL-terminated string valid during the call.
unsafe extern "C" fn log(
    _level: llama_cpp_sys_4::ggml_log_level,
    text: *const c_char,
    _: *mut std::ffi::c_void,
) {
    if text.is_null() {
        return;
    }
    let verbose = VERBOSE.try_with(Cell::get).unwrap_or(false);
    #[cfg(target_os = "android")]
    {
        let priority = match _level {
            llama_cpp_sys_4::GGML_LOG_LEVEL_ERROR => 6,
            llama_cpp_sys_4::GGML_LOG_LEVEL_WARN => 5,
            _ => 4,
        };
        if verbose || priority == 6 {
            unsafe { __android_log_write(priority, c"llama.cpp".as_ptr(), text) };
        }
    }
    #[cfg(not(target_os = "android"))]
    if verbose {
        // llama.cpp supplies a NUL-terminated string valid for the callback.
        let text = unsafe { CStr::from_ptr(text) };
        let _ = std::io::stderr().write_all(text.to_bytes());
    }
}

#[cfg(target_os = "android")]
#[link(name = "log")]
unsafe extern "C" {
    fn __android_log_write(priority: i32, tag: *const c_char, text: *const c_char) -> i32;
}

/// RAII guard restoring the previous thread-local verbosity on return or unwind.
pub(super) struct Scope(bool);

impl Scope {
    /// Replace the calling thread's flag and retain its previous value for restoration.
    pub(super) fn new(verbose: bool) -> Self {
        Self(VERBOSE.replace(verbose))
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        VERBOSE.set(self.0);
    }
}

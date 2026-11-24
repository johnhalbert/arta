use std::{cell::Cell, ffi::CStr, io::Write, os::raw::c_char, ptr};

thread_local! {
    static VERBOSE: Cell<bool> = const { Cell::new(false) };
}

/// llama.cpp has a process-wide callback. Read verbosity from the model worker's scope.
pub(super) fn install() {
    // The callback ignores user data and is valid for the lifetime of the backend.
    unsafe { llama_cpp_sys_4::llama_log_set(Some(log), ptr::null_mut()) };
}

unsafe extern "C" fn log(
    _: llama_cpp_sys_4::ggml_log_level,
    text: *const c_char,
    _: *mut std::ffi::c_void,
) {
    if VERBOSE.try_with(Cell::get).unwrap_or(false) && !text.is_null() {
        // llama.cpp supplies a NUL-terminated string valid for the callback.
        let text = unsafe { CStr::from_ptr(text) };
        let _ = std::io::stderr().write_all(text.to_bytes());
    }
}

pub(super) struct Scope(bool);

impl Scope {
    pub(super) fn new(verbose: bool) -> Self {
        Self(VERBOSE.replace(verbose))
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        VERBOSE.set(self.0);
    }
}

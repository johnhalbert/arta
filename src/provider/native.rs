use libloading::Library;
use std::path::Path;

pub(super) fn load_library(path: &Path) -> Result<Library, libloading::Error> {
    // The configured library and its native dependencies must be trusted by the caller.
    #[cfg(windows)]
    {
        use libloading::os::windows::{
            LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
        };
        unsafe {
            libloading::os::windows::Library::load_with_flags(
                path,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
            )
        }
        .map(Into::into)
    }
    #[cfg(not(windows))]
    {
        unsafe { Library::new(path) }
    }
}

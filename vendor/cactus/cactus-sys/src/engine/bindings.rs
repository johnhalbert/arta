//! Keep the static declarations and dynamic symbol types on the same C signatures.

macro_rules! declare_api {
    ($( $(#[$doc:meta])* pub fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)?; )*) => {
        #[cfg(not(feature = "dynamic-link"))]
        unsafe extern "C" {
            $( $(#[$doc])* pub fn $name($($arg: $ty),*) $(-> $ret)?; )*
        }

        #[cfg(feature = "dynamic-link")]
        struct Api {
            _library: libloading::Library,
            $( $name: unsafe extern "C" fn($($ty),*) $(-> $ret)?, )*
        }

        #[cfg(feature = "dynamic-link")]
        static API: std::sync::OnceLock<Api> = std::sync::OnceLock::new();

        /// Load the Cactus shared library for the lifetime of this process.
        ///
        /// # Safety
        /// `path` must identify a trusted library implementing this module's exact C ABI.
        /// Loading it may execute native initializers.
        #[cfg(feature = "dynamic-link")]
        pub unsafe fn load_library(path: &std::path::Path) -> std::io::Result<()> {
            use std::io::Error;
            // SAFETY: the caller vouches for the library and its ABI. Every required symbol
            // is resolved before the table is published and the library stays resident.
            let api = unsafe {
                let library = libloading::Library::new(path).map_err(Error::other)?;
                Api {
                    $( $name: *library.get(concat!(stringify!($name), "\0").as_bytes())
                        .map_err(Error::other)?, )*
                    _library: library,
                }
            };
            API.set(api).map_err(|_| Error::other("Cactus library already loaded"))
        }

        /// Whether the dynamic API is ready for engine calls.
        #[cfg(feature = "dynamic-link")]
        pub fn is_library_loaded() -> bool {
            API.get().is_some()
        }

        $(
            $(#[$doc])*
            #[cfg(feature = "dynamic-link")]
            pub unsafe extern "C" fn $name($($arg: $ty),*) $(-> $ret)? {
                let api = API.get().expect("load the Cactus library before calling its C API");
                // SAFETY: callers must meet the same contract as for the static C declaration.
                unsafe { (api.$name)($($arg),*) }
            }
        )*
    };
}

//! Resolves and loads Cactus Compute's pinned shared runtime without Python.

use std::{
    env, fs,
    io::{self, Cursor, Read},
    path::{Path, PathBuf},
};

use crate::{
    Result,
    download::{self, Pin},
};

const VERSION: &str = "4.1.4";
const REVISION: &str = "f4ab2741203a282c6e7b89983de2b5e5b2e805a8";

// Digests from the 4.1.4 release manifest. A wheel is a ZIP container for the
// shared library and can be extracted without a Python interpreter.
fn platform() -> io::Result<(&'static str, &'static str)> {
    Ok(
        match (
            env::consts::OS,
            env::consts::ARCH,
            cfg!(target_env = "musl"),
        ) {
            ("windows", "x86_64", _) => (
                "win_amd64",
                "76f21737d93a5246afe2bfee65a5a338df35217cb6258c7aa9cc747e6ea2bb73",
            ),
            ("windows", "aarch64", _) => (
                "win_arm64",
                "f8dae7aacb3912623814db25c2d336b5b2a1b00fb1eb96d630ef53899b156e07",
            ),
            ("macos", "aarch64", _) => (
                "macosx_11_0_arm64",
                "45eaae74b37291425bc8bedfc3fa0ff6984cf8ea1997d2ea1711387974d25e43",
            ),
            ("macos", "x86_64", _) => (
                "macosx_11_0_x86_64",
                "bb04050273486582ef8484bd5c95cba3a43cb2db4510970b1f97cc6432f464cd",
            ),
            ("linux", "aarch64", false) => (
                "manylinux2014_aarch64",
                "45b703ee71fd979d3155f945a733c1dd1f18637035314e9c96fdfd60ae96b787",
            ),
            ("linux", "x86_64", false) => (
                "manylinux2014_x86_64",
                "314cd027e95cae1cb8537c478ca6cddaec3fd02165f80d960f1934399871fc33",
            ),
            ("linux", "aarch64", true) => (
                "musllinux_1_2_aarch64",
                "f54163ee24baf7a5fef6e864bd440d3e6e87363932ed30ac56f676507e57d5bf",
            ),
            ("linux", "x86_64", true) => (
                "musllinux_1_2_x86_64",
                "78fd9d4ec5d1c091fd354afc6a0c4714b5fa1a3a2dda95019adc421374495e40",
            ),
            _ => {
                return Err(io::Error::other(
                    "no pinned Cactus runtime for this target; supply a compatible shared library",
                ));
            }
        },
    )
}

/// Resolve an explicit library, `CACTUS_LIB_PATH`, or a checksum-verified release artifact.
/// The downloader honors `HF_HUB_OFFLINE` and the Hugging Face cache configuration.
pub fn library_path(configured: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = configured {
        return Ok(std::path::absolute(path)?);
    }
    if let Some(path) = env::var_os("CACTUS_LIB_PATH") {
        return Ok(std::path::absolute(path)?);
    }

    let (tag, sha256) = platform()?;
    let wheel = &format!("cactus_runtime-{VERSION}-py3-none-{tag}.whl");
    let wheel_bytes = download::fetch_at(
        &Pin {
            repo: "Cactus-Compute/cactus",
            revision: REVISION,
            file: wheel,
            sha256,
            env: "",
            cache_dir: "engine",
        },
        &format!("releases/{VERSION}/python/{wheel}"),
    )?;

    // The precise archive member is selected from a trusted, pinned wheel. No archive paths
    // are written to disk directly.
    let name = format!(
        "{}cactus{}",
        env::consts::DLL_PREFIX,
        env::consts::DLL_SUFFIX
    );
    // Windows wheels use the `lib` prefix, too.
    let name = if cfg!(windows) {
        "libcactus.dll"
    } else {
        name.as_str()
    };
    let mut archive = zip::ZipArchive::new(Cursor::new(wheel_bytes)).map_err(io::Error::other)?;
    let mut library = Vec::new();
    archive
        .by_name(&format!("cactus/{name}"))
        .map_err(io::Error::other)?
        .read_to_end(&mut library)?;
    let destination = download::cache_base()
        .join("cactus-rs")
        .join("engine")
        .join(REVISION)
        .join(tag)
        .join(name);
    let already_cached = || fs::read(&destination).is_ok_and(|cached| cached == library);
    if !already_cached()
        && let Err(error) = download::install(&destination, &library)
    {
        // Another worker may have installed and opened the same DLL before our rename.
        if !already_cached() {
            return Err(error);
        }
    }
    Ok(std::path::absolute(destination)?)
}

/// Load the shared library for all models in this process.
///
/// # Safety
/// The library must be trusted and implement the Cactus engine 4 ABI.
pub unsafe fn load_library(path: &Path) -> Result<()> {
    // SAFETY: the caller vouches for the library and the ABI.
    Ok(unsafe { cactus_sys::load_library(path) }?)
}

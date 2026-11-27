use std::{
    env,
    fs::{self, File},
    io::{self, Read},
    path::{Path, PathBuf},
};

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("Cargo must set CARGO_CFG_TARGET_OS");
    if target_os == "android" {
        return;
    }
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must set OUT_DIR"));
    if target_os == "windows" && env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        // audio.cpp's narrow C API paths are UTF-8.
        let manifest = Path::new("native/windows-utf8.manifest");
        let path = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join(manifest);
        println!("cargo:rerun-if-changed={}", manifest.display());
        println!("cargo:rustc-link-arg-bin=arta=/MANIFEST:EMBED");
        println!(
            "cargo:rustc-link-arg-bin=arta=/MANIFESTINPUT:{}",
            path.display()
        );
    }
    let profile = profile_directory(&out_dir);
    build_library(
        &out_dir,
        &profile,
        &target_os,
        "stable-diffusion",
        "stable-diffusion",
        ["SD_CUDA", "SD_HIPBLAS", "SD_VULKAN"],
    );
    build_library(
        &out_dir.join("audio"),
        &profile,
        &target_os,
        "audio",
        "audiocpp",
        [
            "ENGINE_ENABLE_CUDA",
            "ENGINE_ENABLE_HIP",
            "ENGINE_ENABLE_VULKAN",
        ],
    );
}

fn build_library(
    out_dir: &Path,
    profile: &Path,
    target_os: &str,
    project: &str,
    name: &str,
    features: [&str; 3],
) {
    let source = format!("native/{project}");
    println!("cargo:rerun-if-changed={source}/CMakeLists.txt");
    let (install_dir, filename) = match target_os {
        "windows" => ("bin", format!("{name}.dll")),
        "macos" => ("lib", format!("lib{name}.dylib")),
        _ => ("lib", format!("lib{name}.so")),
    };

    let mut cmake = cmake::Config::new(source);
    cmake
        .out_dir(out_dir)
        .profile("Release")
        .define("CMAKE_INSTALL_BINDIR", "bin")
        .define("CMAKE_INSTALL_LIBDIR", "lib");
    for (option, enabled) in features.into_iter().zip([
        cfg!(feature = "cuda"),
        cfg!(feature = "hip"),
        cfg!(feature = "vulkan"),
    ]) {
        cmake.define(option, if enabled { "ON" } else { "OFF" });
    }
    let library = cmake.build().join(install_dir).join(&filename);
    for directory in [
        profile.to_owned(),
        profile.join("deps"),
        profile.join("examples"),
    ] {
        let destination = directory.join(&filename);
        println!("cargo:rerun-if-changed={}", destination.display());
        stage_library(&library, &destination).unwrap_or_else(|error| {
            panic!(
                "failed to stage '{}' at '{}': {error}",
                library.display(),
                destination.display()
            )
        });
    }
}

fn profile_directory(out_dir: &Path) -> PathBuf {
    // Cargo's OUT_DIR has the shape <target>/<profile>/build/<package>/out.
    let profile = out_dir
        .ancestors()
        .nth(3)
        .expect("Cargo must provide a profile directory above OUT_DIR");
    // Cargo also prepends its actual output path for shared libraries. This covers builds
    // that keep their intermediates and final artifacts in different directories.
    let variable = if cfg!(windows) {
        "PATH"
    } else if cfg!(target_os = "macos") {
        "DYLD_FALLBACK_LIBRARY_PATH"
    } else {
        "LD_LIBRARY_PATH"
    };
    if let Some(paths) = env::var_os(variable) {
        for path in env::split_paths(&paths) {
            let candidate = if path.ends_with("deps") {
                path.parent().unwrap_or(&path)
            } else {
                &path
            };
            if candidate.is_absolute() && candidate.file_name() == profile.file_name() {
                return candidate.to_owned();
            }
        }
    }
    profile.to_owned()
}

fn stage_library(source: &Path, destination: &Path) -> io::Result<()> {
    if contents_match(source, destination)? {
        return Ok(());
    }
    let parent = destination
        .parent()
        .expect("the staged library has a target directory");
    fs::create_dir_all(parent)?;
    let staged = tempfile::NamedTempFile::new_in(parent)?;
    fs::copy(source, staged.path())?;
    staged.persist(destination)?;
    Ok(())
}

fn contents_match(source: &Path, destination: &Path) -> io::Result<bool> {
    let mut source = File::open(source)?;
    let mut destination = match File::open(destination) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error),
    };
    if source.metadata()?.len() != destination.metadata()?.len() {
        return Ok(false);
    }
    let mut expected = [0; 64 * 1024];
    let mut actual = [0; 64 * 1024];
    loop {
        let count = source.read(&mut expected)?;
        if count == 0 {
            return Ok(true);
        }
        destination.read_exact(&mut actual[..count])?;
        if expected[..count] != actual[..count] {
            return Ok(false);
        }
    }
}

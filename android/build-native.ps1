[CmdletBinding()]
param(
    [string]$SdkRoot = $env:ANDROID_HOME,
    [string]$CMakePath,
    [string]$NinjaPath
)

$ErrorActionPreference = 'Stop'

if (!$SdkRoot) { $SdkRoot = $env:ANDROID_SDK_ROOT }
if (!$SdkRoot) { $SdkRoot = Join-Path $env:LOCALAPPDATA 'Android/Sdk' }
$ndkVersion = '28.2.13676358'
$ndk = Join-Path $SdkRoot "ndk/$ndkVersion"
$toolchain = Join-Path $ndk 'toolchains/llvm/prebuilt/windows-x86_64'
$ndkBin = Join-Path $toolchain 'bin'
if (!(Test-Path -LiteralPath (Join-Path $ndkBin 'clang.exe') -PathType Leaf)) {
    throw "Install Android NDK $ndkVersion under '$SdkRoot' or pass -SdkRoot."
}
$toolchain = (Get-Item -LiteralPath $toolchain).FullName
$ndkBin = Join-Path $toolchain 'bin'

$target = 'aarch64-linux-android'
if ($target -notin (rustup target list --installed)) {
    throw "Install the Rust target with 'rustup target add $target'."
}

function Find-BuildTool([string]$Name, [string]$Override, [string]$VisualStudioPath) {
    if ($Override) { return (Get-Item -LiteralPath $Override).FullName }
    $onPath = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($onPath) { return $onPath.Source }

    $sdkTools = Get-ChildItem -LiteralPath (Join-Path $SdkRoot 'cmake') -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending
    foreach ($tool in $sdkTools) {
        $candidate = Join-Path $tool.FullName "bin/$Name"
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    if (Test-Path -LiteralPath $vswhere -PathType Leaf) {
        $found = & $vswhere -all -products '*' -find $VisualStudioPath | Select-Object -First 1
        if ($found) { return $found }
    }
    throw "Couldn’t find $Name. Add it to PATH or pass its path explicitly."
}

$vsTools = 'Common7\IDE\CommonExtensions\Microsoft\CMake'
$cmake = Find-BuildTool 'cmake.exe' $CMakePath "$vsTools\CMake\bin\cmake.exe"
$ninja = Find-BuildTool 'ninja.exe' $NinjaPath "$vsTools\Ninja\ninja.exe"
$workspace = Split-Path -Parent $PSScriptRoot
$cargoOutput = Join-Path $PSScriptRoot 'app/build/cargo'
$staging = Join-Path $PSScriptRoot 'app/build/rustJniLibs/arm64-v8a'

$clang = Join-Path $ndkBin 'clang.exe'
$sysroot = Join-Path $toolchain 'sysroot'
$resourceDir = & $clang -print-resource-dir
if ($LASTEXITCODE -ne 0) { throw 'Could not query the Android Clang resource directory.' }

$buildEnvironment = @{
    Path = "$(Split-Path -Parent $ninja);$env:Path"
    ANDROID_NDK = (Get-Item -LiteralPath $ndk).FullName
    ARTA_ANDROID_CMAKE = $cmake
    CMAKE_aarch64_linux_android = Join-Path $PSScriptRoot 'cmake-arm64.cmd'
    CMAKE_GENERATOR_aarch64_linux_android = 'Ninja'
    LLAMA_BUILD_SHARED_LIBS = '0'
    LLAMA_OPENSSL = 'OFF'
    LLAMA_CURL = 'OFF'
    CC_aarch64_linux_android = $clang
    CXX_aarch64_linux_android = Join-Path $ndkBin 'clang++.exe'
    AR_aarch64_linux_android = Join-Path $ndkBin 'llvm-ar.exe'
    CFLAGS_aarch64_linux_android = '--target=aarch64-linux-android28'
    CXXFLAGS_aarch64_linux_android = '--target=aarch64-linux-android28'
    CXXSTDLIB_aarch64_linux_android = 'c++_shared'
    CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER = $clang
    CARGO_TARGET_AARCH64_LINUX_ANDROID_RUSTFLAGS = '-C link-arg=--target=aarch64-linux-android28 -C link-arg=-Wl,-z,max-page-size=16384 -C link-arg=-Wl,-z,defs'
    BINDGEN_EXTRA_CLANG_ARGS_aarch64_linux_android = "--target=aarch64-linux-android28 --sysroot=`"$sysroot`" -resource-dir `"$resourceDir`""
}

if (!$env:LIBCLANG_PATH) {
    $llvm = Join-Path $env:ProgramFiles 'LLVM/bin'
    if (Test-Path -LiteralPath (Join-Path $llvm 'libclang.dll') -PathType Leaf) { $buildEnvironment.LIBCLANG_PATH = $llvm }
}

$previousEnvironment = @{}
try {
    foreach ($variable in $buildEnvironment.Keys) {
        $previousEnvironment[$variable] = [Environment]::GetEnvironmentVariable($variable, 'Process')
        [Environment]::SetEnvironmentVariable($variable, $buildEnvironment[$variable], 'Process')
    }
    cargo build --locked --release -p arta-android --target $target --target-dir $cargoOutput --manifest-path (Join-Path $workspace 'Cargo.toml')
    if ($LASTEXITCODE -ne 0) { throw 'Could not build the Arta Android library.' }
} finally {
    foreach ($variable in $previousEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($variable, $previousEnvironment[$variable], 'Process')
    }
}

New-Item -ItemType Directory -Path $staging -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $cargoOutput "$target/release/libarta_android.so") -Destination $staging
Copy-Item -LiteralPath (Join-Path $sysroot 'usr/lib/aarch64-linux-android/libc++_shared.so') -Destination $staging
Write-Output "Staged ARM64 libraries in $staging"

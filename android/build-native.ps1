[CmdletBinding()]
param(
    [string]$SdkRoot = $env:ANDROID_HOME,
    [string]$CMakePath,
    [string]$NinjaPath,
    [switch]$CpuOnly
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

function Get-VerifiedArchive([string]$Directory, [string]$Name, [string]$Revision, [string]$Sha256) {
    $archive = Join-Path $Directory "$Name.tar.gz"
    if (!(Test-Path -LiteralPath $archive -PathType Leaf) -or
        (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $Sha256) {
        Write-Host "Downloading pinned $Name"
        Invoke-WebRequest -UseBasicParsing -Uri "https://codeload.github.com/KhronosGroup/$Name/tar.gz/$Revision" -OutFile $archive
        if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $Sha256) {
            throw "The $Name archive failed SHA-256 verification."
        }
    }
    return $archive
}

function Set-ShaderHostEnvironment([hashtable]$BuildEnvironment, [string]$LlvmBin) {
    if ($env:VSCMD_ARG_TGT_ARCH -ne 'x64' -or !$env:INCLUDE -or !$env:LIB -or
        !(Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
        if (!(Test-Path -LiteralPath $vswhere -PathType Leaf)) {
            throw 'Install Visual Studio C++ Build Tools to build the Vulkan shader generator.'
        }
        $vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if (!$vs) { throw 'Install the x64 C++ tools in Visual Studio Build Tools.' }
        $vcvars = Join-Path $vs 'VC/Auxiliary/Build/vcvars64.bat'
        $variables = & cmd.exe /d /c "call `"$vcvars`" >nul && set"
        if ($LASTEXITCODE -ne 0) { throw 'Could not initialize the Windows C++ build environment.' }
        foreach ($entry in $variables) {
            if ($entry -match '^(Path|INCLUDE|LIB|LIBPATH)=(.*)$') {
                if ($Matches[1] -ieq 'Path') {
                    $BuildEnvironment.Path = "$($BuildEnvironment.Path);$($Matches[2])"
                } else {
                    $BuildEnvironment[$Matches[1]] = $Matches[2]
                }
            }
        }
    }

    # CMake sees NDK clang on PATH during the cross build. Select Windows LLVM explicitly.
    $hostClang = Join-Path $LlvmBin 'clang.exe'
    $hostClangCpp = Join-Path $LlvmBin 'clang++.exe'
    if (!(Test-Path -LiteralPath $hostClang) -or !(Test-Path -LiteralPath $hostClangCpp)) {
        throw 'Install Windows LLVM, including clang and clang++, to build the Vulkan shader generator.'
    }
    $BuildEnvironment.ARTA_ANDROID_HOST_CC = $hostClang
    $BuildEnvironment.ARTA_ANDROID_HOST_CXX = $hostClangCpp
}

$vsTools = 'Common7\IDE\CommonExtensions\Microsoft\CMake'
$cmake = Find-BuildTool 'cmake.exe' $CMakePath "$vsTools\CMake\bin\cmake.exe"
$ninja = Find-BuildTool 'ninja.exe' $NinjaPath "$vsTools\Ninja\ninja.exe"
$workspace = Split-Path -Parent $PSScriptRoot
$cargoOutput = Join-Path $PSScriptRoot 'app/build/cargo'
$staging = Join-Path $PSScriptRoot 'app/build/rustJniLibs/arm64-v8a'

$clang = Join-Path $ndkBin 'clang.exe'
$sysroot = Join-Path $toolchain 'sysroot'
$llvm = Join-Path $env:ProgramFiles 'LLVM/bin'
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
    if (Test-Path -LiteralPath (Join-Path $llvm 'libclang.dll') -PathType Leaf) { $buildEnvironment.LIBCLANG_PATH = $llvm }
}

$features = @()
if (!$CpuOnly) {
    # The nested shader-generator build can exceed MAX_PATH inside a checkout.
    $hasher = [Security.Cryptography.SHA256]::Create()
    try {
        $workspaceHash = [BitConverter]::ToString($hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($workspace))).Replace('-', '')
    } finally {
        $hasher.Dispose()
    }
    $cargoOutput = Join-Path $env:LOCALAPPDATA "arta/vk/$($workspaceHash.Substring(0, 12))"
    $headers = Join-Path $PSScriptRoot 'app/build/vulkan-headers-1.4.350.0'
    $prefix = Join-Path $headers 'install'
    $spirvConfig = Join-Path $prefix 'share/cmake/SPIRV-Headers'
    if (!(Test-Path (Join-Path $spirvConfig 'SPIRV-HeadersConfig.cmake')) -or
        !(Test-Path (Join-Path $prefix 'include/vulkan/vulkan.hpp'))) {
        New-Item -ItemType Directory -Path $prefix -Force | Out-Null
        $vulkanRev = '8864cdc896bbc2a9b6eb36b3218fc9ef57908d77'
        $spirvRev = 'ad9184e76a66b1001c29db9b0a3e87f646c64de0'
        $vulkanArchive = Get-VerifiedArchive $headers 'Vulkan-Headers' $vulkanRev '35824117F649BEFC664A12622BC801EDDBBA036AC03D906DB400EE1C1E7903E9'
        $spirvArchive = Get-VerifiedArchive $headers 'SPIRV-Headers' $spirvRev 'B5B7EBA62453EB8C6F6A5FBF7155B71CDE693BAFE9CD5F03B79ED8C714816AFE'
        tar -xf $vulkanArchive -C $prefix --strip-components 1 "Vulkan-Headers-$vulkanRev/include"
        if ($LASTEXITCODE -ne 0) { throw 'Could not extract Vulkan headers.' }
        tar -xf $spirvArchive -C $headers
        if ($LASTEXITCODE -ne 0) { throw 'Could not extract SPIR-V headers.' }
        $spirvBuild = Join-Path $headers 'spirv-build'
        & $cmake -S (Join-Path $headers "SPIRV-Headers-$spirvRev") -B $spirvBuild "-DCMAKE_INSTALL_PREFIX=$prefix" '-DSPIRV_HEADERS_ENABLE_TESTS=OFF'
        if ($LASTEXITCODE -ne 0) { throw 'Could not configure SPIR-V headers.' }
        & $cmake --install $spirvBuild --config Release
        if ($LASTEXITCODE -ne 0) { throw 'Could not install SPIR-V headers into the build cache.' }
    }
    $buildEnvironment.ARTA_ANDROID_VULKAN_INCLUDE = Join-Path $prefix 'include'
    $buildEnvironment.ARTA_ANDROID_VULKAN_LIBRARY = Join-Path $sysroot 'usr/lib/aarch64-linux-android/28/libvulkan.so'
    $buildEnvironment.ARTA_ANDROID_GLSLC = Join-Path $ndk 'shader-tools/windows-x86_64/glslc.exe'
    $buildEnvironment.ARTA_ANDROID_SPIRV_CONFIG = $spirvConfig
    $buildEnvironment.ARTA_ANDROID_LLAMA_HOOK = Join-Path $PSScriptRoot 'llama-vulkan-guard.cmake'
    $hookHash = (Get-FileHash -LiteralPath $buildEnvironment.ARTA_ANDROID_LLAMA_HOOK -Algorithm SHA256).Hash
    $guardHash = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'native/llama-vulkan-guard.cpp') -Algorithm SHA256).Hash
    # llama-cpp-sys tracks LLAMA_* options so Cargo reruns its native build when these sources change.
    $buildEnvironment.LLAMA_ARTA_VULKAN_REVISION = "$hookHash-$guardHash"
    $buildEnvironment.CARGO_TARGET_AARCH64_LINUX_ANDROID_RUSTFLAGS += ' -C link-arg=-Wl,--wrap=llama_decode -C link-arg=-Wl,--wrap=ggml_backend_sched_synchronize'
    $hostLlvm = if ($env:LIBCLANG_PATH -and (Test-Path -LiteralPath $env:LIBCLANG_PATH -PathType Container)) {
        $env:LIBCLANG_PATH
    } else { $llvm }
    Set-ShaderHostEnvironment $buildEnvironment $hostLlvm
    $features = @('--features', 'vulkan')
}

$previousEnvironment = @{}
try {
    foreach ($variable in $buildEnvironment.Keys) {
        $previousEnvironment[$variable] = [Environment]::GetEnvironmentVariable($variable, 'Process')
        [Environment]::SetEnvironmentVariable($variable, $buildEnvironment[$variable], 'Process')
    }
    cargo build --locked --release -p arta-android --target $target --target-dir $cargoOutput --manifest-path (Join-Path $workspace 'Cargo.toml') @features
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

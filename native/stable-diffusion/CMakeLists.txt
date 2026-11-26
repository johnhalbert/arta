cmake_minimum_required(VERSION 3.18...3.21)
project(arta_stable_diffusion LANGUAGES C CXX)

include(FetchContent)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# The GGML and CUTLASS revisions match the submodules in this stable-diffusion.cpp release.
set(SD_REVISION 52491bd9f1fc15281791723bdff13baab7fd6819)
set(GGML_REVISION eeda9ec48fc84a3df0bb9dcc19ae9053075a2539)
FetchContent_Declare(arta_sd
    URL "https://codeload.github.com/leejet/stable-diffusion.cpp/tar.gz/${SD_REVISION}"
    URL_HASH SHA256=0ac33ca3343c2b2a38cbe749d7422797d589982b373f62884f02ab3143521ad6)
FetchContent_Declare(arta_sd_ggml
    URL "https://codeload.github.com/leejet/ggml/tar.gz/${GGML_REVISION}"
    URL_HASH SHA256=8bd7adc7eb1366d31670467951d21eb1a5f55e8f6d89b9b91909ffb0246ac5d2
    SOURCE_SUBDIR arta-download-only)
# Populate GGML before stable-diffusion.cpp configures its own copy with its backend settings.
FetchContent_MakeAvailable(arta_sd_ggml)
set(SD_GGML_SOURCE_DIR "${arta_sd_ggml_SOURCE_DIR}" CACHE PATH "" FORCE)

if(SD_CUDA)
    FetchContent_Declare(arta_sd_cutlass
        URL "https://codeload.github.com/NVIDIA/cutlass/tar.gz/ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e"
        URL_HASH SHA256=36429479df2ab1cd1c7981c2d34e786106326ccfefc8846f902f76151a65e112
        SOURCE_SUBDIR arta-download-only)
    FetchContent_MakeAvailable(arta_sd_cutlass)
    set(SD_CUDA_CUTLASS_PATH "${arta_sd_cutlass_SOURCE_DIR}" CACHE PATH "" FORCE)
endif()

# The archives lack .git metadata. Supply the pins instead of letting upstream identify
# the enclosing Arta checkout as the revision of either native dependency.
set(GIT_EXE "")
set(SDCPP_BUILD_VERSION master-1133-52491bd)
string(SUBSTRING "${SD_REVISION}" 0 7 SDCPP_BUILD_COMMIT)
string(SUBSTRING "${GGML_REVISION}" 0 7 GGML_BUILD_COMMIT)

set(SD_BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(SD_BUILD_SHARED_GGML_LIB OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SD_WEBP OFF CACHE BOOL "" FORCE)
set(SD_WEBM OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(arta_sd)

if(WIN32)
    set_target_properties(stable-diffusion PROPERTIES PREFIX "")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Prevent the process's dynamically linked llama.cpp GGML from interposing on SD's copy.
    target_link_options(stable-diffusion PRIVATE "LINKER:--exclude-libs,ALL")
endif()

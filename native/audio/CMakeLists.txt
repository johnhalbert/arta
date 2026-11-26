cmake_minimum_required(VERSION 3.20)
project(arta_audio LANGUAGES C CXX)

include(FetchContent)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# The archive includes audio.cpp's patched GGML and sentencepiece source trees.
set(AUDIO_REVISION dd1ad8bd88497eeb50999d1237c1e19e9d5a2133)
FetchContent_Declare(arta_audio_source
    URL "https://codeload.github.com/0xShug0/audio.cpp/tar.gz/${AUDIO_REVISION}"
    URL_HASH SHA256=0f6ef57dd2562ce86b643d3bc0fee578e36ebaa5508aee4f15e4c66cf35b273d
    SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/src"
    SOURCE_SUBDIR arta-download-only)
FetchContent_MakeAvailable(arta_audio_source)

set(AUDIOCPP_VERSION "${AUDIO_REVISION}" CACHE STRING "" FORCE)
set(AUDIOCPP_BUILD_C_API ON CACHE BOOL "" FORCE)
set(AUDIOCPP_DEPLOYMENT_BUILD ON CACHE BOOL "" FORCE)
set(ENGINE_ENABLE_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_COMMIT "${AUDIO_REVISION}")
set(GIT_EXE "")
set(CMAKE_DISABLE_FIND_PACKAGE_Git TRUE)

# cmake-rs supplies its own compiler flags, so enable C++ exception unwinding explicitly.
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
endif()

# VS 2019 needs its conforming lambda parser for the upstream C++17 sources.
if(MSVC AND MSVC_VERSION LESS 1930)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:lambda>)
endif()

# Build the shared C API and its dependencies without the CLI and server targets.
add_subdirectory("${arta_audio_source_SOURCE_DIR}" audio EXCLUDE_FROM_ALL)
if(MSVC AND MSVC_VERSION LESS 1930)
    # The VS 2019 optimizer crashes on these two translation units.
    set_property(SOURCE
        "${arta_audio_source_SOURCE_DIR}/src/community_models/confucius4_r2t2/audio_encoder.cpp"
        "${arta_audio_source_SOURCE_DIR}/src/framework/modules/speech_encoders/qwen3_audio_encoder_runtime.cpp"
        TARGET_DIRECTORY audiocpp APPEND PROPERTY COMPILE_OPTIONS /Od)
endif()
add_custom_target(arta_audiocpp ALL DEPENDS audiocpp)
# Stage the same unversioned filename that the Rust loader looks up.
set_property(TARGET audiocpp PROPERTY VERSION)
set_property(TARGET audiocpp PROPERTY SOVERSION)
if(WIN32)
    set_target_properties(audiocpp PROPERTIES PREFIX "")
endif()
install(TARGETS audiocpp
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")

@echo off
rem Override the pinned llama-cpp-sys Android configure flags for the ARM64 baseline.
if "%~1"=="--build" goto passthrough
if "%~1"=="--version" goto passthrough
if defined ARTA_ANDROID_VULKAN_INCLUDE goto vulkan
"%ARTA_ANDROID_CMAKE%" %* "-DCMAKE_C_FLAGS=-march=armv8-a" "-DCMAKE_CXX_FLAGS=-march=armv8-a" "-DANDROID_STL=c++_shared"
exit /b %errorlevel%

:vulkan
"%ARTA_ANDROID_CMAKE%" %* "-DCMAKE_C_FLAGS=-march=armv8-a" "-DCMAKE_CXX_FLAGS=-march=armv8-a" "-DANDROID_STL=c++_shared" "-DVulkan_INCLUDE_DIR=%ARTA_ANDROID_VULKAN_INCLUDE%" "-DVulkan_LIBRARY=%ARTA_ANDROID_VULKAN_LIBRARY%" "-DVulkan_GLSLC_EXECUTABLE=%ARTA_ANDROID_GLSLC%" "-DSPIRV-Headers_DIR=%ARTA_ANDROID_SPIRV_CONFIG%" "-DHOST_C_COMPILER=%ARTA_ANDROID_HOST_CC%" "-DHOST_CXX_COMPILER=%ARTA_ANDROID_HOST_CXX%" "-DCMAKE_PROJECT_llama.cpp_INCLUDE=%ARTA_ANDROID_LLAMA_HOOK%"
exit /b %errorlevel%

:passthrough
"%ARTA_ANDROID_CMAKE%" %*
exit /b %errorlevel%

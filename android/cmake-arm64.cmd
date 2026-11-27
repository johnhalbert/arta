@echo off
rem Override the pinned llama-cpp-sys Android configure flags for the ARM64 baseline.
if "%~1"=="--build" goto passthrough
if "%~1"=="--version" goto passthrough
"%ARTA_ANDROID_CMAKE%" %* "-DCMAKE_C_FLAGS=-march=armv8-a" "-DCMAKE_CXX_FLAGS=-march=armv8-a" "-DANDROID_STL=c++_shared"
exit /b %errorlevel%

:passthrough
"%ARTA_ANDROID_CMAKE%" %*
exit /b %errorlevel%

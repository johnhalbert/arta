# Arta for Android

The first build runs text chat on device using Arta’s Rust harness and llama.cpp with ARM64 CPU inference. Import a single-file instruction-tuned GGUF with an embedded chat template. The app copies it into private storage and reloads it on launch. Conversation history lives in memory. Stop cancels the current generation and excludes that exchange from future model context.

## Build on Windows

The app uses Android SDK platform 35, Build Tools 36.0.0, the side-by-side NDK `28.2.13676358`, Gradle 9.5.0, and a compatible JDK. It also needs Rust, CMake, Ninja, and LLVM `libclang`. The native script checks `PATH`, the SDK, and Visual Studio for CMake and Ninja. Set `LIBCLANG_PATH` if LLVM is installed in a custom location.

Install the missing SDK components through Android Studio or `sdkmanager`, then run these commands from the repository root. Adjust `ANDROID_HOME` if you use a different SDK location.

```powershell
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
rustup target add aarch64-linux-android
.\android\build-native.ps1
gradle -p android :app:assembleDebug
```

The APK is `android/app/build/outputs/apk/debug/app-debug.apk`. Install it on an ARM64 Android 9 or newer device with USB debugging enabled.

```powershell
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
```

To verify the prototype, import a small quantized GGUF, send a short prompt, and check that the reply streams. Send a follow-up that refers to the first turn. Then stop another response and verify you can start a fresh turn. Use **New chat** to clear conversation context.

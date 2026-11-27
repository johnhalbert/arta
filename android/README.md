# Arta for Android

Arta runs text chat on device using its Rust harness and llama.cpp on ARM64. Import a single-file instruction-tuned GGUF with an embedded chat template. The app copies it into private storage and reloads it on launch. Conversation history lives in memory. Stop cancels the current generation and excludes that exchange from future model context.

Open **GPU offload** to see the Vulkan GPUs detected by llama.cpp. After loading a model, use the dialog to select a layer count or offload all layers. Enter **0** to switch back to the CPU. Changing the setting reloads the current model and keeps the conversation. If the requested placement fails to load, Arta attempts to restore the previous placement. Each app process starts in CPU mode. Offload requires Vulkan 1.2 or newer and a driver with the extensions the bundled llama.cpp backend needs. The Android build supports CPU and Vulkan GPU inference.

Inference runs in a separate process. If it exits during chat or a placement change, Arta keeps the app open and attempts to reload the current model on CPU with all completed turns. The chat shows the Android exit reason and signal when available. Open **GPU offload** then **Last native crash** to copy the most recent native inference crash that Android still has. On Android 12 or newer the report includes the abort message and crashing thread stack when available.

On the proprietary Adreno 830 driver, Arta places Q4_K matrix operations on the CPU to avoid a Vulkan pipeline failure. Other supported operations can still run on the GPU. [llama.cpp issue #28635](https://github.com/ggml-org/llama.cpp/issues/28635)

## Build on Windows

The app uses Android SDK platform 35, Build Tools 36.0.0, the side-by-side NDK `28.2.13676358`, Gradle 9.5.0, and a compatible JDK. It also needs Rust, CMake, Ninja, and LLVM `libclang`. The native script checks `PATH`, the SDK, and Visual Studio for CMake and Ninja. Set `LIBCLANG_PATH` if LLVM is installed in a custom location.

The default native build enables Vulkan. It uses the NDK shader compiler and Windows LLVM with the Visual Studio x64 C++ tools for the host shader generator. The script downloads and verifies pinned Vulkan and SPIR-V headers. It keeps the Vulkan native build cache under `%LOCALAPPDATA%\arta\vk` to stay within Windows path limits. Pass `-CpuOnly` to `build-native.ps1` to build without Vulkan.

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

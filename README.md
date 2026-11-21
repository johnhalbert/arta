# arta

Run a single prompt against a local GGUF model with llama.cpp.

Build with Rust, CMake, a C++ compiler, and LLVM's `libclang`. On Windows, use the Visual Studio C++ Build Tools.

Place a local GGUF model under `gguf/` or pass its path with `--model`. Model weights are stored locally.

```powershell
cargo build --locked --release
.\target\release\arta.exe --model .\gguf\Qwen3.5-4B-Q4_0.gguf --prompt "Reply with one sentence about the Moon."
```

Keep the native libraries Cargo places alongside the executable if you move it. You can also invoke the CLI with `cargo run --release --` followed by the same arguments.

Use `--prompt -` to read UTF-8 text from stdin. The response goes to stdout. Use `--verbose` to see full llama.cpp diagnostics on stderr. Models with a chat template receive a single user message, with thinking disabled where the template supports it. Models without a template receive the prompt as plain text.

`--context-length` controls the combined prompt and response budget and defaults to 4096 tokens or the model's trained limit, whichever is lower. `--max-tokens` caps generation and defaults to 256.

For GPU offload, build with the NVIDIA CUDA Toolkit installed and pass the number of layers to offload.

```powershell
cargo run --locked --release --features cuda --target-dir target/cuda -- --model .\gguf\Qwen3.5-4B-Q4_0.gguf --prompt "Hello" --gpu-layers 20
```

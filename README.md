# arta

`arta` uses llama.cpp to process one prompt with a model in a local GGUF file.

## Build and run on Windows

To build `arta`, you need Rust, CMake, a C++ compiler, and LLVM's `libclang`. On Windows, use the Visual Studio C++ Build Tools.

1. Build the release executable.

   ```powershell
   cargo build --locked --release
   ```

2. Specify the local GGUF file with `--model`. The example uses a file in `gguf/`.

3. Run the executable with your prompt.

   ```powershell
   .\target\release\arta.exe --model .\gguf\Qwen3.5-4B-Q4_0.gguf --prompt "Reply with one sentence about the Moon."
   ```

If you move the executable, keep the native libraries from Cargo in the same directory as the executable.

To run `arta` through Cargo, add the same `arta` arguments after `cargo run --release --`.

## Configure providers

Copy `config.example.toml` to `config.toml`. Use `[app]` to select the provider and model. Set model paths relative to the config file.

To select a different config file, use `--config PATH`. To change the selection for one prompt, use `--provider ID` or `--model-id ID`.

For llama.cpp, the existing flags override the config values for one run. A relative path with `--model` starts from the working directory. Use `--verbose=false` to disable detailed diagnostics for one run.

```powershell
.\target\release\arta.exe --config config.toml --prompt "Hello" --max-tokens 32
```

## Prompt input and output

Use `--prompt -` to read UTF-8 text from standard input (`stdin`). `arta` writes the response to standard output (`stdout`). Use `--verbose` to see full llama.cpp diagnostics on standard error (`stderr`).

If the model has a chat template, `arta` passes the prompt as one user message. If the template supports a thinking setting, `arta` disables thinking. If the model has no chat template, `arta` passes the prompt as plain text.

## Token limits

- `--context-length` sets the total number of tokens available for the prompt and response. The default is 4096 tokens or the model's trained context length, whichever is lower.
- `--max-tokens` sets the maximum number of tokens `arta` can generate. The default is 256 tokens.

## Use a GPU

1. Install the NVIDIA CUDA Toolkit.
2. Use the `cuda` feature when you build `arta`.
3. Use `--gpu-layers` to specify how many model layers the GPU processes.

```powershell
cargo run --locked --release --features cuda --target-dir target/cuda -- --model .\gguf\Qwen3.5-4B-Q4_0.gguf --prompt "Hello" --gpu-layers 20
```

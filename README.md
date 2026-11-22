# arta

`arta` uses llama.cpp to run one-shot prompts and interactive conversations with a model in a local GGUF file.

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

## Interactive conversations

Set up `config.toml` in your working directory, then run `arta` with no arguments in your terminal.

```powershell
.\target\release\arta.exe
```

The full-screen view has two empty panes across the top and a conversation in the bottom half. Press Enter to send a prompt. A spinner appears below Arta's name until the reply starts streaming. Use Page Up and Page Down to read earlier turns, or Ctrl+End to return to the latest reply. Enter `/reset` to clear the conversation, or use `/exit` or Ctrl+C to quit.

With redirected input or output, `arta` reads one prompt per line and writes replies to standard output. `/reset`, `/exit`, and end-of-input work in this mode.

The current conversation counts toward the context limit. Use `/reset` to start a fresh conversation if the history fills the context.

## Use tools during inference

Configure MCP servers under `[mcp.servers]` to make their tools available in one-shot prompts and interactive conversations. For a local stdio server, set `command` and optional `args` and `env`. The process starts in the config file's directory. For a Streamable HTTP server, set `url` and optionally set `bearer_token_env` to the name of an environment variable containing its bearer token. See `config.example.toml` for both forms.

Arta discovers tools at startup and exposes them as `server__tool` names, with periods replaced by underscores. Set `tools` on a server to list the original names to expose. Omitting it exposes every tool the server advertises. The model can invoke exposed tools automatically, so configure servers and tools you trust to run.

The llama.cpp model needs an embedded chat template that supports tool calls. Arta runs up to eight calls per prompt and feeds text or structured JSON results back into the model. Tool schemas and results count toward the context limit. Each inference step has its own `max_tokens` budget. With tools enabled, Arta buffers inference steps and displays the final reply when it is ready.

Rust callers can implement `arta::tool::Tool` and use `Harness::register_tool` for in-process tools. Call `Harness::connect_mcp` to load configured MCP tools. Create sessions after registering tools so their agents receive the resulting tool set.

## One-shot input and output

Use `--prompt -` to read UTF-8 text from standard input (`stdin`). `arta` writes the response to standard output (`stdout`). Use `--verbose` to see full llama.cpp diagnostics on standard error (`stderr`).

If the model has a chat template, `arta` passes the prompt as one user message. If the template supports a thinking setting, `arta` disables thinking. If the model has no chat template, `arta` passes the prompt as plain text.

## Token limits

- `--context-length` sets the total number of tokens available for the prompt, conversation history, and response. In `config.toml`, use `context_length`. The default is 4096 tokens or the model's trained context length, whichever is lower.
- `--max-tokens` sets the maximum number of tokens `arta` can generate. The default is 256 tokens.

## Use a GPU

1. Install the NVIDIA CUDA Toolkit.
2. Use the `cuda` feature when you build `arta`.
3. Use `--gpu-layers` to specify how many model layers the GPU processes.

```powershell
cargo run --locked --release --features cuda --target-dir target/cuda -- --model .\gguf\Qwen3.5-4B-Q4_0.gguf --prompt "Hello" --gpu-layers 20
```

For the full-screen interface, set `gpu_layers` in the selected model's `config.toml` entry, then run the CUDA executable with no arguments.

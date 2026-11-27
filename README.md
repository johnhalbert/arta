# arta

`arta` runs one-shot prompts, interactive conversations, and workflows with local models. It supports GGUF language models through llama.cpp, Needle 3 through Cactus Compute, image generation through stable-diffusion.cpp, and audio generation through audio.cpp.

For a guided tour of the code, follow a request from startup through model loading, tool calls, workflow execution, and terminal output in the [architecture guide](ARCHITECTURE.md#reading-paths).

See [Arta for Android](android/README.md) for the on-device chat prototype and its build instructions.

## Build and run on Windows

To build `arta`, you need Rust, CMake 3.20 or newer, a C++ compiler, and LLVM's `libclang`. On Windows, use the Visual Studio C++ Build Tools. The default build enables CUDA and requires CUDA Toolkit 12.0 or newer. Use `--no-default-features` for a CPU build.

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

Cargo builds the vendored `llama-cpp-4` 0.11.0 wrapper with Arta's model-load setters. It also downloads and verifies the pinned stable-diffusion.cpp and audio.cpp sources, builds their shared libraries, and stages them beside the executable in the active target profile. On Windows these are `stable-diffusion.dll` and `audiocpp.dll`. The `cuda`, `hip`, and `vulkan` features select GPU backends for all three providers. [Stable-diffusion.cpp source pin](native/stable-diffusion/CMakeLists.txt) · [audio.cpp source pin](native/audio/CMakeLists.txt)

To run `arta` through Cargo, add the same `arta` arguments after `cargo run --release --`.

## Configure providers

Copy `config.example.toml` to `config.toml`. Use `[app]` to select the provider and model. Set model paths relative to the config file.

To select a different config file, use `--config PATH`. To change the selection for one prompt, use `--provider ID` or `--model-id ID`.

For llama.cpp, the existing flags override the config values for one run. A relative path with `--model` starts from the working directory. Use `--verbose=false` to disable detailed diagnostics for one run.

```powershell
.\target\release\arta.exe --config config.toml --prompt "Hello" --max-tokens 32
```

The llama.cpp runtime settings are optional. An entry with just `id` and `path` uses CPU model loading, a context of up to 4096 tokens, greedy decoding, and a 256-token response limit. The settings under the provider apply to each registered model. `config.example.toml` lists the available keys and values.

### Cactus Compute and Needle 3

Arta runs Needle 3 through Cactus’s native library in an isolated Rust worker. On first inference it downloads and verifies the pinned library and weights, then caches them. Set `offline = true` to use cached files, or supply local `library_path` and `weights` paths. Relative paths start from the TOML file's directory. Existing `python` and `telemetry` keys are accepted for config compatibility and ignored by the native worker. Rust library callers can set `worker_path` to an Arta executable if it lives outside their application's directory. [Cactus runtime releases](https://huggingface.co/Cactus-Compute/cactus)

```toml
[app]
provider = "cactus"
model = "needle"

[providers.cactus.runtime]
# offline = true

[[providers.cactus.models]]
kind = "needle"
id = "needle"
```

Configure an MCP server or register an in-process tool to give Needle something to call. Run `arta --config cactus.toml --prompt "What's the weather in Lagos?"` with a weather tool exposed, for example. Needle returns structured calls and Arta runs them, sends the results back to Needle, and displays the results in call order. A request that matches no calls yields a short status message. [Needle model documentation](https://cactuscompute.com/needle)

For structured extraction without visible tools, set `extraction_schema` to a JSON file with Needle's `name`, `description`, and `parameters` fields. Arta returns the extracted argument object or `null`. Tool use takes precedence when tools are visible. Arta's prompt interfaces accept text. [Needle extraction reference](https://www.cactuscompute.com/blog/structured-extraction-with-needle)

`config.example.toml` lists the runtime and per-model settings. Register additional `kind = "needle"` entries with different IDs to select different `.cact` archives or inference settings from the CLI, an agent profile, or `/model`. Use `--max-tokens` for a one-run token budget override. Needle can replay earlier prompts and recorded workflow results. Set `stateless = true` to start each prompt independently while still allowing multiple tool steps within that prompt.

| TOML section | Settings |
| --- | --- |
| `[providers."llama.cpp"]` | Context length, response limit, diagnostics, and NUMA. |
| `[providers."llama.cpp".context]` | CPU threads, batch and microbatch sizes, Flash Attention, KV cache types and offload, RoPE and YaRN, and context memory settings. Omitted fields inherit llama.cpp defaults. Set the provider's `context_length = 0` to use the model's full trained length. Larger explicit values enable context extension on compatible models. |
| `[providers."llama.cpp".sampling]` | Sampler order, seed, temperature, filters, repetition penalties, DRY, XTC, adaptive sampling, Mirostat, logit biases, grammars, JSON Schema, and reasoning budgets. Adding the table opts into llama.cpp's sampler defaults. Set `temp = 0.0` for greedy token selection with the default sampler order. Tool calls use the chat template's grammar. |
| `[providers."llama.cpp".chat]` | Template overrides, template arguments, thinking, tool choice, and stop strings. The model's embedded template is used by default. |
| `[[providers."llama.cpp".models]]` | GGUF path and shards, GPU placement, tensor validation and buffer overrides, file loading and lazy tensor modes, MTP weight loading, and one GGUF metadata override per model. |

Resolve `template_file` and `additional_shards` relative to `config.toml`. Server, quantization, embedding, and speculative decoding options belong to separate llama.cpp workflows.

### Stable Diffusion image models

Download a single-file checkpoint, such as SD 1.5 or SDXL, and register it in `config.toml`. Arta loads the library built by Cargo from beside the executable on first use. [Pinned upstream release](https://github.com/leejet/stable-diffusion.cpp/releases/tag/master-1133-52491bd)

```toml
[providers."stable-diffusion.cpp"]
output_dir = "images"
default_model = "sdxl"

[[providers."stable-diffusion.cpp".models]]
id = "sdxl"
path = "models/sd_xl_base_1.0.safetensors"
options = { width = 1024, height = 1024, steps = 25 }
```

Configured paths are relative to `config.toml`. To use a custom build, set `library_path` to a library built with the same C API and with GGML linked into it. Omit `output_dir` to use `images/`, or `default_model` to use the first listed model. A model entry can also set `vae` to a separate VAE file. Arta reuses the current model context across generations until you switch checkpoints. Each PNG goes in a unique subdirectory of `output_dir`.

Keep `[app]` pointed at your language model. With at least one image model configured, models that support tool calls receive `generate_image` automatically in chat, one-shot prompts, and workflow agent steps. The tool takes `prompt` and optional `model` and `options`, then returns the saved PNG path and model ID. `options` can override `width`, `height`, `steps`, `seed`, and `negative_prompt`. Set a nonnegative `seed` for repeatable runs. An omitted seed uses the model default or a random seed.

### Local audio models

Download a compatible audio.cpp model file or package and register it in `config.toml`. Use `task = "tts"` for speech or `task = "gen"` for music and sound effects. The `family` hint identifies the upstream model family when the package cannot be identified from its path. [audio.cpp model catalog](https://github.com/0xShug0/audio.cpp/tree/dd1ad8bd88497eeb50999d1237c1e19e9d5a2133/model_specs)

```toml
[providers."audio.cpp"]
output_dir = "audio"

[[providers."audio.cpp".models]]
id = "narrator"
path = "models/Supertonic-3-GGUF/supertonic-3-q8_0.gguf"
family = "supertonic"
task = "tts"
options = { voice_id = "F1", language = "en" }
```

Paths are relative to `config.toml`. The provider defaults to the first listed model and saves each WAV to a unique subdirectory of `audio/`. Use `default_model`, `output_dir`, `backend`, or `threads` to override those settings. The backend defaults to the GPU backend selected at build time or CPU in a CPU build. `library_path` can point to a compatible custom audio.cpp shared library.

Keep `[app]` pointed at your language model. With an audio model configured, agents whose language models support tools receive `generate_audio` automatically. The tool takes `prompt` and optional `model` and `options`, and returns the saved WAV path and model ID. Options use the selected audio.cpp model's request keys and string values. For example, `voice_id = "F1"` selects a Supertonic voice and `duration_sec = "5"` requests five seconds from Stable Audio. A tool call or workflow node can override a model's options by key.

## Interactive conversations

Set up `config.toml` in your working directory, then run `arta` with no arguments in your terminal.

```powershell
.\target\release\arta.exe
```

The top-right Model Input pane shows the full text rendered by llama.cpp before tokenization, including the conversation, system instructions, tool definitions, and tool results as rendered by the model's chat template. Use Shift+Page Up and Shift+Page Down to scroll. If a turn runs several inference steps, use Alt+Left and Alt+Right to inspect each input. The pane keeps the inputs for the latest submitted turn until you send another prompt or run `/reset`.

The conversation is in the bottom half. Press Enter to send a prompt. Animated dots appear below Arta's name until the reply starts streaming. The conversation renders Markdown in replies as they arrive, including headings, emphasis, lists, links, code blocks, and tables. Use Page Up and Page Down to read earlier turns, or Ctrl+End to return to the latest reply.

Type `/`, `$`, or `@` at the start of a word to see matching commands, skills, or MCP servers. Use the Up and Down arrow keys to select a suggestion, Tab to complete it, and Enter to submit. Use `/reset` to clear conversation history and active mentions, and `/exit` or Ctrl+C to quit.

Use `/model` to show the current model or `/model <id>` to switch to another model configured for the current provider. Put double quotes around IDs that contain spaces. Arta unloads the current model before loading the replacement and keeps the conversation history. In an interactive session, a load failure keeps the previous selection, which Arta reloads on the next prompt.

Separate slash commands from other text with whitespace. On submission, commands run from left to right and the REPL sends any remaining prompt text to the model. For example, `/reset Explain gravity` clears the conversation before sending `Explain gravity`. If the prompt contains `/exit`, the REPL exits after processing the commands.

Use `$skill-name` to load a configured skill's instructions and `@server-name` to expose a configured MCP server's tools. Mentions work alongside a prompt or on their own and stay active until `/reset`. Repeating a mention reuses the active skill or tools.

With redirected input or output, `arta` reads one prompt per line and writes replies to standard output. It formats each complete reply when standard output is a terminal and streams the original Markdown when standard output is redirected. `/model`, `/reset`, `/exit`, and end-of-input work in this mode. Model command messages go to standard error.

The current conversation counts toward the context limit. Use `/reset` to start a fresh conversation if the history fills the context.

## Define an agent

Save the following as `~/.arta/agents/reviewer.md`. Create the `review` workspace directory or point `workspace` to an existing directory. The filename without `.md` is the agent name. Use letters, digits, underscores, or hyphens in the name.

```markdown
---
provider: llama.cpp
workspace: ~/workspaces/review
config:
  context_length: 4096
  max_tokens: 512
---
Review changes for correctness. Cite the file and line for each finding.
```

Run `arta --agent reviewer` for an interactive conversation or `arta --agent reviewer --prompt "Review this change"` for one reply. The Markdown body supplies the system prompt on every turn, including after `/reset` and during tool calls. If the chat template lacks a system role, llama.cpp folds the instructions into the first user message. For GGUF files without a chat template, Arta adds the instructions to the beginning of the plain-text transcript.

The agent's `provider` and optional `config` override the corresponding settings from `config.toml`. Omit `model` to inherit `[app].model`, or set it to a model ID for the agent's provider. For example, if an agent specifies `model: local` and you rename that model ID in TOML, update or remove the agent's `model` line. For llama.cpp, `config` accepts `model_path`, `context_length`, `max_tokens`, `gpu_layers`, and `verbose`. Set `model_path` to use a GGUF directly. CLI flags take precedence over the agent settings.

List configured servers under `mcp_servers` in the agent's YAML front matter to expose their tools on every REPL turn.

```yaml
mcp_servers:
  - desktop-commander
```

Use the exact names from `[mcp.servers]` in `config.toml`. You can add other servers to the conversation with `@server-name`. After `/reset`, the servers from the agent profile are visible on the next turn.

Relative `workspace` and `model_path` values start from the Markdown file's directory. Both also accept absolute paths and `~/` paths. The workspace is the working directory of local MCP server processes. Model paths, skill directories, and MCP executable paths set in TOML keep their config-relative resolution.

## Run a workflow

Set the workflow root in `config.toml`. Relative paths start from the configuration file's directory. Use `--workflow-path PATH` to override the root for one run.

```toml
[workflows]
path = "workflows"
```

Create `workflows/code-writing-loop/` with `workflow.yaml`, `developer.md`, `tester.md`, and `pull-request.md`. Define the order in `workflow.yaml`.

```yaml
id: code-writing-loop
nodes:
  - id: dev-loop
    type: loop
    nodes:
      - id: code
        type: agent
        agent: developer
      - id: qa
        type: agent
        agent: tester
  - id: pull-request
    type: agent
    agent: pull-request
```

Use the same Markdown front matter and system prompt as for standalone agents. The `agent` field matches the filename without `.md`. Agent settings such as `model`, `workspace`, `mcp_servers`, and `config` apply to that agent. The workspace and agent-level model paths are relative to the Markdown file. CLI model flags override the corresponding settings for agent nodes.

```powershell
arta --workflow code-writing-loop --prompt "Implement the requested change and review it."
```

Nodes run in order. Agent nodes receive the original request and the previous node's output. Each node ID must be unique in the workflow. An agent node keeps its own conversation and tool results across iterations. Arta unloads the current language model before loading a different model or agent-specific provider configuration.

Use `type: image` to generate a PNG directly in a workflow. An image-only workflow needs just `workflow.yaml`.

```yaml
id: illustration
nodes:
  - id: render
    type: image
    model: sdxl
    options:
      seed: 42
```

The `provider` field defaults to `stable-diffusion.cpp`, and `model` defaults to the provider's configured default. Set `prompt` for a literal prompt. When it is omitted, the image node uses the previous node's nonblank output or falls back to the workflow request. Arta releases the workflow's active language model before image generation. An image node outputs the absolute PNG path for the next node or the workflow's final result.

Use `type: audio` to generate a WAV directly. A workflow can consist of just this node.

```yaml
id: narrate
nodes:
  - id: read-intro
    type: audio
    model: narrator
    prompt: "Welcome to the show."
    options: { voice_id: "F2" }
```

The `provider` field defaults to `audio.cpp`. If `prompt` is omitted, the node uses the previous node's nonblank output or the workflow request. Arta releases the active workflow language model before generation and passes the absolute WAV path to the next node or returns it as the workflow result.

The final agent in a loop calls `continue` to repeat the loop or `break` to exit it. Set the optional `message` argument to pass feedback or results to the next node. The run reports an error if the agent finishes without making a decision. Place image and audio nodes before the deciding agent. If the final child is itself a loop, its last agent takes a separate turn to decide whether to repeat the enclosing loop after the inner loop exits.

In the terminal UI, enter `/workflow code-writing-loop` to preview the workflow and run it with your next prompt. Enter `/workflow code-writing-loop Implement the change` to select and run it in one step. The upper-left pane highlights the active node and shows the loop iteration and model changes. Use Ctrl+Page Up and Ctrl+Page Down to scroll through longer workflows. The active node's rendered model input stays in the upper-right pane, and each agent's reply appears in the chat. Ordinary chat resumes after the workflow finishes.

The TUI records each `continue` or `break` call with its arguments in the chat. In the CLI, those calls, node names, loop iterations, and model changes go to standard error. Each node's reply goes to standard output as it becomes available. Use `--prompt -` to read the request from standard input. Rust callers can use `Harness::run_workflow` for the final reply or `Workflow::events` for live progress.

## Use tools during inference

Configure MCP servers under `[mcp.servers]` to make their tools available in one-shot prompts and interactive conversations. For a local stdio server, set `command` and optional `args` and `env`. The process starts in the selected agent's workspace or defaults to the config file's directory. For a Streamable HTTP server, set `url` and optionally set `bearer_token_env` to the name of an environment variable containing its bearer token. See `config.example.toml` for both forms.

Arta discovers tools at startup and exposes them as `server__tool` names, with periods replaced by underscores. One-shot prompts can use tools from all configured servers. In a REPL conversation, expose a server's tools with `@server-name` or the agent's `mcp_servers` setting. Set `tools` on a server to list the original names to expose. Omitting it exposes every tool the server advertises. The model can invoke exposed tools automatically, so configure servers and tools you trust to run.

For llama.cpp, use a model with a chat template that supports tool calls. Arta runs up to eight calls per prompt and feeds text or structured JSON results back into the model. Tool schemas and results count toward the context limit. Each inference step has its own `max_tokens` budget. With tools enabled, full-screen chat previews assistant text during generation. Arta runs tool calls after parsing the complete inference step.

Rust callers can implement `arta::tool::Tool` and use `Harness::register_tool` for in-process tools. Call `Harness::connect_mcp` to load configured MCP tools. Create sessions after registering tools so their agents receive the resulting tool set.

## Use agent skills

List skill directories in `config.toml`. Relative paths start from the configuration file's directory.

```toml
[skills]
directories = [".agents/skills"]
```

Place each skill in its own subdirectory, such as `.agents/skills/review-notes/SKILL.md`. Each file starts with YAML front matter that provides `name` and `description`, followed by Markdown instructions. Use unique names across the configured directories.

When Arta finds skills at startup, it advertises their names and descriptions in the `activate_skill` tool. When the model selects a skill by name, the tool returns the Markdown body. In a REPL conversation, `$skill-name` adds the instructions directly to the agent's context and removes that skill from the activation tool's catalog for the conversation.

## One-shot input and output

Use `--prompt -` to read UTF-8 text from standard input (`stdin`). `arta` renders the complete response as Markdown when standard output (`stdout`) is a terminal. Redirect standard output to a file or pipe it to another command to receive the original Markdown as it streams. Use `--verbose` to see full llama.cpp diagnostics on standard error (`stderr`).

If the model has a chat template, `arta` passes the prompt as one user message. Thinking is off by default and can be enabled with `chat.enable_thinking` for templates that support it. Without a chat template, `arta` passes the prompt as plain text.

## Token limits

- `--context-length` sets the total number of tokens available for the prompt, conversation history, and response. In `config.toml`, use `context_length`. The default is 4096 tokens or the model's trained context length, whichever is lower.
- `--max-tokens` sets the maximum number of tokens `arta` can generate. The default is 256 tokens.

## Use a GPU

Choose the build features for the backends you want available. Each GPU backend requires its development toolkit to build and a compatible driver to run. Linked backend libraries also need their runtime DLLs present at startup, including when you select CPU inference.

| Build | Command | GPU build tools |
| --- | --- | --- |
| CUDA (default) | `cargo build --locked --release` | NVIDIA CUDA Toolkit |
| ROCm | `cargo build --locked --release --no-default-features --features hip` | AMD ROCm or the Windows HIP SDK with Clang |
| Vulkan | `cargo build --locked --release --no-default-features --features vulkan` | Vulkan SDK and shader compiler |
| CUDA and Vulkan | `cargo build --locked --release --features vulkan` | CUDA Toolkit and Vulkan SDK |
| CPU only | `cargo build --locked --release --no-default-features` | — |

Build ROCm with `--no-default-features` to select HIP. The pinned linked build uses a shared backend entry point for CUDA and HIP. llama.cpp also supports shipping backends as separately built, dynamically loaded libraries. [llama.cpp GPU backend build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#notes-about-gpu-accelerated-backends)

For HIP on Windows, select the SDK's `clang` and `clang++` with `CC` and `CXX` in a Visual Studio developer shell. [llama.cpp HIP setup](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#hip)

Set `gpu_layers` in the model's `config.toml` entry to offload weights, or use `--gpu-layers` for one run. Use `devices` for an ordered list of backend names, such as `["CUDA0"]`, `["ROCm0"]`, or `["Vulkan0"]`. Omit it for automatic device selection or set `devices = []` for CPU-only inference in the same executable.

Set `split_mode = "none"` to use the device at the zero-based `main_gpu` index. The default `"layer"` mode distributes layers across devices. The `"row"` and experimental `"tensor"` modes depend on backend and model support. Set `tensor_split` to relative weights in device order.

Set `tensor_buffer_overrides` to an ordered list of entries with `pattern` and `buffer_type`. llama.cpp uses the first matching regular expression for each tensor. Use a registered backend buffer type such as `"CPU"`.

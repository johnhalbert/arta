use anyhow::{Context, Result, ensure};
use arta::{
    agent::Agent,
    provider::llama_cpp::{LlamaBackend, LlamaCPP, LlamaCPPConfig},
};
use clap::Parser;
use std::{
    io::{self, Read, Write},
    num::NonZeroU32,
    path::PathBuf,
    process::ExitCode,
};

#[derive(Parser, Debug)]
#[command(version, about = "Run a single prompt against a local GGUF model")]
struct Args {
    /// Path to a GGUF model file.
    #[arg(short, long, value_name = "PATH")]
    model: PathBuf,

    /// Prompt text. Use - to read UTF-8 text from stdin.
    #[arg(short, long, value_name = "TEXT")]
    prompt: String,

    /// Total token budget for the prompt and response (defaults to 4096 or the model's limit, whichever is lower).
    #[arg(short, long, value_name = "TOKENS")]
    context_length: Option<NonZeroU32>,

    /// Maximum number of tokens to generate.
    #[arg(long, default_value = "256", value_name = "TOKENS")]
    max_tokens: NonZeroU32,

    /// Number of layers to offload to the GPU (requires a CUDA build).
    #[arg(long, default_value_t = 0, value_name = "LAYERS")]
    gpu_layers: u32,

    /// Show llama.cpp diagnostics on stderr.
    #[arg(long)]
    verbose: bool,
}

fn run(args: Args) -> Result<()> {
    let mut prompt = args.prompt;
    if prompt == "-" {
        prompt.clear();
        io::stdin()
            .read_to_string(&mut prompt)
            .context("failed to read UTF-8 prompt from stdin")?;
    }
    ensure!(!prompt.trim().is_empty(), "prompt cannot be empty");

    let mut backend = LlamaBackend::init().context("failed to initialize llama.cpp")?;
    if !args.verbose {
        backend.void_logs();
    }
    let provider = LlamaCPP::new(
        &backend,
        LlamaCPPConfig {
            model_path: args.model,
            context_length: args.context_length,
            max_tokens: args.max_tokens,
            gpu_layers: args.gpu_layers,
        },
    )?;
    let response = Agent::new(provider).prompt(&prompt)?;

    let mut stdout = io::stdout().lock();
    stdout.write_all(response.as_bytes())?;
    if !response.is_empty() && !response.ends_with('\n') {
        stdout.write_all(b"\n")?;
    }
    stdout.flush()?;
    Ok(())
}

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

//! The native Needle session running in the isolated worker process.

use anyhow::{Context, Result, ensure};
use cactus_rs::{
    needle::{CompleteOptions, Needle, Tool, Weights},
    runtime,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{io, num::NonZeroU32, path::PathBuf};

#[derive(Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
enum Request {
    Start {
        threads: u32,
        options: Options,
    },
    Complete {
        text: String,
        max_tokens: NonZeroU32,
    },
}

#[derive(Deserialize)]
struct Options {
    tools: Vec<Tool>,
    system: Option<String>,
    weights: Option<PathBuf>,
    tool_index_path: Option<PathBuf>,
    buffer_size: NonZeroU32,
    auto_date: bool,
}

pub(in crate::provider::cactus) fn run() -> Result<()> {
    let mut session = None;
    crate::provider::cactus::runtime::serve(io::stdin().lock(), io::stdout().lock(), |request| {
        match request {
            Request::Start { threads, options } => {
                ensure!(session.is_none(), "Needle worker is already initialized");
                session = Some((open(threads, &options)?, options.buffer_size));
                Ok(Value::Null)
            }
            Request::Complete { text, max_tokens } => {
                let (needle, buffer_size) = session
                    .as_mut()
                    .context("Needle worker is not initialized")?;
                let response = needle.complete_with_options(
                    text,
                    CompleteOptions::new()
                        .with_max_new_tokens(max_tokens.get())
                        .with_buffer_size(*buffer_size),
                )?;
                let calls: Vec<_> = response
                    .calls()
                    .iter()
                    .map(|call| {
                        json!({
                            "name": call.name(), "arguments": call.arguments(),
                        })
                    })
                    .collect();
                Ok(json!({
                    "type": if response.kind().is_call() { "call" } else { "respond" },
                    "success": true,
                    "function_calls": calls,
                    "reasoning": response.reasoning(),
                    "confidence": response.confidence_score(),
                    "validation": {
                        "ungrounded": response.validation().ungrounded(),
                        "negation": response.validation().negation(),
                    },
                }))
            }
        }
    })
}

fn open(threads: u32, options: &Options) -> Result<Needle> {
    let library =
        runtime::library_path(None).context("could not resolve the Cactus shared library")?;
    // SAFETY: the resolver supplies the pinned Cactus release or a trusted ABI-compatible
    // library explicitly selected by Arta's configuration or environment. This worker loads once.
    unsafe { runtime::load_library(&library) }
        .with_context(|| format!("could not load Cactus engine '{}'", library.display()))?;
    let (weights, calibrated) = match &options.weights {
        Some(path) => {
            let weights = Weights::from_file(path)
                .with_context(|| format!("could not read Needle weights '{}'", path.display()))?;
            let calibrated = weights.has_confidence_head();
            (weights, calibrated)
        }
        None => (
            Weights::fetch().context("could not resolve Needle 3 weights")?,
            true,
        ),
    };
    let mut builder = Needle::builder(weights)
        .threads(threads)
        .system(options.system.as_deref().unwrap_or_default())
        .tools(options.tools.iter().cloned())
        .sdk_grounding(options.auto_date)
        .calibrated_confidence(calibrated);
    if let Some(path) = &options.tool_index_path {
        builder = builder.tool_index(path);
    }
    Ok(builder.build()?)
}

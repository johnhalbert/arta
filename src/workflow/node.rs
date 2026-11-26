//! Structural validation of the workflow tree uses a shared ID set at every depth. This makes
//! node IDs suitable keys for conversation histories.

use crate::provider::{audio::AudioOptions, image::ImageOptions};
use anyhow::{Result, ensure};
use serde::Deserialize;
use std::collections::HashSet;

/// One step in a workflow. A loop executes its children in order on each iteration.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Node {
    /// Invoke the named Markdown profile. The node ID identifies this leaf's conversation
    /// across visits.
    Agent {
        /// Unique key used for this node's history and progress events.
        id: String,
        /// Filename stem of the Markdown profile in the workflow directory.
        agent: String,
    },
    /// Generate a PNG and pass its path to the next node.
    Image {
        /// Unique key used for progress events.
        id: String,
        /// Image provider, defaulting to stable-diffusion.cpp.
        provider: Option<String>,
        /// Model ID, defaulting to the provider's configured default.
        model: Option<String>,
        /// Literal prompt. If omitted, use the previous output or the workflow request.
        prompt: Option<String>,
        /// Per-generation overrides for the selected model's defaults.
        #[serde(default)]
        options: ImageOptions,
    },
    /// Generate a WAV and pass its path to the next node.
    Audio {
        /// Unique key used for progress events.
        id: String,
        /// Audio provider, defaulting to audio.cpp.
        provider: Option<String>,
        /// Model ID, defaulting to the provider's configured default.
        model: Option<String>,
        /// Literal prompt. If omitted, use the previous output or the workflow request.
        prompt: Option<String>,
        /// Per-generation overrides for the selected model's defaults.
        #[serde(default)]
        options: AudioOptions,
    },
    /// Repeat an ordered nonempty sequence until its deciding agent calls the break tool.
    Loop {
        /// Unique key identifying the loop in iteration paths and control calls.
        id: String,
        /// Children in execution order. The deepest final leaf decides whether to repeat.
        nodes: Vec<Node>,
    },
}

impl Node {
    /// Walk depth first, rejecting empty sequences, blank IDs, and duplicate IDs throughout
    /// the tree. Call the visitor once per agent leaf to resolve its profile.
    pub(crate) fn visit_sequence<'a>(
        nodes: &'a [Self],
        ids: &mut HashSet<&'a str>,
        visit: &mut impl FnMut(&str, &str) -> Result<()>,
    ) -> Result<()> {
        ensure!(!nodes.is_empty(), "a workflow or loop must contain nodes");
        for node in nodes {
            let id = match node {
                Self::Agent { id, .. }
                | Self::Image { id, .. }
                | Self::Audio { id, .. }
                | Self::Loop { id, .. } => id,
            };
            ensure!(!id.trim().is_empty(), "node id cannot be empty");
            ensure!(ids.insert(id), "duplicate node id '{id}'");
            match node {
                Self::Agent { agent, .. } => visit(id, agent)?,
                Self::Image {
                    provider,
                    model,
                    prompt,
                    ..
                }
                | Self::Audio {
                    provider,
                    model,
                    prompt,
                    ..
                } => {
                    let kind = if matches!(node, Self::Image { .. }) {
                        "image"
                    } else {
                        "audio"
                    };
                    for (name, value) in
                        [("provider", provider), ("model", model), ("prompt", prompt)]
                    {
                        ensure!(
                            value.as_ref().is_none_or(|text| !text.trim().is_empty()),
                            "{kind} node '{id}' {name} cannot be empty"
                        );
                    }
                }
                Self::Loop { nodes, .. } => {
                    ensure!(
                        !matches!(nodes.last(), Some(Self::Image { .. } | Self::Audio { .. })),
                        "loop '{id}' must end with an agent or a loop that ends with an agent"
                    );
                    Self::visit_sequence(nodes, ids, visit)?;
                }
            }
        }
        Ok(())
    }
}

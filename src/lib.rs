//! Local model inference, conversations, tools, and declarative workflows.
//!
//! Start with [`config::Config`] and [`harness::Harness`]. A harness constructs providers,
//! registers tools, and binds named agents to models in a [`session::Session`]. Supply
//! completed [`provider::Turn`] values to continue a chat. For a sequence of agents with
//! separate histories, load a [`workflow::Workflow`] and poll its events.
//!
//! # A conversation from Rust
//!
//! The caller owns the history. Commit the first turn after its complete reply arrives,
//! then pass it back to the same agent for the follow-up.
//!
//! ```no_run
//! use arta::{config::Config, harness::Harness, model::ModelId, provider::Turn};
//! use std::path::Path;
//!
//! # async fn conversation() -> anyhow::Result<()> {
//! let config = Config::load(Some(Path::new("config.toml")))?;
//! let mut harness = Harness::new(&config)?;
//! let selected = ModelId { provider: &config.app.provider, model: &config.app.model };
//! harness.create_session("local", &[("assistant", selected)])?;
//! let agent = harness.agent("local", "assistant")?;
//!
//! let prompt = "Describe the water cycle.";
//! let reply = agent.chat(&[], prompt).await?.into_text().await?;
//! let history = [Turn { prompt: prompt.into(), reply }];
//! let follow_up = agent.chat(&history, "Which stage produces clouds?").await?;
//! println!("{}", follow_up.into_text().await?);
//! # Ok(())
//! # }
//! ```

pub mod agent;
pub mod config;
mod front_matter;
pub mod harness;
pub mod mcp;
pub mod model;
pub mod provider;
pub mod session;
pub mod skill;
pub mod stream;
pub mod tool;
pub mod workflow;

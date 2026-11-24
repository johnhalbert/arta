//! Named agent bindings and the framed request/reply API. A session can multiplex several
//! agents by name. Each invocation of the stream API addresses one agent and processes its
//! messages in order.

use crate::{
    agent::{Agent, AgentOptions},
    model::ModelId,
    stream::{Frame, Stream},
};
use anyhow::{Context, Result, bail, ensure};
use futures_util::{StreamExt, TryStreamExt, stream};
use std::collections::{HashMap, hash_map::Entry};

/// A group of one or more agents owned by the harness.
pub struct Session {
    /// Bindings indexed by unique session-local names.
    agents: HashMap<String, Agent>,
}

impl Session {
    /// Bind all agents into a temporary map. Reject empty sessions and blank or duplicate
    /// agent IDs before the session can be published by the harness.
    pub(crate) fn new<'a>(
        bindings: impl IntoIterator<Item = (&'a str, ModelId<'a>, AgentOptions<'a>)>,
        bind: impl Fn(ModelId<'a>, AgentOptions<'a>) -> Result<Agent>,
    ) -> Result<Self> {
        let mut agents = HashMap::new();
        for (agent_id, model, options) in bindings {
            ensure!(
                !agent_id.trim().is_empty(),
                "agent identifier cannot be empty"
            );
            match agents.entry(agent_id.to_owned()) {
                Entry::Vacant(entry) => {
                    entry.insert(bind(model, options)?);
                }
                Entry::Occupied(_) => {
                    bail!("agent '{agent_id}' is already registered in this session")
                }
            }
        }
        ensure!(!agents.is_empty(), "a session must have at least one agent");
        Ok(Self { agents })
    }

    /// Borrow an agent by its session-local name.
    pub fn agent(&self, agent_id: &str) -> Result<&Agent> {
        self.agents
            .get(agent_id)
            .with_context(|| format!("unknown agent '{agent_id}'"))
    }

    /// Resolve a session-local name for an exclusive binding update.
    pub(crate) fn agent_mut(&mut self, agent_id: &str) -> Result<&mut Agent> {
        self.agents
            .get_mut(agent_id)
            .with_context(|| format!("unknown agent '{agent_id}'"))
    }

    /// Read complete input messages and stream each reply before reading the next message.
    /// Dropping the reply stream also drops the input stream and any active reply.
    pub fn stream<'a>(&'a self, agent_id: &str, input: Stream<'a>) -> Result<Stream<'a>> {
        let agent = self.agent(agent_id)?;
        let replies = stream::try_unfold(input, move |mut input| async move {
            let Some(prompt) = read_prompt(&mut input).await? else {
                return Ok::<_, anyhow::Error>(None);
            };
            let reply = agent.prompt(&prompt).await?.into_stream();
            Ok(Some((reply, input)))
        });
        Ok(Stream::new(replies.try_flatten()))
    }
}

/// Accumulate chunks between Begin and End. EOF between messages is clean. A misplaced frame
/// or EOF inside a message fails before inference starts.
async fn read_prompt(input: &mut Stream<'_>) -> Result<Option<String>> {
    match input.next().await.transpose()? {
        None => return Ok(None),
        Some(Frame::Begin) => {}
        Some(_) => bail!("expected a message begin frame"),
    }

    let mut prompt = String::new();
    while let Some(frame) = input.next().await.transpose()? {
        match frame {
            Frame::Chunk(text) => prompt.push_str(&text),
            Frame::End => return Ok(Some(prompt)),
            Frame::Begin => {
                bail!("received a message begin frame before the previous message ended")
            }
        }
    }
    bail!("input stream ended before the message end frame")
}

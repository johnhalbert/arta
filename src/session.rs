use crate::{
    agent::Agent,
    model::ModelId,
    stream::{Frame, Stream},
};
use anyhow::{Context, Result, bail, ensure};
use futures_util::{StreamExt, TryStreamExt, stream};
use std::collections::{HashMap, hash_map::Entry};

/// A group of one or more agents owned by the harness.
pub struct Session {
    agents: HashMap<String, Agent>,
}

impl Session {
    pub(crate) fn new(
        bindings: &[(&str, ModelId<'_>)],
        bind: impl Fn(ModelId<'_>) -> Result<Agent>,
    ) -> Result<Self> {
        ensure!(
            !bindings.is_empty(),
            "a session must have at least one agent"
        );
        let mut agents = HashMap::with_capacity(bindings.len());
        for &(agent_id, model) in bindings {
            ensure!(
                !agent_id.trim().is_empty(),
                "agent identifier cannot be empty"
            );
            match agents.entry(agent_id.to_owned()) {
                Entry::Vacant(entry) => {
                    entry.insert(bind(model)?);
                }
                Entry::Occupied(_) => {
                    bail!("agent '{agent_id}' is already registered in this session")
                }
            }
        }
        Ok(Self { agents })
    }

    pub fn agent(&self, agent_id: &str) -> Result<&Agent> {
        self.agents
            .get(agent_id)
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

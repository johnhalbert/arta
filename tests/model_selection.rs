use anyhow::{Context, Result};
use arta::{
    config::Config,
    harness::Harness,
    model::{Model, ModelId},
    provider::Provider,
    stream::{Frame, Message, Stream},
};
use futures_util::{FutureExt, StreamExt, future::BoxFuture, stream};
use std::{
    ptr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering::Relaxed},
    },
};
use tokio::sync::oneshot;

struct TestModel(&'static str);

impl Model for TestModel {
    fn id(&self) -> &str {
        self.0
    }

    fn trained_context_length(&self) -> u32 {
        128
    }
}

struct TestProvider {
    name: &'static str,
    models: Vec<Arc<TestModel>>,
}

impl TestProvider {
    fn new(name: &'static str, models: &[&'static str]) -> Self {
        Self {
            name,
            models: models.iter().map(|&id| Arc::new(TestModel(id))).collect(),
        }
    }
}

impl Provider for TestProvider {
    fn model(&self, model_id: &str) -> Result<Arc<dyn Model>> {
        let model = self
            .models
            .iter()
            .find(|model| model.id() == model_id)
            .with_context(|| format!("unknown model '{model_id}'"))?;
        Ok(model.clone())
    }

    fn send<'a>(&'a self, model_id: &'a str, prompt: &'a str) -> BoxFuture<'a, Result<Message>> {
        Box::pin(async move {
            self.model(model_id)?;
            Ok(Message::new(stream::iter([Ok(format!(
                "{prompt} from {} model {model_id}",
                self.name
            ))])))
        })
    }
}

#[tokio::test]
async fn sessions_keep_agents_bound_to_their_providers_and_models() -> Result<()> {
    let mut harness = Harness::new(&Config::default())?;
    harness.register_provider("first", || {
        Ok(Arc::new(TestProvider::new("first", &["small", "large"])))
    })?;
    harness.register_provider("second", || {
        Ok(Arc::new(TestProvider::new("second", &["small"])))
    })?;

    let cases = [
        ("one", "assistant", "first", "small"),
        ("one", "reviewer", "first", "large"),
        ("two", "assistant", "second", "small"),
        ("two", "shared", "first", "small"),
    ];
    let first = ModelId {
        provider: "first",
        model: "small",
    };
    let large = ModelId {
        provider: "first",
        model: "large",
    };
    let second = ModelId {
        provider: "second",
        model: "small",
    };
    harness.create_session("one", &[("assistant", first), ("reviewer", large)])?;
    harness.create_session("two", &[("assistant", second), ("shared", first)])?;

    for (session, agent_id, provider, model) in cases {
        let agent = harness.agent(session, agent_id)?;
        assert!(ptr::addr_eq(agent.provider(), harness.provider(provider)?));
        let handle = harness.model(ModelId { provider, model })?;
        assert!(ptr::addr_eq(agent.model(), handle.as_ref()));
        assert_eq!(
            agent.prompt("Hello").await?.into_text().await?,
            format!("Hello from {provider} model {model}")
        );
    }
    harness.close_session("one")?;
    assert!(harness.session("one").is_err());
    assert_eq!(harness.model(first)?.id(), "small");
    assert_eq!(
        harness
            .agent("two", "shared")?
            .prompt("Again")
            .await?
            .into_text()
            .await?,
        "Again from first model small"
    );
    assert_eq!(
        harness
            .agent("two", "assistant")?
            .prompt("Again")
            .await?
            .into_text()
            .await?,
        "Again from second model small"
    );
    Ok(())
}

#[test]
fn an_invalid_agent_binding_does_not_register_a_session() -> Result<()> {
    let mut harness = Harness::new(&Config::default())?;
    harness.register_provider("known", || {
        Ok(Arc::new(TestProvider::new("known", &["small"])))
    })?;

    let valid = ModelId {
        provider: "known",
        model: "small",
    };
    let missing_provider = ModelId {
        provider: "missing",
        model: "small",
    };
    let missing_model = ModelId {
        provider: "known",
        model: "missing",
    };
    assert_eq!(
        harness
            .create_session("test", &[("assistant", missing_provider)])
            .unwrap_err()
            .to_string(),
        "unknown provider 'missing'"
    );
    assert_eq!(
        harness
            .create_session("test", &[("assistant", valid), ("reviewer", missing_model)])
            .unwrap_err()
            .to_string(),
        "unknown model 'missing'"
    );
    assert!(harness.session("test").is_err());
    harness.create_session("test", &[("assistant", valid)])?;
    assert!(harness.agent("test", "reviewer").is_err());
    Ok(())
}

#[test]
fn harness_rejects_empty_sessions_and_duplicate_names() -> Result<()> {
    let mut harness = Harness::new(&Config::default())?;
    harness.register_provider("known", || {
        Ok(Arc::new(TestProvider::new("known", &["small"])))
    })?;
    let model = ModelId {
        provider: "known",
        model: "small",
    };

    assert!(harness.create_session("test", &[]).is_err());
    assert!(
        harness
            .create_session("test", &[("agent", model), ("agent", model)])
            .is_err()
    );
    assert!(harness.session("test").is_err());
    harness.create_session("test", &[("agent", model)])?;
    assert!(harness.create_session("test", &[("other", model)]).is_err());
    assert_eq!(harness.agent("test", "agent")?.model().id(), "small");
    Ok(())
}

#[test]
fn harness_rejects_duplicate_registration_before_creating_a_provider() -> Result<()> {
    let mut harness = Harness::new(&Config::default())?;
    harness.register_provider("known", || {
        Ok(Arc::new(TestProvider::new("known", &["small"])))
    })?;

    let duplicate = harness.register_provider("known", || {
        panic!("duplicate provider factory ran");
    });
    assert_eq!(
        duplicate.unwrap_err().to_string(),
        "provider 'known' is already registered"
    );
    Ok(())
}

#[derive(Default)]
struct Activity {
    prompts: Mutex<Vec<String>>,
    chunks: AtomicUsize,
}

struct StreamingProvider {
    model: Arc<TestModel>,
    activity: Arc<Activity>,
}

impl Provider for StreamingProvider {
    fn model(&self, model_id: &str) -> Result<Arc<dyn Model>> {
        assert_eq!(model_id, self.model.id());
        Ok(self.model.clone())
    }

    fn send<'a>(&'a self, model_id: &'a str, prompt: &'a str) -> BoxFuture<'a, Result<Message>> {
        Box::pin(async move {
            self.model(model_id)?;
            self.activity
                .prompts
                .lock()
                .unwrap()
                .push(prompt.to_owned());
            let activity = Arc::clone(&self.activity);
            let chunks = [format!("{prompt}!"), " tail".into()];
            Ok(Message::new(stream::iter(chunks).map(move |chunk| {
                activity.chunks.fetch_add(1, Relaxed);
                Ok(chunk)
            })))
        })
    }
}

fn streaming_harness(activity: &Arc<Activity>) -> Result<Harness> {
    let mut harness = Harness::new(&Config::default())?;
    harness.register_provider("streaming", || {
        Ok(Arc::new(StreamingProvider {
            model: Arc::new(TestModel("model")),
            activity: activity.clone(),
        }))
    })?;
    harness.create_session(
        "chat",
        &[(
            "agent",
            ModelId {
                provider: "streaming",
                model: "model",
            },
        )],
    )?;
    Ok(harness)
}

#[tokio::test]
async fn session_buffers_each_prompt_and_only_pulls_requested_output() -> Result<()> {
    let activity = Arc::new(Activity::default());
    let harness = streaming_harness(&activity)?;
    let input_frames = [
        Frame::Begin,
        Frame::Chunk("hel".into()),
        Frame::Chunk("lo".into()),
        Frame::End,
        Frame::Begin,
        Frame::Chunk("again".into()),
        Frame::End,
    ];
    let frames_read = AtomicUsize::new(0);
    let input = Stream::new(stream::iter(input_frames).map(|frame| {
        if frames_read.fetch_add(1, Relaxed) < 4 {
            assert!(activity.prompts.lock().unwrap().is_empty());
        }
        Ok(frame)
    }));
    let mut replies = harness.session("chat")?.stream("agent", input)?;
    assert_eq!(frames_read.load(Relaxed), 0);

    assert_eq!(replies.next().await.transpose()?, Some(Frame::Begin));
    assert_eq!(activity.prompts.lock().unwrap().as_slice(), ["hello"]);
    assert_eq!(activity.chunks.load(Relaxed), 0);
    assert_eq!(
        replies.next().await.transpose()?,
        Some(Frame::Chunk("hello!".into()))
    );
    assert_eq!(activity.chunks.load(Relaxed), 1);
    assert_eq!(
        replies.next().await.transpose()?,
        Some(Frame::Chunk(" tail".into()))
    );
    assert_eq!(replies.next().await.transpose()?, Some(Frame::End));
    assert_eq!(frames_read.load(Relaxed), 4);

    assert_eq!(replies.next().await.transpose()?, Some(Frame::Begin));
    assert_eq!(
        replies.next().await.transpose()?,
        Some(Frame::Chunk("again!".into()))
    );
    drop(replies);
    assert_eq!(activity.chunks.load(Relaxed), 3);
    assert_eq!(
        activity.prompts.lock().unwrap().as_slice(),
        ["hello", "again"]
    );
    Ok(())
}

#[tokio::test]
async fn waiting_for_an_input_frame_leaves_other_turns_runnable() -> Result<()> {
    let activity = Arc::new(Activity::default());
    let harness = streaming_harness(&activity)?;
    let session = harness.session("chat")?;
    let (finish, end) = oneshot::channel();
    let input = Stream::new(
        stream::iter([Ok(Frame::Begin), Ok(Frame::Chunk("waiting".into()))]).chain(stream::once(
            async {
                end.await?;
                Ok(Frame::End)
            },
        )),
    );
    let mut pending = session.stream("agent", input)?;
    assert!(pending.next().now_or_never().is_none());
    assert!(activity.prompts.lock().unwrap().is_empty());

    let other_input = Message::new(stream::iter([Ok("ready".into())]));
    let mut ready = session.stream("agent", other_input.into_stream())?;
    assert_eq!(ready.next().await.transpose()?, Some(Frame::Begin));
    assert_eq!(
        ready.next().await.transpose()?,
        Some(Frame::Chunk("ready!".into()))
    );
    assert_eq!(activity.prompts.lock().unwrap().as_slice(), ["ready"]);

    finish.send(()).unwrap();
    assert_eq!(pending.next().await.transpose()?, Some(Frame::Begin));
    assert_eq!(
        pending.next().await.transpose()?,
        Some(Frame::Chunk("waiting!".into()))
    );
    Ok(())
}

#[tokio::test]
async fn malformed_input_never_reaches_the_agent() -> Result<()> {
    let activity = Arc::new(Activity::default());
    let harness = streaming_harness(&activity)?;
    let session = harness.session("chat")?;
    let cases = [
        (
            vec![Frame::Chunk("orphan".into())],
            "expected a message begin frame",
        ),
        (
            vec![Frame::Begin, Frame::Begin],
            "before the previous message ended",
        ),
        (
            vec![Frame::Begin, Frame::Chunk("partial".into())],
            "ended before the message end frame",
        ),
    ];

    for (frames, expected) in cases {
        let mut replies = session.stream(
            "agent",
            Stream::new(stream::iter(frames.into_iter().map(Ok))),
        )?;
        assert!(
            replies
                .next()
                .await
                .unwrap()
                .unwrap_err()
                .to_string()
                .contains(expected)
        );
        assert!(replies.next().await.is_none());
    }
    assert!(activity.prompts.lock().unwrap().is_empty());
    Ok(())
}

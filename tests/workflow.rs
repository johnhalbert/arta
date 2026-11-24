//! Workflow execution contracts. Temporary definitions and Markdown profiles exercise the
//! real loader. Scripted provider responses make history replay, model release, tool calls,
//! and loop control observable.

use anyhow::Result;
use arta::{
    config::Config,
    harness::Harness,
    model::Model,
    provider::{
        PromptObserver, Provider,
        tool::{ToolAdapter, ToolCall, ToolExchange, ToolRequest, ToolResponse},
    },
    stream::Message,
    tool::{Tool, ToolArguments, ToolOutput, ToolSpec},
    workflow::{LoopIteration, Workflow, WorkflowEvent},
};
use futures_util::{StreamExt, future::BoxFuture};
use serde_json::Value;
use std::{
    collections::VecDeque,
    fs,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

const DEFINITION: &str = "\
id: code-writing-loop
nodes:
  - id: dev-loop
    type: loop
    nodes:
    - id: code
      agent: developer
      type: agent
    - id: rehearse
      agent: developer
      type: agent
    - id: qa
      agent: tester
      type: agent
  - id: pull-request
    type: agent
    agent: pull-request
";

/// Owns the temporary workflow package and its configuration for one test.
struct Fixture(PathBuf);

impl Fixture {
    fn new(definition: &str) -> Result<Self> {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let fixture = Self(std::env::temp_dir().join(format!(
            "arta-workflow-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )));
        let directory = fixture.0.join("workflows/code-writing-loop");
        fs::create_dir_all(&directory)?;
        fs::write(
            fixture.0.join("settings.toml"),
            "[workflows]\npath = 'workflows'\n",
        )?;
        fs::write(directory.join("workflow.yaml"), definition)?;
        for (name, provider, model) in [
            ("developer", "fake", "code"),
            ("tester", "fake", "qa"),
            ("pull-request", "another", "publish"),
        ] {
            fs::write(
                directory.join(format!("{name}.md")),
                format!("---\nprovider: {provider}\nmodel: {model}\nworkspace: .\n---\n{name}\n"),
            )?;
        }
        Ok(fixture)
    }

    fn config(&self) -> Result<Config> {
        Config::load(Some(&self.0.join("settings.toml")))
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Shared script queue, resident test model, and chronological resource events across
/// provider instances.
#[derive(Default)]
struct Runtime {
    loaded: Option<Arc<TestModel>>,
    events: Vec<String>,
    steps: VecDeque<Step>,
}

struct TestModel(String);

impl Model for TestModel {
    fn id(&self) -> &str {
        &self.0
    }
    fn trained_context_length(&self) -> u32 {
        4096
    }
}

struct TestProvider(Arc<Mutex<Runtime>>);

enum Answer {
    Text(&'static str),
    Call(&'static str, &'static str),
}

/// Expected inputs and scripted outcome for one inference step, including prior tool
/// exchanges.
struct Step {
    model: &'static str,
    instructions: &'static str,
    input_end: &'static str,
    history: &'static [&'static str],
    tools: &'static [&'static str],
    remembered_tool: Option<&'static str>,
    answer: Answer,
}

impl Provider for TestProvider {
    fn model(&self, id: &str) -> Result<Arc<dyn Model>> {
        let mut runtime = self.0.lock().unwrap();
        assert!(runtime.loaded.is_none());
        let model = Arc::new(TestModel(id.into()));
        runtime.loaded = Some(model.clone());
        runtime.events.push(format!("load {id}"));
        Ok(model)
    }

    fn unload_model(&self, id: &str) -> Result<()> {
        let mut runtime = self.0.lock().unwrap();
        let model = runtime.loaded.take().unwrap();
        assert_eq!(model.id(), id);
        assert_eq!(Arc::strong_count(&model), 1);
        runtime.events.push(format!("unload {id}"));
        Ok(())
    }

    fn switch_model(&self, current: &str, target: &str) -> Result<Arc<dyn Model>> {
        self.unload_model(current)?;
        self.model(target)
    }

    fn send<'a>(&'a self, _: &'a str, _: &'a str) -> BoxFuture<'a, Result<Message>> {
        unreachable!("the test registers a tool for every turn")
    }

    fn tool_adapter(&self) -> Option<&dyn ToolAdapter> {
        Some(self)
    }
}

impl ToolAdapter for TestProvider {
    fn infer<'a>(&'a self, _: ToolRequest<'a>) -> BoxFuture<'a, Result<ToolResponse>> {
        unreachable!("workflow turns include recorded history")
    }

    fn infer_recorded<'a>(
        &'a self,
        request: ToolRequest<'a>,
        past: &'a [Vec<ToolExchange>],
        observer: Option<PromptObserver>,
    ) -> BoxFuture<'a, Result<ToolResponse>> {
        Box::pin(async move {
            if let Some(observe) = observer {
                observe(request.system_prompt.unwrap());
            }
            let mut runtime = self.0.lock().unwrap();
            let step = runtime.steps.pop_front().expect("unexpected inference");
            assert_eq!(runtime.loaded.as_ref().unwrap().id(), step.model);
            assert_eq!(request.model_id, step.model);
            assert!(request.system_prompt.unwrap().contains(step.instructions));
            assert!(
                request.prompt.starts_with("Build it")
                    || request.prompt.starts_with("Workflow request:\nBuild it")
            );
            assert!(request.prompt.ends_with(step.input_end));
            assert_eq!(
                request
                    .history
                    .iter()
                    .map(|turn| turn.reply.as_str())
                    .collect::<Vec<_>>(),
                step.history
            );
            assert_eq!(
                request
                    .tools
                    .iter()
                    .map(|tool| tool.name.as_str())
                    .collect::<Vec<_>>(),
                step.tools
            );
            let remembered = past
                .iter()
                .flatten()
                .flat_map(|exchange| &exchange.calls)
                .next()
                .map(|(call, output)| (call.name.as_str(), output.content.as_str().unwrap()));
            assert_eq!(remembered.map(|(name, _)| name), step.remembered_tool);
            if let Some(("lookup", value)) = remembered {
                assert_eq!(value, "saved detail");
            }
            if let Some(("continue", value)) = remembered {
                assert_eq!(value, "fix requested");
            }
            if step.input_end == "Build it" && !request.exchanges.is_empty() {
                assert_eq!(request.exchanges[0].calls[0].1.content, "saved detail");
            }
            Ok(match step.answer {
                Answer::Text(text) => ToolResponse::Reply(text.into()),
                Answer::Call(name, message) => {
                    let mut arguments = ToolArguments::new();
                    if !message.is_empty() {
                        arguments.insert("message".into(), Value::from(message));
                    }
                    let mut calls = vec![ToolCall {
                        id: "1".into(),
                        name: name.into(),
                        arguments,
                    }];
                    if name == "continue" {
                        calls.push(ToolCall {
                            id: "2".into(),
                            name: "break".into(),
                            arguments: ToolArguments::new(),
                        });
                    }
                    ToolResponse::Calls {
                        content: None,
                        reasoning: None,
                        calls,
                    }
                }
            })
        })
    }
}

struct Lookup(ToolSpec);

impl Tool for Lookup {
    fn spec(&self) -> &ToolSpec {
        &self.0
    }

    fn call(&self, _: ToolArguments) -> BoxFuture<'_, Result<ToolOutput>> {
        Box::pin(async { Ok(ToolOutput::success("saved detail")) })
    }
}

fn setup(
    fixture: &Fixture,
    steps: impl Into<VecDeque<Step>>,
) -> Result<(Harness, Arc<Mutex<Runtime>>)> {
    let mut harness = Harness::new(&fixture.config()?)?;
    let runtime = Arc::new(Mutex::new(Runtime {
        steps: steps.into(),
        ..Runtime::default()
    }));
    for id in ["fake", "another"] {
        harness.register_provider(id, || Ok(Arc::new(TestProvider(runtime.clone()))))?;
    }
    harness.register_tool(Arc::new(Lookup(ToolSpec {
        name: "lookup".into(),
        description: "Look up a detail".into(),
        input_schema: Default::default(),
    })))?;
    Ok((harness, runtime))
}

#[tokio::test]
async fn keeps_per_node_context_and_releases_models_before_switching() -> Result<()> {
    use Answer::{Call, Text};
    let fixture = Fixture::new(DEFINITION)?;
    let lookup = &["lookup"];
    let control = &["break", "continue", "lookup"];
    let steps = [
        Step {
            model: "code",
            instructions: "developer",
            input_end: "Build it",
            history: &[],
            tools: lookup,
            remembered_tool: None,
            answer: Call("lookup", ""),
        },
        Step {
            model: "code",
            instructions: "developer",
            input_end: "Build it",
            history: &[],
            tools: lookup,
            remembered_tool: None,
            answer: Text("draft"),
        },
        Step {
            model: "code",
            instructions: "developer",
            input_end: "draft",
            history: &[],
            tools: lookup,
            remembered_tool: None,
            answer: Text("rehearsed draft"),
        },
        Step {
            model: "qa",
            instructions: "loop 'dev-loop'",
            input_end: "rehearsed draft",
            history: &[],
            tools: control,
            remembered_tool: None,
            answer: Call("continue", "fix requested"),
        },
        Step {
            model: "code",
            instructions: "developer",
            input_end: "fix requested",
            history: &["draft"],
            tools: lookup,
            remembered_tool: Some("lookup"),
            answer: Text("revised draft"),
        },
        Step {
            model: "code",
            instructions: "developer",
            input_end: "revised draft",
            history: &["rehearsed draft"],
            tools: lookup,
            remembered_tool: None,
            answer: Text("rehearsed revision"),
        },
        Step {
            model: "qa",
            instructions: "loop 'dev-loop'",
            input_end: "rehearsed revision",
            history: &["fix requested"],
            tools: control,
            remembered_tool: Some("continue"),
            answer: Call("break", "approved"),
        },
        Step {
            model: "publish",
            instructions: "pull-request",
            input_end: "approved",
            history: &[],
            tools: lookup,
            remembered_tool: None,
            answer: Text("ready to publish"),
        },
    ];
    let (mut harness, runtime) = setup(&fixture, steps)?;
    let workflow = Workflow::load(&fixture.config()?, "code-writing-loop")?;
    let mut events = workflow.events(&mut harness, "Build it", true);
    let mut nodes = Vec::new();
    let mut switches = Vec::new();
    let mut outputs = Vec::new();
    let mut inputs = 0;
    let mut finished = 0;
    let mut loop_calls = Vec::new();
    assert_eq!(
        events.next().await.transpose()?,
        Some(WorkflowEvent::NodeStarted {
            id: "code".into(),
            agent: "developer".into(),
            loops: vec![LoopIteration {
                id: "dev-loop".into(),
                number: 1
            }],
        })
    );
    assert!(runtime.lock().unwrap().events.is_empty());
    while let Some(event) = events.next().await.transpose()? {
        match event {
            WorkflowEvent::NodeStarted { id, loops, .. } => {
                nodes.push((id, loops.first().map(|iteration| iteration.number)))
            }
            WorkflowEvent::ModelLoading { previous, next } => switches.push((previous, next)),
            WorkflowEvent::Output(text) => outputs.push(text),
            WorkflowEvent::ModelInput(_) => inputs += 1,
            WorkflowEvent::LoopToolCall {
                iteration,
                name,
                arguments,
                error,
            } => {
                loop_calls.push((iteration, name, arguments, error));
            }
            WorkflowEvent::NodeFinished => finished += 1,
            WorkflowEvent::ModelLoaded { .. } => {}
        }
    }
    assert_eq!(
        nodes
            .iter()
            .map(|(id, iteration)| (id.as_str(), *iteration))
            .collect::<Vec<_>>(),
        [
            ("rehearse", Some(1)),
            ("qa", Some(1)),
            ("code", Some(2)),
            ("rehearse", Some(2)),
            ("qa", Some(2)),
            ("pull-request", None),
        ]
    );
    assert_eq!(switches[0], (None, "fake/code".into()));
    assert_eq!(switches[1], (Some("fake/code".into()), "fake/qa".into()));
    assert_eq!(
        switches.last(),
        Some(&(Some("fake/qa".into()), "another/publish".into()))
    );
    assert_eq!(switches.len(), 5);
    assert_eq!(inputs, 8);
    assert_eq!(finished, 7);
    assert_eq!(loop_calls.len(), 2);
    for ((iteration, name, arguments, error), (number, tool, message)) in loop_calls
        .iter()
        .zip([(1, "continue", "fix requested"), (2, "break", "approved")])
    {
        assert_eq!(iteration.id, "dev-loop");
        assert_eq!(iteration.number, number);
        assert_eq!(name, tool);
        assert_eq!(arguments["message"], message);
        assert_eq!(*error, None);
    }
    assert_eq!(
        outputs,
        [
            "draft",
            "rehearsed draft",
            "fix requested",
            "revised draft",
            "rehearsed revision",
            "approved",
            "ready to publish"
        ]
    );
    let runtime = runtime.lock().unwrap();
    assert!(runtime.loaded.is_none());
    assert!(runtime.steps.is_empty());
    assert_eq!(
        runtime.events,
        [
            "load code",
            "unload code",
            "load qa",
            "unload qa",
            "load code",
            "unload code",
            "load qa",
            "unload qa",
            "load publish",
            "unload publish"
        ]
    );
    Ok(())
}

#[tokio::test]
async fn missing_loop_decision_stops_the_run_and_releases_the_model() -> Result<()> {
    let fixture = Fixture::new(
        "id: missing-decision\nnodes:\n  - id: loop\n    type: loop\n    nodes:\n      - { id: qa, type: agent, agent: tester }\n  - { id: later, type: agent, agent: pull-request }\n",
    )?;
    let (mut harness, runtime) = setup(
        &fixture,
        [Step {
            model: "qa",
            instructions: "loop 'loop'",
            input_end: "Build it",
            history: &[],
            tools: &["break", "continue", "lookup"],
            remembered_tool: None,
            answer: Answer::Text("approved in prose"),
        }],
    )?;
    let error = harness
        .run_workflow("code-writing-loop", "Build it")
        .await
        .unwrap_err();
    assert!(
        format!("{error:#}").contains("requires the final agent to call 'continue' or 'break'")
    );
    assert_eq!(runtime.lock().unwrap().events, ["load qa", "unload qa"]);
    Ok(())
}

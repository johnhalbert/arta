use arta::{agent::Agent, model::Model, provider::Provider};

struct TestModel(&'static str);

impl Model for TestModel {
    fn trained_context_length(&self) -> u32 {
        128
    }
}

struct TestProvider;

impl Provider for TestProvider {
    type Model<'provider> = TestModel;

    fn send<'provider>(
        &'provider self,
        model: &Self::Model<'provider>,
        prompt: &str,
    ) -> anyhow::Result<String> {
        Ok(format!("{prompt} from {}", model.0))
    }
}

#[test]
fn agent_uses_the_model_selected_for_each_prompt() -> anyhow::Result<()> {
    let first = TestModel("first model");
    let second = TestModel("second model");
    let provider = TestProvider;
    let agent = Agent::new(&provider);

    assert_eq!(agent.prompt(&first, "Hello")?, "Hello from first model");
    assert_eq!(agent.prompt(&second, "Hello")?, "Hello from second model");
    assert_eq!(agent.prompt(&first, "Again")?, "Again from first model");
    Ok(())
}

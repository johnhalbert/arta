//! Configuration and registry contracts. These tests build provider registries and TOML
//! tables in memory to check discovery, name validation, and the precise scope of CLI
//! overrides.

use anyhow::Result;
use arta::{
    config::Config,
    harness::Harness,
    provider::{
        llama_cpp::{LlamaCPP, LlamaCPPConfig, LlamaCPPOverrides},
        registry::ProviderRegistry,
    },
};
use std::{
    num::NonZeroU32,
    path::{self, Path},
};

#[test]
fn harness_creates_all_discovered_providers() -> Result<()> {
    let registry = ProviderRegistry::discover()?;
    assert!(registry.list().any(|(id, _)| id == LlamaCPP::ID));
    let harness = Harness::new(&Config::default())?;
    for (id, _) in registry.list() {
        assert!(harness.provider(id).is_ok());
    }
    Ok(())
}

#[test]
fn registry_rejects_duplicate_factory_identifiers() -> Result<()> {
    let mut registry = ProviderRegistry::discover()?;
    let (_, factory) = registry.list().find(|(id, _)| *id == LlamaCPP::ID).unwrap();
    assert!(registry.register(LlamaCPP::ID, factory).is_err());
    registry.register("alias", factory)?;
    let harness = Harness::with_registry(&Config::default(), &registry)?;
    assert!(harness.provider(LlamaCPP::ID).is_ok());
    assert!(harness.provider("alias").is_ok());
    Ok(())
}

#[test]
fn harness_rejects_an_unknown_configured_provider() -> Result<()> {
    let config: Config = toml::from_str("[providers.typo]")?;
    let error = Harness::new(&config).err().unwrap();
    assert_eq!(error.to_string(), "unknown configured provider 'typo'");
    Ok(())
}

#[test]
fn cli_overrides_change_only_the_selected_settings() -> Result<()> {
    let file = Path::new(env!("CARGO_MANIFEST_DIR")).join("config.example.toml");
    let mut config = Config::load(Some(&file))?;
    let original: LlamaCPPConfig = config.provider_settings(LlamaCPP::ID)?;
    assert_eq!(config.app.model, "local");
    assert_eq!(
        config.resolve_path(&original.models[0].path),
        file.with_file_name("gguf").join("Qwen3.5-4B-Q4_0.gguf")
    );

    LlamaCPPOverrides {
        model_path: Some("replacement.gguf".into()),
        context_length: None,
        max_tokens: NonZeroU32::new(32),
        gpu_layers: Some(5),
        verbose: None,
    }
    .apply(&mut config)?;

    let merged: LlamaCPPConfig = config.provider_settings(LlamaCPP::ID)?;
    assert_eq!(merged.models[0].id, "local");
    assert_eq!(merged.models[0].path, path::absolute("replacement.gguf")?);
    assert_eq!(merged.models[0].gpu_layers, 5);
    assert_eq!(merged.max_tokens.get(), 32);
    assert_eq!(merged.context_length, original.context_length);
    assert_eq!(merged.verbose, original.verbose);
    Ok(())
}

#[test]
fn runtime_settings_survive_cli_overrides() -> Result<()> {
    let mut config: Config = toml::from_str(
        r#"
[providers."llama.cpp"]
context_length = 0
numa = "distribute"

[providers."llama.cpp".context]
n_threads = 4
flash_attention = "auto"
cache_type_k = "q8_0"

[providers."llama.cpp".sampling]
seed = 42
samplers = ["top_k", "temperature"]
temp = 0.0
logit_bias = [{ token = 123, bias = -2.0 }]

[providers."llama.cpp".chat]
template_file = "chat.jinja"
enable_thinking = true
stop = ["END"]

[[providers."llama.cpp".models]]
id = "local"
path = "first.gguf"
additional_shards = ["second.gguf"]
devices = ["CUDA0", "CUDA1"]
split_mode = "layer"
tensor_split = [3.0, 1.0]
tensor_buffer_overrides = [{ pattern = '^blk\.0\.', buffer_type = "CPU" }]
check_tensors = true
use_extra_bufts = false
no_host = true
load_mode = "mmap+mlock"
kv_override = { key = "tokenizer.ggml.add_bos_token", value = true }
"#,
    )?;
    let original: LlamaCPPConfig = config.provider_settings(LlamaCPP::ID)?;
    let mut expected = toml::Table::try_from(original)?;
    expected["max_tokens"] = toml::Value::Integer(32);
    LlamaCPPOverrides {
        max_tokens: NonZeroU32::new(32),
        ..Default::default()
    }
    .apply(&mut config)?;
    let merged: LlamaCPPConfig = config.provider_settings(LlamaCPP::ID)?;
    assert_eq!(toml::Table::try_from(merged)?, expected);

    LlamaCPPOverrides {
        model_path: Some("replacement.gguf".into()),
        ..Default::default()
    }
    .apply(&mut config)?;
    let merged: LlamaCPPConfig = config.provider_settings(LlamaCPP::ID)?;
    assert!(merged.models[0].additional_shards.is_empty());
    assert!(merged.models[0].kv_override.is_some());
    Ok(())
}

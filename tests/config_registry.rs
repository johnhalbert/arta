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

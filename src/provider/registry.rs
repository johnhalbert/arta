//! Link-time discovery of provider constructors. The registry stores factories. The harness
//! invokes them to own distinct provider instances.

use super::Provider;
use crate::config::Config;
use anyhow::{Result, bail, ensure};
use std::{
    collections::{BTreeMap, btree_map::Entry},
    sync::Arc,
};

/// Constructor that interprets the supplied application's settings and returns a shared
/// provider.
pub type ProviderFactory = fn(&Config) -> Result<Arc<dyn Provider>>;

/// A factory declared by a provider module with `inventory::submit!`.
pub struct ProviderRegistration {
    /// Stable key used by TOML, the CLI, and harness lookups.
    pub id: &'static str,
    /// Constructor invoked during harness initialization.
    pub create: ProviderFactory,
}

inventory::collect!(ProviderRegistration);

/// Lists provider factories before a harness creates their instances.
#[derive(Default)]
pub struct ProviderRegistry {
    /// Constructors keyed by provider ID for deterministic harness initialization.
    factories: BTreeMap<String, ProviderFactory>,
}

impl ProviderRegistry {
    /// Collect every linked registration and reject colliding IDs.
    pub fn discover() -> Result<Self> {
        let mut registry = Self::default();
        for registration in inventory::iter::<ProviderRegistration> {
            registry.register(registration.id, registration.create)?;
        }
        Ok(registry)
    }

    /// Insert a factory under a nonblank unique ID without instantiating it.
    pub fn register(&mut self, provider_id: &str, create: ProviderFactory) -> Result<()> {
        ensure!(
            !provider_id.trim().is_empty(),
            "provider identifier cannot be empty"
        );
        match self.factories.entry(provider_id.to_owned()) {
            Entry::Vacant(entry) => {
                entry.insert(create);
                Ok(())
            }
            Entry::Occupied(_) => bail!("provider factory '{provider_id}' is already registered"),
        }
    }

    /// Yield factories in provider ID order.
    pub fn list(&self) -> impl Iterator<Item = (&str, ProviderFactory)> {
        self.factories
            .iter()
            .map(|(id, &create)| (id.as_str(), create))
    }

    /// Check that a provider table has a corresponding registered factory.
    pub(crate) fn contains(&self, provider_id: &str) -> bool {
        self.factories.contains_key(provider_id)
    }
}

use super::Provider;
use crate::config::Config;
use anyhow::{Result, bail, ensure};
use std::{
    collections::{BTreeMap, btree_map::Entry},
    sync::Arc,
};

pub type ProviderFactory = fn(&Config) -> Result<Arc<dyn Provider>>;

/// A factory declared by a provider module with `inventory::submit!`.
pub struct ProviderRegistration {
    pub id: &'static str,
    pub create: ProviderFactory,
}

inventory::collect!(ProviderRegistration);

/// Lists provider factories before a harness creates their instances.
#[derive(Default)]
pub struct ProviderRegistry {
    factories: BTreeMap<String, ProviderFactory>,
}

impl ProviderRegistry {
    pub fn discover() -> Result<Self> {
        let mut registry = Self::default();
        for registration in inventory::iter::<ProviderRegistration> {
            registry.register(registration.id, registration.create)?;
        }
        Ok(registry)
    }

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

    pub fn list(&self) -> impl Iterator<Item = (&str, ProviderFactory)> {
        self.factories
            .iter()
            .map(|(id, &create)| (id.as_str(), create))
    }

    pub(crate) fn contains(&self, provider_id: &str) -> bool {
        self.factories.contains_key(provider_id)
    }
}

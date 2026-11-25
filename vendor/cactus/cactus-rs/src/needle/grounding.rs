//! Compatibility with `cactus-compute` 4.2.7's text completion checks.
//!
//! The logic follows `cactus/agent/grounding.py` from Cactus Compute (Apache-2.0).

use std::{collections::BTreeSet, sync::LazyLock};

use fancy_regex::Regex;
use serde_json::Value;

use super::{Completion, Tool};

const MONTHS: &str = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?";
const END_YEAR: &str = r"(?![0-9A-Za-z]|:\d|\s*[ap]\.?m\b)";

static YEARS: LazyLock<[Regex; 5]> = LazyLock::new(|| {
    [
        format!(r"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTHS}[\s,]+(\d{{1,4}}){END_YEAR}"),
        format!(r"\b{MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?\s*,?\s+(\d{{1,4}}){END_YEAR}"),
        format!(r"\b{MONTHS}[\s,]+(\d{{3,4}})(?![0-9A-Za-z])"),
        r"\byear\s+(\d{1,4})(?![0-9A-Za-z])".into(),
        r"(?<![0-9])(\d{4})(?=[-/]\d{1,2}[-/]\d{1,2}(?![0-9]))".into(),
    ]
    .map(|pattern| regex(&pattern))
});
static DATE_FACT: LazyLock<Regex> =
    LazyLock::new(|| regex(r"(?<![A-Za-z])date:|\d{4}-\d{2}-\d{2}"));
static RELATIVE: LazyLock<Regex> = LazyLock::new(|| {
    regex(
        r"(?i)\b(today|tonight|tomorrow|yesterday|next|this|coming|now|in \d+ (?:days?|weeks?|months?|years?)|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    )
});

fn regex(pattern: &str) -> Regex {
    Regex::new(pattern).expect("valid built-in Needle grounding pattern")
}

fn source_years(text: &str) -> BTreeSet<u16> {
    let lowercase = text.to_lowercase();
    YEARS
        .iter()
        .flat_map(|pattern| pattern.captures_iter(&lowercase))
        .flatten()
        .filter_map(|capture| capture.get(1)?.as_str().parse().ok())
        .collect()
}

pub(super) fn with_local_date(system: &str) -> String {
    with_date_fact(
        system,
        &chrono::Local::now()
            .format("date: %Y-%m-%d %a %H:%M")
            .to_string(),
    )
}

fn with_date_fact(system: &str, fact: &str) -> String {
    if DATE_FACT.is_match(system).unwrap_or(false) || system.trim_start().starts_with('{') {
        return system.to_owned();
    }
    if system.trim().is_empty() {
        return fact.to_owned();
    }
    format!("{fact}; {system}")
}

#[derive(Debug)]
pub(super) struct Grounding {
    tools: Vec<Tool>,
    system_years: BTreeSet<u16>,
    seen_years: BTreeSet<u16>,
}

impl Grounding {
    pub(super) fn new(tools: Vec<Tool>, system: &str) -> Self {
        Self {
            tools,
            system_years: source_years(system),
            seen_years: BTreeSet::new(),
        }
    }

    pub(super) fn annotate(&mut self, text: &str, response: &mut Completion) {
        self.seen_years.extend(source_years(text));
        if self.seen_years.is_empty() || response.calls().is_empty() {
            return;
        }
        let mut allowed = self.seen_years.clone();
        if RELATIVE.is_match(text).unwrap_or(false) {
            allowed.extend(&self.system_years);
        }
        let mut ungrounded = Vec::new();
        for call in response.calls() {
            let Some(tool) = self
                .tools
                .iter()
                .rev()
                .find(|tool| tool.name() == call.name())
            else {
                continue;
            };
            let schema = tool.parameters();
            let mut failures = BTreeSet::new();
            check_dates(
                call.arguments(),
                schema,
                schema,
                "",
                &allowed,
                &mut failures,
            );
            ungrounded.extend(
                failures
                    .into_iter()
                    .map(|path| format!("{}.{path}", call.name())),
            );
        }
        response.add_ungrounded(ungrounded);
    }

    pub(super) fn reset(&mut self) {
        self.seen_years.clear();
    }
}

fn resolve_ref<'a>(mut node: &'a Value, root: &'a Value) -> &'a Value {
    let mut seen = BTreeSet::new();
    while let Some(reference) = node.get("$ref").and_then(Value::as_str) {
        if !seen.insert(reference) {
            break;
        }
        let Some(target) = reference
            .strip_prefix('#')
            .and_then(|path| root.pointer(path))
        else {
            break;
        };
        node = target;
    }
    node
}

fn check_dates(
    value: &Value,
    node: &Value,
    root: &Value,
    path: &str,
    allowed: &BTreeSet<u16>,
    failures: &mut BTreeSet<String>,
) {
    let mut node = resolve_ref(node, root);
    let variants = node
        .get("anyOf")
        .and_then(Value::as_array)
        .filter(|variants| !variants.is_empty())
        .or_else(|| node.get("oneOf").and_then(Value::as_array));
    if let Some(variants) = variants {
        let mut concrete = variants
            .iter()
            .map(|node| resolve_ref(node, root))
            .filter(|node| node["type"] != "null");
        if let Some(only) = concrete.next()
            && concrete.next().is_none()
        {
            node = only;
        }
    }
    if let Some(text) = value.as_str()
        && matches!(node["format"].as_str(), Some("date" | "date-time"))
    {
        if text.as_bytes().get(4) == Some(&b'-')
            && let Some(year) = text.get(..4).and_then(|year| year.parse::<u16>().ok())
            && text.as_bytes()[..4].iter().all(u8::is_ascii_digit)
            && !allowed.contains(&year)
        {
            failures.insert(path.to_owned());
        }
    } else if let Some(object) = value.as_object() {
        for (key, item) in object {
            if let Some(property) = node["properties"].get(key) {
                let child = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                check_dates(item, property, root, &child, allowed, failures);
            }
        }
    } else if let Some(list) = value.as_array()
        && let Some(items) = node.get("items")
    {
        for (index, item) in list.iter().enumerate() {
            check_dates(
                item,
                items,
                root,
                &format!("{path}[{index}]"),
                allowed,
                failures,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn checks_nested_dates_and_retains_years_until_reset() {
        let schema = json!({"properties":{"rows":{"items":{"anyOf":[{"type":"null"},{"$ref":"#/$defs/date"}]}}},"$defs":{"date":{"format":"date"}}});
        let mut grounding = Grounding::new(
            vec![Tool::new("save", "save dates", schema)],
            "date: 2026-11-25",
        );
        let envelope = r#"{"type":"call","function_calls":[{"name":"save","arguments":{"rows":["2026-12-01"]}}]}"#;
        let mut output: Completion = envelope.parse().unwrap();
        grounding.annotate("March 4, 2025", &mut output);
        assert_eq!(output.validation().ungrounded(), ["save.rows[0]"]);

        let mut output: Completion = envelope.parse().unwrap();
        grounding.annotate("next Tuesday", &mut output);
        assert!(output.validation().ungrounded().is_empty());
        grounding.reset();
        grounding.annotate("No explicit year", &mut output);
        assert!(output.validation().ungrounded().is_empty());
    }

    #[test]
    fn preserves_explicit_dates_and_json_facts() {
        let fact = "date: 2026-11-25 Wed 09:00";
        assert_eq!(
            with_date_fact("locale: en", fact),
            format!("{fact}; locale: en")
        );
        assert_eq!(with_date_fact("date: 2025-01-01", fact), "date: 2025-01-01");
        assert_eq!(with_date_fact(" {}", fact), " {}");
    }
}

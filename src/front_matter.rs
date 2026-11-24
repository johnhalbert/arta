//! Shared boundary between Markdown and YAML in agent profiles and skill packages.

use anyhow::{Context, Result, bail};
use serde::de::DeserializeOwned;

/// Accept an optional byte-order mark and LF or CRLF fences. Deserialize only the header and
/// borrow the exact bytes following the closing fence as the body.
pub(crate) fn parse<T: DeserializeOwned>(source: &str) -> Result<(T, &str)> {
    let source = source.strip_prefix('\u{feff}').unwrap_or(source);
    let source = source
        .strip_prefix("---\n")
        .or_else(|| source.strip_prefix("---\r\n"))
        .context("expected YAML front matter beginning with '---'")?;
    let mut offset = 0;
    for line in source.split_inclusive('\n') {
        if line.trim_end_matches(['\r', '\n']) == "---" {
            let metadata =
                serde_saphyr::from_str(&source[..offset]).context("invalid YAML front matter")?;
            return Ok((metadata, &source[offset + line.len()..]));
        }
        offset += line.len();
    }
    bail!("missing closing '---' for YAML front matter")
}

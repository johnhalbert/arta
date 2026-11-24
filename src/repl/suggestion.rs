//! Completion uses the token prefix up to the cursor to select matches, then replaces the
//! entire token. Offsets are UTF-8 byte positions shared with the editor.

use std::ops::Range;

/// Parser-provided token and its user-facing explanation.
pub(crate) struct Candidate {
    /// Complete replacement token including its command or mention prefix.
    pub(crate) name: String,
    /// Display text accompanying the token in the menu.
    pub(crate) description: String,
}

/// Sorted matching candidates and the byte range of the complete token under the cursor.
pub(crate) struct Suggestions<'a> {
    /// UTF-8 byte span covering the whole token, including text to the right of the cursor.
    token: Range<usize>,
    /// Matching parser entries sorted by full token name.
    pub(crate) candidates: Vec<&'a Candidate>,
}

/// Applies the selected completion and returns the new cursor byte offset.
pub(crate) fn complete(
    candidates: &[&Candidate],
    input: &mut String,
    cursor: usize,
    selected: usize,
) -> Option<usize> {
    let suggestions = matches(candidates, input, cursor)?;
    let name = &suggestions.candidates.get(selected)?.name;
    let name_end = suggestions.token.start + name.len();
    let separator = input[suggestions.token.end..].chars().next();

    input.replace_range(suggestions.token, name);
    if separator.is_none() {
        input.push(' ');
    }
    Some(name_end + separator.map_or(1, char::len_utf8))
}

/// Finds commands that match the prefix before the cursor and the token to replace.
pub(crate) fn matches<'a>(
    candidates: &[&'a Candidate],
    input: &str,
    cursor: usize,
) -> Option<Suggestions<'a>> {
    let prefix = input[..cursor].rsplit(char::is_whitespace).next()?;
    if prefix.is_empty() {
        return None;
    }

    let mut candidates: Vec<_> = candidates
        .iter()
        .copied()
        .filter(|candidate| candidate.name.starts_with(prefix))
        .collect();
    if candidates.is_empty() {
        return None;
    }
    candidates.sort_unstable_by(|left, right| left.name.cmp(&right.name));

    let suffix = &input[cursor..];
    let end = cursor + suffix.find(char::is_whitespace).unwrap_or(suffix.len());
    Some(Suggestions {
        token: cursor - prefix.len()..end,
        candidates,
    })
}

use std::ops::Range;

pub(crate) struct Candidate {
    pub(crate) name: String,
    pub(crate) description: String,
}

pub(crate) struct Suggestions<'a> {
    token: Range<usize>,
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

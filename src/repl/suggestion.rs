use super::command::{self, Command};
use std::ops::Range;

pub(crate) struct Suggestions {
    token: Range<usize>,
    pub(crate) commands: Vec<&'static Command>,
}

/// Applies the selected completion and returns the new cursor byte offset.
pub(crate) fn complete(input: &mut String, cursor: usize, selected: usize) -> Option<usize> {
    let suggestions = matches(input, cursor)?;
    let name = suggestions.commands.get(selected)?.name;
    let name_end = suggestions.token.start + name.len();
    let separator = input[suggestions.token.end..].chars().next();

    input.replace_range(suggestions.token, name);
    if separator.is_none() {
        input.push(' ');
    }
    Some(name_end + separator.map_or(1, char::len_utf8))
}

/// Finds commands that match the prefix before the cursor and the token to replace.
pub(crate) fn matches(input: &str, cursor: usize) -> Option<Suggestions> {
    let prefix = input[..cursor].rsplit(char::is_whitespace).next()?;
    if !prefix.starts_with('/') {
        return None;
    }

    let mut commands: Vec<_> = command::registered()
        .filter(|command| command.name.starts_with(prefix))
        .collect();
    if commands.is_empty() {
        return None;
    }
    commands.sort_unstable_by_key(|command| command.name);

    let suffix = &input[cursor..];
    let end = cursor + suffix.find(char::is_whitespace).unwrap_or(suffix.len());
    Some(Suggestions {
        token: cursor - prefix.len()..end,
        commands,
    })
}

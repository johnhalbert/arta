use super::command::{self, Command};

/// Extracts registered command tokens in source order and returns the remaining prompt.
pub(super) fn parse(input: String) -> (String, Vec<&'static Command>) {
    let mut commands = Vec::new();
    let mut prompt = String::new();
    let mut offset = 0;
    let mut copied = 0;
    for segment in input.split_inclusive(char::is_whitespace) {
        let start = offset;
        offset += segment.len();
        let token = segment.trim_end_matches(char::is_whitespace);
        if token.starts_with('/')
            && let Some(command) = command::registered().find(|command| command.name == token)
        {
            commands.push(command);
            prompt.push_str(&input[copied..start]);
            copied = offset;
        }
    }
    if commands.is_empty() {
        return (input, commands);
    }
    prompt.push_str(&input[copied..]);
    (prompt.trim().into(), commands)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_preserves_text_and_lists_commands_in_source_order() {
        let ordinary = "  /resetting path/exit /exit/path /reset. /unknown  ";
        let (prompt, commands) = parse(ordinary.into());
        assert_eq!(prompt, ordinary);
        assert!(commands.is_empty());

        let (prompt, commands) = parse("/reset café /exit /reset 👋 /unknown".into());
        assert_eq!(prompt, "café 👋 /unknown");
        let names: Vec<_> = commands.iter().map(|command| command.name).collect();
        assert_eq!(names, ["/reset", "/exit", "/reset"]);
    }
}

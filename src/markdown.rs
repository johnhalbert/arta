use crossterm::terminal;
use ratatui::{backend::IntoCrossterm, text::Line};
use std::{
    borrow::Cow,
    fmt::Write as _,
    io::{self, IsTerminal},
};
use tui_markdown::{Options, StyleSheet, from_str_with_options};

#[derive(Clone)]
struct ReplyStyles;

impl StyleSheet for ReplyStyles {
    fn heading_marker(&self, _: u8) -> &str {
        ""
    }
}

pub(super) fn render(source: &str, width: u16) -> Vec<Line<'_>> {
    let options = Options::new(ReplyStyles).table_width(width);
    let mut text = from_str_with_options(source, &options);
    for span in text.lines.iter_mut().flat_map(|line| &mut line.spans) {
        // Clean the rendered spans as well as literal text, including decoded entities.
        if let Cow::Owned(clean) = clean_terminal_text(&span.content) {
            span.content = Cow::Owned(clean);
        }
    }
    text.lines
}

pub(super) fn clean_terminal_text(text: &str) -> Cow<'_, str> {
    if !text.chars().any(char::is_control) {
        return Cow::Borrowed(text);
    }
    let mut clean = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '\t' => clean.push_str("    "),
            c if !c.is_control() => clean.push(c),
            _ => {}
        }
    }
    Cow::Owned(clean)
}

#[derive(Clone, Copy)]
pub(super) struct TerminalOutput {
    pub(super) width: u16,
    pub(super) ansi: bool,
}

impl TerminalOutput {
    pub(super) fn detect() -> Option<Self> {
        io::stdout().is_terminal().then(|| {
            #[cfg(windows)]
            let ansi = crossterm::ansi_support::supports_ansi();
            #[cfg(not(windows))]
            let ansi = std::env::var_os("TERM").is_none_or(|term| term != "dumb");

            Self {
                width: terminal::size().map_or(80, |(width, _)| width),
                ansi,
            }
        })
    }

    pub(super) fn format(self, source: &str) -> String {
        let mut output = String::new();
        if source.is_empty() {
            return output;
        }
        for line in render(source, self.width) {
            for span in line.spans {
                if self.ansi {
                    let style = line.style.patch(span.style).into_crossterm();
                    write!(output, "{}", style.apply(span.content)).expect("writing to a String");
                } else {
                    output.push_str(&span.content);
                }
            }
            output.push('\n');
        }
        output
    }
}

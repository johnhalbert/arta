use super::{
    markdown,
    repl::{Repl, parser::Parser, suggestion},
};
use anyhow::Result;
use arta::{provider::Turn, stream::Frame as ReplyFrame};
use crossterm::{
    cursor::Show,
    event::{
        DisableBracketedPaste, EnableBracketedPaste, Event, EventStream, KeyCode, KeyEventKind,
        KeyModifiers,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use futures_util::StreamExt;
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Layout, Rect},
    style::Stylize,
    text::Line,
    widgets::{Block, BorderType, Clear, Paragraph, Wrap},
};
use std::{
    io, mem,
    time::{Duration, Instant},
};
use tokio::time::{MissedTickBehavior, interval};
use unicode_segmentation::UnicodeSegmentation;

const BUSY_FRAMES: [&str; 6] = [".", "..", "...", " ..", "  .", ""];

pub(super) async fn run(repl: &mut Repl<'_>) -> Result<()> {
    let mut screen = Screen::enter()?;
    let mut events = EventStream::new();
    let mut tui = Tui::new(repl.parser());

    loop {
        screen.draw(|frame| tui.render(frame, repl.history(), None))?;
        let Some(event) = events.next().await.transpose()? else {
            break;
        };
        match tui.handle(event) {
            InputEvent::Quit => break,
            InputEvent::Continue => {}
            InputEvent::Submit => {
                let submission = repl.submit(tui.input.take());
                if submission.exit {
                    break;
                }
                if submission.history_cleared {
                    tui.failure = None;
                    tui.scroll = None;
                }
                if let Some(prompt) = submission.prompt
                    && !tui.respond(repl, prompt, &mut events, &mut screen).await?
                {
                    break;
                }
            }
        }
    }
    Ok(())
}

type Console = Terminal<CrosstermBackend<io::Stdout>>;

/// Restores the shell's screen and input modes on exit, including on errors.
struct Screen(Console);

impl Screen {
    fn enter() -> io::Result<Self> {
        let terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
        enable_raw_mode()?;
        let mut screen = Self(terminal);
        execute!(screen.0.backend_mut(), EnterAlternateScreen)?;
        if let Err(error) = execute!(screen.0.backend_mut(), EnableBracketedPaste)
            && error.kind() != io::ErrorKind::Unsupported
        {
            return Err(error);
        }
        screen.0.clear()?;
        Ok(screen)
    }

    fn draw(&mut self, render: impl FnOnce(&mut Frame<'_>)) -> io::Result<()> {
        self.0.draw(render)?;
        Ok(())
    }
}

impl Drop for Screen {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(
            self.0.backend_mut(),
            DisableBracketedPaste,
            LeaveAlternateScreen,
            Show
        );
    }
}

enum InputEvent {
    Continue,
    Quit,
    Submit,
}

#[derive(Default)]
struct Tui<'a> {
    input: Editor<'a>,
    /// A fixed top row while reading back. None follows the newest text.
    scroll: Option<u16>,
    latest: u16,
    page: u16,
    busy_frame: usize,
    failure: Option<(Turn, String)>,
}

impl<'a> Tui<'a> {
    fn new(parser: &'a Parser) -> Self {
        Self {
            input: Editor {
                candidates: parser.suggestions().collect(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn handle(&mut self, event: Event) -> InputEvent {
        match event {
            Event::Paste(text) => {
                let text: String = text
                    .chars()
                    .map(|character| {
                        if character.is_control() {
                            ' '
                        } else {
                            character
                        }
                    })
                    .collect();
                self.input.insert(&text);
                self.input.selected = 0;
            }
            Event::Key(key) if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) => {
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && !key.modifiers.contains(KeyModifiers::ALT)
                {
                    match key.code {
                        KeyCode::Char('c') => return InputEvent::Quit,
                        KeyCode::Char('d') if self.input.text.is_empty() => {
                            return InputEvent::Quit;
                        }
                        KeyCode::End => self.scroll = None,
                        _ => {}
                    }
                } else {
                    match key.code {
                        KeyCode::Enter => return InputEvent::Submit,
                        KeyCode::PageUp => {
                            self.scroll =
                                Some(self.scroll.unwrap_or(self.latest).saturating_sub(self.page));
                        }
                        KeyCode::PageDown => {
                            self.scroll = self.scroll.and_then(|row| {
                                let next = row.saturating_add(self.page);
                                (next < self.latest).then_some(next)
                            });
                        }
                        code => self.input.edit(code),
                    }
                }
            }
            _ => {}
        }
        InputEvent::Continue
    }

    async fn respond(
        &mut self,
        repl: &mut Repl<'_>,
        prompt: String,
        events: &mut EventStream,
        screen: &mut Screen,
    ) -> Result<bool> {
        self.scroll = None;
        self.failure = None;
        self.busy_frame = 0;
        let mut turn = Turn {
            prompt,
            reply: String::new(),
        };
        screen.draw(|frame| self.render(frame, repl.history(), Some(&turn)))?;
        let mut replies = repl.reply(&turn.prompt);
        let started = Instant::now();
        let mut refresh = interval(Duration::from_millis(33));
        refresh.set_missed_tick_behavior(MissedTickBehavior::Skip);
        let mut dirty = false;
        let error = loop {
            tokio::select! {
                event = events.next() => {
                    let Some(event) = event.transpose()? else { return Ok(false) };
                    if matches!(self.handle(event), InputEvent::Quit) {
                        return Ok(false);
                    }
                    screen.draw(|frame| self.render(frame, repl.history(), Some(&turn)))?;
                    dirty = false;
                }
                frame = replies.next() => match frame {
                    Some(Ok(ReplyFrame::Chunk(text))) => {
                        turn.reply.push_str(&text);
                        dirty = true;
                    }
                    Some(Ok(ReplyFrame::Begin)) => {}
                    Some(Ok(ReplyFrame::End)) | None => break None,
                    Some(Err(error)) => break Some(format!("{error:#}")),
                },
                _ = refresh.tick(), if dirty || turn.reply.is_empty() => {
                    let waiting = turn.reply.is_empty();
                    let next = (started.elapsed().as_millis() / 120 % BUSY_FRAMES.len() as u128) as usize;
                    if dirty || (waiting && next != self.busy_frame) {
                        self.busy_frame = next;
                        screen.draw(|frame| self.render(frame, repl.history(), Some(&turn)))?;
                        dirty = false;
                    }
                }
            }
        };
        drop(replies);
        if let Some(error) = error {
            self.failure = Some((turn, error));
        } else {
            repl.complete(turn);
        }
        Ok(true)
    }

    fn render(&mut self, frame: &mut Frame<'_>, history: &[Turn], pending: Option<&Turn>) {
        let area = frame.area();
        let [upper, chat] =
            Layout::vertical([Constraint::Length(area.height / 2), Constraint::Min(0)]).areas(area);
        let [left, right] =
            Layout::horizontal([Constraint::Length(upper.width / 2), Constraint::Min(0)])
                .areas(upper);
        for pane in [left, right] {
            frame.render_widget(Block::bordered().border_type(BorderType::Rounded), pane);
        }

        let bordered = chat.height >= 4 && chat.width >= 3;
        let input_height = if bordered { 3 } else { 1 };
        let [transcript, mut prompt] =
            Layout::vertical([Constraint::Min(0), Constraint::Length(input_height)]).areas(chat);

        let mut lines = Vec::new();
        for turn in history {
            append_turn(&mut lines, turn, transcript.width);
        }
        if let Some(turn) = pending {
            append_turn(&mut lines, turn, transcript.width);
            if turn.reply.is_empty() {
                lines.push(Line::raw(BUSY_FRAMES[self.busy_frame]).green());
            }
        }
        if let Some((turn, error)) = &self.failure {
            append_turn(&mut lines, turn, transcript.width);
            lines.push(Line::raw("Error").red().bold());
            lines.extend(display_lines(error).map(|line| line.red()));
        }
        self.page = transcript.height.saturating_sub(1).max(1);
        (self.latest, self.scroll) = render_transcript(frame, transcript, lines, self.scroll);

        let suggestions = self.input.suggestions().filter(|_| !transcript.is_empty());
        let menu_visible = suggestions.is_some();
        if let Some(suggestions) = suggestions {
            let height = suggestions
                .candidates
                .len()
                .min(usize::from(transcript.height));
            let first = self.input.selected.saturating_sub(height - 1);
            let options: Vec<_> = suggestions
                .candidates
                .iter()
                .enumerate()
                .skip(first)
                .take(height)
                .map(|(index, candidate)| {
                    let line = Line::from(vec![
                        candidate.name.as_str().cyan().bold(),
                        "  ".into(),
                        markdown::clean_terminal_text(&candidate.description).into(),
                    ]);
                    if index == self.input.selected {
                        line.reversed()
                    } else {
                        line
                    }
                })
                .collect();
            let width = options
                .iter()
                .map(Line::width)
                .max()
                .unwrap_or_default()
                .min(usize::from(transcript.width)) as u16;
            let menu = Rect::new(
                transcript.x,
                transcript.bottom() - height as u16,
                width,
                height as u16,
            );
            frame.render_widget(Clear, menu);
            frame.render_widget(Paragraph::new(options), menu);
        }

        if bordered {
            let title = if pending.is_some() {
                " Generating… "
            } else {
                " Prompt "
            };
            let controls = if menu_visible {
                " Tab complete · ↑/↓ select "
            } else if pending.is_some() {
                " PgUp/PgDn scroll · Ctrl+C exit "
            } else {
                " Enter send · PgUp/PgDn scroll · / $ @ "
            };
            let block = Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(ratatui::style::Style::new().cyan())
                .title(title.bold())
                .title_bottom(controls.dim());
            let inner = block.inner(prompt);
            frame.render_widget(block, prompt);
            prompt = inner;
        }
        self.input.render(frame, prompt);
    }
}

fn render_transcript(
    frame: &mut Frame<'_>,
    mut area: Rect,
    lines: Vec<Line<'_>>,
    scroll: Option<u16>,
) -> (u16, Option<u16>) {
    if area.is_empty() {
        return (0, scroll);
    }
    let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
    let count = paragraph.line_count(area.width);
    let visible = usize::from(area.height);
    let latest = u16::try_from(count.saturating_sub(visible)).unwrap_or(u16::MAX);
    let offset = scroll.unwrap_or(latest).min(latest);
    let padding = visible.saturating_sub(count) as u16;
    area.y += padding;
    area.height -= padding;
    frame.render_widget(paragraph.scroll((offset, 0)), area);
    (latest, (offset < latest).then_some(offset))
}

fn append_turn<'a>(lines: &mut Vec<Line<'a>>, turn: &'a Turn, width: u16) {
    if !lines.is_empty() {
        lines.push(Line::default());
    }
    lines.push(Line::raw("You").cyan().bold());
    lines.extend(display_lines(&turn.prompt));
    lines.push(Line::raw("Arta").green().bold());
    lines.extend(markdown::render(&turn.reply, width));
}

fn display_lines(text: &str) -> impl Iterator<Item = Line<'_>> {
    text.lines()
        .map(|line| Line::raw(markdown::clean_terminal_text(line)))
}

#[derive(Default)]
struct Editor<'a> {
    text: String,
    cursor: usize,
    selected: usize,
    candidates: Vec<&'a suggestion::Candidate>,
}

impl<'a> Editor<'a> {
    fn take(&mut self) -> String {
        self.cursor = 0;
        self.selected = 0;
        mem::take(&mut self.text)
    }

    fn insert(&mut self, text: &str) {
        self.text.insert_str(self.cursor, text);
        self.cursor = self.boundary_at_or_after(self.cursor + text.len());
    }

    fn boundary_at_or_after(&self, offset: usize) -> usize {
        self.text
            .grapheme_indices(true)
            .map(|(index, _)| index)
            .find(|&index| index >= offset)
            .unwrap_or(self.text.len())
    }

    fn suggestions(&self) -> Option<suggestion::Suggestions<'a>> {
        suggestion::matches(&self.candidates, &self.text, self.cursor)
    }

    fn edit(&mut self, code: KeyCode) {
        match code {
            KeyCode::Up | KeyCode::Down => {
                if let Some(suggestions) = self.suggestions() {
                    self.selected = if code == KeyCode::Up {
                        self.selected.saturating_sub(1)
                    } else {
                        (self.selected + 1).min(suggestions.candidates.len() - 1)
                    };
                }
                return;
            }
            KeyCode::Tab => {
                if let Some(cursor) = suggestion::complete(
                    &self.candidates,
                    &mut self.text,
                    self.cursor,
                    self.selected,
                ) {
                    self.cursor = self.boundary_at_or_after(cursor);
                }
            }
            KeyCode::Char(c) if !c.is_control() => self.insert(c.encode_utf8(&mut [0; 4])),
            KeyCode::Left | KeyCode::Backspace => {
                let previous = self.text[..self.cursor]
                    .grapheme_indices(true)
                    .next_back()
                    .map_or(0, |(index, _)| index);
                if code == KeyCode::Backspace {
                    self.text.drain(previous..self.cursor);
                }
                self.cursor = self.boundary_at_or_after(previous);
            }
            KeyCode::Right | KeyCode::Delete => {
                let next = self.boundary_at_or_after(self.cursor + 1);
                if code == KeyCode::Delete {
                    self.text.drain(self.cursor..next);
                    self.cursor = self.boundary_at_or_after(self.cursor);
                } else {
                    self.cursor = next;
                }
            }
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.text.len(),
            _ => {}
        }
        self.selected = 0;
    }

    fn render(&self, frame: &mut Frame<'_>, area: Rect) {
        if area.is_empty() {
            return;
        }
        let mut offset = self.cursor;
        let mut column = 0;
        for (index, grapheme) in self.text[..self.cursor].grapheme_indices(true).rev() {
            let width = Line::raw(grapheme).width();
            if column + width >= usize::from(area.width) {
                break;
            }
            offset = index;
            column += width;
        }
        frame.render_widget(Line::raw(&self.text[offset..]), area);
        frame.set_cursor_position((area.x + column as u16, area.y));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyEvent;
    use ratatui::backend::TestBackend;

    #[test]
    fn transcript_grows_up_and_can_page_back() -> Result<()> {
        let mut terminal = Terminal::new(TestBackend::new(60, 24))?;
        let mut tui = Tui::default();
        let history = [Turn {
            prompt: "First".into(),
            reply: "Alpha".into(),
        }];
        terminal.draw(|frame| tui.render(frame, &history, None))?;
        let buffer = terminal.backend().buffer();
        assert_eq!(buffer[(0, 0)].symbol(), "╭");
        assert_eq!(buffer[(29, 0)].symbol(), "╮");
        assert_eq!(buffer[(30, 0)].symbol(), "╭");
        assert_eq!(buffer[(59, 11)].symbol(), "╯");
        assert_eq!(buffer[(1, 10)].symbol(), " ");
        assert_eq!(buffer[(31, 10)].symbol(), " ");
        assert_eq!(buffer[(0, 16)].symbol(), " ");
        assert_eq!(buffer[(0, 17)].symbol(), "Y");
        assert_eq!(buffer[(0, 20)].symbol(), "A");
        assert_eq!(buffer[(0, 21)].symbol(), "╭");

        let pending = Turn {
            prompt: "Second".into(),
            reply: "One  \nTwo  \nThree  \nFour  \nFive  \nLast".into(),
        };
        terminal.draw(|frame| tui.render(frame, &history, Some(&pending)))?;
        assert_eq!(terminal.backend().buffer()[(0, 20)].symbol(), "L");

        tui.handle(Event::Key(KeyEvent::new(
            KeyCode::PageUp,
            KeyModifiers::NONE,
        )));
        terminal.draw(|frame| tui.render(frame, &history, Some(&pending)))?;
        assert_eq!(terminal.backend().buffer()[(0, 13)].symbol(), "F");

        tui.handle(Event::Key(KeyEvent::new(
            KeyCode::PageDown,
            KeyModifiers::NONE,
        )));
        terminal.draw(|frame| tui.render(frame, &history, Some(&pending)))?;
        assert_eq!(terminal.backend().buffer()[(0, 20)].symbol(), "L");

        terminal.backend_mut().resize(30, 3);
        terminal.draw(|frame| tui.render(frame, &history, Some(&pending)))?;
        assert_eq!(terminal.backend().buffer()[(0, 1)].symbol(), "L");
        Ok(())
    }

    #[test]
    fn ellipsis_frames_are_replaced_by_reply_text_under_arta() -> Result<()> {
        let mut terminal = Terminal::new(TestBackend::new(60, 24))?;
        let mut tui = Tui::default();
        let mut pending = Turn {
            prompt: "Hello".into(),
            reply: String::new(),
        };
        for (index, expected) in [".  ", ".. ", "...", " ..", "  .", "   "]
            .into_iter()
            .enumerate()
        {
            tui.busy_frame = index;
            terminal.draw(|frame| tui.render(frame, &[], Some(&pending)))?;
            let buffer = terminal.backend().buffer();
            assert_eq!(buffer[(0, 19)].symbol(), "A");
            let indicator: String = (0..3).map(|column| buffer[(column, 20)].symbol()).collect();
            assert_eq!(indicator, expected);
        }

        pending.reply.push_str("**Hi**");
        terminal.draw(|frame| tui.render(frame, &[], Some(&pending)))?;
        let buffer = terminal.backend().buffer();
        assert_eq!(buffer[(0, 19)].symbol(), "A");
        assert_eq!(buffer[(0, 20)].symbol(), "H");
        assert!(
            buffer[(0, 20)]
                .modifier
                .contains(ratatui::style::Modifier::BOLD)
        );
        assert_eq!(buffer[(1, 20)].symbol(), "i");
        Ok(())
    }

    #[test]
    fn editor_moves_and_deletes_whole_graphemes() {
        let mut editor = Editor::default();
        editor.insert("a👩‍💻e\u{301}z");
        editor.edit(KeyCode::Left);
        editor.edit(KeyCode::Backspace);
        assert_eq!(editor.text, "a👩‍💻z");
        editor.edit(KeyCode::Left);
        editor.edit(KeyCode::Delete);
        assert_eq!(editor.take(), "az");
        assert_eq!(editor.cursor, 0);
    }

    #[test]
    fn slash_suggestions_render_and_complete_from_the_keyboard() -> Result<()> {
        let mut parser = Parser::default();
        crate::repl::command::register(&mut parser);
        let mut terminal = Terminal::new(TestBackend::new(60, 24))?;
        let mut tui = Tui::new(&parser);
        let history = [Turn {
            prompt: "Earlier prompt".into(),
            reply: "First  \n**xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx**  \nLast".into(),
        }];
        terminal.draw(|frame| tui.render(frame, &history, None))?;
        assert_eq!(terminal.backend().buffer()[(0, 18)].symbol(), "F");
        assert_eq!(terminal.backend().buffer()[(25, 19)].symbol(), "x");
        let page = tui.page;

        let key = |code| Event::Key(KeyEvent::new(code, KeyModifiers::NONE));
        tui.handle(key(KeyCode::Char('/')));
        terminal.draw(|frame| tui.render(frame, &history, None))?;
        let buffer = terminal.backend().buffer();
        assert_eq!(tui.page, page);
        assert_eq!(buffer[(0, 18)].symbol(), "F");
        assert_eq!(buffer[(25, 19)].symbol(), " ");
        assert_eq!(buffer[(40, 19)].symbol(), "x");
        let first: String = (0..5).map(|column| buffer[(column, 19)].symbol()).collect();
        let second: String = (0..6).map(|column| buffer[(column, 20)].symbol()).collect();
        assert_eq!(first, "/exit");
        assert_eq!(second, "/reset");
        assert!(
            buffer[(0, 19)]
                .modifier
                .contains(ratatui::style::Modifier::REVERSED)
        );

        tui.handle(key(KeyCode::Down));
        tui.handle(key(KeyCode::Tab));
        assert_eq!(tui.input.text, "/reset ");
        assert_eq!(tui.input.cursor, tui.input.text.len());
        assert!(tui.input.suggestions().is_none());
        terminal.draw(|frame| tui.render(frame, &history, None))?;
        assert_eq!(terminal.backend().buffer()[(25, 19)].symbol(), "x");

        tui.handle(Event::Paste("café /exzt\u{2003}later".into()));
        for _ in 0..8 {
            tui.handle(key(KeyCode::Left));
        }
        assert!(tui.input.text[..tui.input.cursor].ends_with("/ex"));
        tui.handle(key(KeyCode::Tab));
        assert_eq!(tui.input.text, "/reset café /exit\u{2003}later");
        assert_eq!(&tui.input.text[tui.input.cursor..], "later");
        Ok(())
    }
}

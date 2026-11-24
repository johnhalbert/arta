//! Fullscreen event loop. Terminal input, rendered model prompts, and asynchronous reply
//! frames update UI state independently. The REPL owns ordinary chat history. Workflow
//! transcripts and incomplete replies stay in the display layer.

mod workflow;

use self::workflow::{WorkflowPane, WorkflowTranscript};
use super::{
    markdown,
    repl::{Repl, parser::Parser, suggestion},
};
use anyhow::Result;
use arta::{
    config::Config,
    provider::Turn,
    stream::Frame as ReplyFrame,
    workflow::{Workflow, WorkflowEvent},
};
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
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{
    sync::mpsc,
    time::{MissedTickBehavior, interval},
};
use unicode_segmentation::UnicodeSegmentation;

/// Activity indicator indexed by elapsed time while a visit awaits output.
const BUSY_FRAMES: [&str; 6] = [".", "..", "...", " ..", "  .", ""];

/// Enter the alternate screen, render the idle editor, and dispatch submissions. A pending
/// workflow consumes one prompt. Ordinary chat and workflows each have a response loop that
/// continues polling the keyboard.
pub(super) async fn run(repl: &mut Repl<'_>, config: &Config) -> Result<()> {
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
                    tui.model_inputs = ModelInputs::default();
                    tui.workflow = None;
                    tui.workflow_history.clear();
                }
                if !submission.model_requests.is_empty() {
                    tui.failure = None;
                    tui.scroll = None;
                }
                if submission.model_requests.iter().any(Option::is_some) {
                    tui.model_status = Some(Ok("Switching model…".into()));
                    screen.draw(|frame| tui.render(frame, repl.history(), None))?;
                }
                tui.model_status = match repl.apply_model_requests(submission.model_requests).await
                {
                    Ok(status) => status.map(Ok),
                    Err(error) => {
                        tui.model_status = Some(Err(format!("{error:#}")));
                        continue;
                    }
                };
                if let Some(request) = submission.workflow_request {
                    match repl.select_workflow(config, request.as_deref(), true) {
                        Ok(workflow) => tui.workflow = Some(WorkflowPane::new(workflow)),
                        Err(error) => {
                            tui.model_status = Some(Err(format!("{error:#}")));
                            continue;
                        }
                    }
                }
                if let Some(prompt) = submission.prompt {
                    let keep_running = match repl.take_workflow() {
                        Some(workflow) => {
                            tui.respond_workflow(repl, workflow, prompt, &mut events, &mut screen)
                                .await?
                        }
                        None => tui.respond(repl, prompt, &mut events, &mut screen).await?,
                    };
                    if !keep_running {
                        break;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Ratatui terminal drawing through Crossterm on standard output.
type Console = Terminal<CrosstermBackend<io::Stdout>>;

/// Restores the shell's screen and input modes on exit, including on errors.
struct Screen(Console);

impl Screen {
    /// Enable raw mode and enter the alternate screen, tolerating unsupported bracketed
    /// paste. The guard restores the terminal on subsequent errors.
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

    /// Draw one frame against the guarded terminal, propagating backend errors.
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

/// Effect of one terminal event on the outer submission loop.
enum InputEvent {
    /// Repaint or keep waiting after handling local editor and navigation state.
    Continue,
    /// Exit the interface, dropping any currently polled reply stream.
    Quit,
    /// Process the current editor buffer in the idle submission loop.
    Submit,
}

/// Editor, pane scroll state, in-flight feedback, and display-only workflow transcripts for
/// one fullscreen session.
#[derive(Default)]
struct Tui<'a> {
    /// Editable text, grapheme-aware cursor, and current completion selection.
    input: Editor<'a>,
    /// A fixed top row while reading back. None follows the newest text.
    scroll: Option<u16>,
    /// First wrapped row of the transcript's bottommost viewport.
    latest: u16,
    /// Chat scroll increment in wrapped rows, leaving one row of overlap where possible.
    page: u16,
    /// Rendered input captures for the current turn or workflow node.
    model_inputs: ModelInputs,
    /// Current position in the waiting indicator animation.
    busy_frame: usize,
    /// Partial ordinary turn and error retained for display after failed inference.
    failure: Option<(Turn, String)>,
    /// Transient command status or error shown in the transcript.
    model_status: Option<Result<String, String>>,
    /// Outline for the selected or most recently run workflow.
    workflow: Option<WorkflowPane>,
    /// Display-only workflow transcripts anchored among ordinary chat turns.
    workflow_history: Vec<WorkflowTranscript>,
}

impl<'a> Tui<'a> {
    /// Build the editor's completion list from the parser used for submission.
    fn new(parser: &'a Parser) -> Self {
        Self {
            input: Editor {
                candidates: parser.suggestions().collect(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Route navigation and editing keys, replace pasted control characters with spaces, and
    /// signal submission or exit to the caller.
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
                        KeyCode::PageUp | KeyCode::PageDown => {
                            if let Some(workflow) = &mut self.workflow {
                                workflow.page(key.code == KeyCode::PageDown);
                            }
                        }
                        _ => {}
                    }
                } else {
                    match key.code {
                        KeyCode::Enter => return InputEvent::Submit,
                        KeyCode::Left if key.modifiers == KeyModifiers::ALT => {
                            self.model_inputs
                                .select(self.model_inputs.selected.saturating_sub(1));
                        }
                        KeyCode::Right if key.modifiers == KeyModifiers::ALT => {
                            self.model_inputs.select(self.model_inputs.selected + 1);
                        }
                        KeyCode::PageUp if key.modifiers.contains(KeyModifiers::SHIFT) => {
                            let pane = &mut self.model_inputs;
                            pane.scroll = pane.scroll.saturating_sub(pane.page);
                        }
                        KeyCode::PageDown if key.modifiers.contains(KeyModifiers::SHIFT) => {
                            let pane = &mut self.model_inputs;
                            pane.scroll = pane.scroll.saturating_add(pane.page).min(pane.last_row);
                        }
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

    /// Drive the run alongside keyboard input and refresh ticks. Apply status events
    /// immediately and coalesce output redraws. Preserve an error with the partial workflow
    /// transcript.
    async fn respond_workflow(
        &mut self,
        repl: &mut Repl<'_>,
        workflow: Workflow,
        prompt: String,
        events: &mut EventStream,
        screen: &mut Screen,
    ) -> Result<bool> {
        self.scroll = None;
        self.failure = None;
        self.busy_frame = 0;
        self.model_inputs = ModelInputs::default();
        self.workflow.insert(WorkflowPane::new(&workflow)).start();
        let (history, mut replies) = repl.workflow_events(&workflow, &prompt, true);
        self.workflow_history
            .push(WorkflowTranscript::new(history.len(), prompt.clone()));
        screen.draw(|frame| self.render(frame, history, None))?;
        let started = Instant::now();
        let mut refresh = interval(Duration::from_millis(33));
        refresh.set_missed_tick_behavior(MissedTickBehavior::Skip);
        let mut dirty = false;
        let error = loop {
            tokio::select! {
                event = events.next() => {
                    let Some(event) = event.transpose()? else { return Ok(false) };
                    if matches!(self.handle(event), InputEvent::Quit) { return Ok(false); }
                    screen.draw(|frame| self.render(frame, history, None))?;
                    dirty = false;
                }
                event = replies.next() => match event {
                    Some(Ok(event)) => {
                        let output = matches!(event, WorkflowEvent::Output(_));
                        self.workflow_event(event);
                        if output {
                            dirty = true;
                        } else {
                            screen.draw(|frame| self.render(frame, history, None))?;
                            dirty = false;
                        }
                    }
                    Some(Err(error)) => break Some(format!("{error:#}")),
                    None => break None,
                },
                _ = refresh.tick() => {
                    let next = (started.elapsed().as_millis() / 120 % BUSY_FRAMES.len() as u128) as usize;
                    let waiting = self.workflow_history.last().is_some_and(|turn| turn.waiting);
                    if dirty || (waiting && next != self.busy_frame) {
                        self.busy_frame = next;
                        screen.draw(|frame| self.render(frame, history, None))?;
                        dirty = false;
                    }
                }
            }
        };
        if let Some(workflow) = &mut self.workflow {
            workflow.finish(error.is_none());
        }
        if let Some(transcript) = self.workflow_history.last_mut() {
            transcript.waiting = false;
            transcript.error = error;
        }
        Ok(true)
    }

    /// Project a node event into the outline and transcript. Reset captured model inputs at
    /// each node start and record control calls beside that node's output.
    fn workflow_event(&mut self, event: WorkflowEvent) {
        let (Some(workflow), Some(transcript)) =
            (&mut self.workflow, self.workflow_history.last_mut())
        else {
            return;
        };
        match event {
            WorkflowEvent::NodeStarted { id, agent, loops } => {
                transcript.node(super::workflow_output::node_label(&id, &agent, &loops));
                workflow.node(&id, loops);
                self.model_inputs = ModelInputs::default();
            }
            WorkflowEvent::ModelLoading { previous, next } => {
                let status = match previous {
                    Some(previous) if previous == next => "Reloading",
                    Some(_) => "Switching to",
                    None => "Loading",
                };
                workflow.model = Some(format!("{status} {next}"));
            }
            WorkflowEvent::ModelLoaded { model } => workflow.model = Some(format!("Model {model}")),
            WorkflowEvent::ModelInput(prompt) => self.model_inputs.push(prompt),
            WorkflowEvent::LoopToolCall {
                iteration,
                name,
                arguments,
                error,
            } => {
                transcript.tool_call(super::workflow_output::loop_tool_label(
                    &iteration,
                    &name,
                    arguments,
                    error.as_deref(),
                ));
            }
            WorkflowEvent::Output(text) => transcript.output(&text),
            WorkflowEvent::NodeFinished => transcript.waiting = false,
        }
    }

    /// Poll reply frames, observed prompts, input events, and a repaint timer together. Drain
    /// queued prompt captures on completion and commit only a successful turn to the REPL.
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
        self.model_inputs = ModelInputs::default();
        let mut turn = Turn {
            prompt,
            reply: String::new(),
        };
        screen.draw(|frame| self.render(frame, repl.history(), Some(&turn)))?;
        let (sender, mut captured) = mpsc::unbounded_channel();
        let observer = Arc::new(move |prompt: &str| {
            let _ = sender.send(prompt.to_owned());
        });
        let mut replies = repl.reply(&turn.prompt, Some(observer));
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
                Some(prompt) = captured.recv() => {
                    self.model_inputs.push(prompt);
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
        // A finished reply can win the select while a captured input is still queued.
        while let Ok(prompt) = captured.try_recv() {
            self.model_inputs.push(prompt);
        }
        if let Some(error) = error {
            self.failure = Some((turn, error));
        } else {
            repl.complete(turn);
        }
        Ok(true)
    }

    /// Interleave display-only workflow transcripts at their chat history offsets, then lay
    /// out the input inspector, workflow outline, transcript, editor, and completions.
    fn render(&mut self, frame: &mut Frame<'_>, history: &[Turn], pending: Option<&Turn>) {
        let area = frame.area();
        let [upper, chat] =
            Layout::vertical([Constraint::Length(area.height / 2), Constraint::Min(0)]).areas(area);
        let [left, right] =
            Layout::horizontal([Constraint::Length(upper.width / 2), Constraint::Min(0)])
                .areas(upper);
        if let Some(workflow) = &mut self.workflow {
            workflow.render(frame, left);
        } else {
            frame.render_widget(Block::bordered().border_type(BorderType::Rounded), left);
        }
        self.model_inputs.render(frame, right);

        let bordered = chat.height >= 4 && chat.width >= 3;
        let input_height = if bordered { 3 } else { 1 };
        let [transcript, mut prompt] =
            Layout::vertical([Constraint::Min(0), Constraint::Length(input_height)]).areas(chat);

        let mut lines = Vec::new();
        let mut workflows = self.workflow_history.iter().peekable();
        for index in 0..=history.len() {
            while let Some(workflow) = workflows.next_if(|workflow| workflow.history_index == index)
            {
                workflow.append(&mut lines, transcript.width, BUSY_FRAMES[self.busy_frame]);
            }
            if let Some(turn) = history.get(index) {
                append_turn(&mut lines, turn, transcript.width);
            }
        }
        if let Some(status) = &self.model_status {
            match status {
                Ok(message) => lines.extend(display_lines(message).map(|line| line.cyan())),
                Err(error) => lines.extend(display_lines(error).map(|line| line.red())),
            }
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

/// Rendered prompts for the current chat turn or workflow node, with one selected step and
/// its wrapped scroll position.
#[derive(Default)]
struct ModelInputs {
    /// Rendered strings in inference-step order for the current inspected turn or node.
    prompts: Vec<String>,
    /// Index of the inspected prompt, zero when the pane is empty.
    selected: usize,
    /// Top wrapped row of the selected prompt.
    scroll: u16,
    /// Highest valid scroll offset computed for the current pane dimensions.
    last_row: u16,
    /// Vertical scroll increment for the inspector.
    page: u16,
}

impl ModelInputs {
    /// Append one observed prompt and follow it only when the reader was already viewing the
    /// latest step.
    fn push(&mut self, prompt: String) {
        let follows_latest = self.selected == self.prompts.len().saturating_sub(1);
        self.prompts.push(prompt);
        if follows_latest {
            self.select(self.prompts.len() - 1);
        }
    }

    /// Clamp the selected step to the available range and reset its scroll when the selection
    /// changes.
    fn select(&mut self, index: usize) {
        let index = index.min(self.prompts.len().saturating_sub(1));
        if index != self.selected {
            self.selected = index;
            self.scroll = 0;
        }
    }

    /// Wrap the selected model input and clamp scroll to the number of rows that fit the
    /// inspector.
    fn render(&mut self, frame: &mut Frame<'_>, area: Rect) {
        let multiple = self.prompts.len() > 1;
        let title = if multiple {
            format!(" Model Input {}/{} ", self.selected + 1, self.prompts.len())
        } else {
            " Model Input ".into()
        };
        let mut block = Block::bordered()
            .border_type(BorderType::Rounded)
            .title(title.bold());
        let inner = block.inner(area);
        if inner.is_empty() {
            frame.render_widget(block, area);
            return;
        }

        let lines = match self.prompts.get(self.selected) {
            Some(prompt) => display_lines(prompt).collect::<Vec<_>>(),
            None => vec![Line::raw("No model input yet").dim()],
        };
        let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
        self.page = inner.height.saturating_sub(1).max(1);
        let last_row = paragraph
            .line_count(inner.width)
            .saturating_sub(usize::from(inner.height));
        self.last_row = u16::try_from(last_row).unwrap_or(u16::MAX);
        self.scroll = self.scroll.min(self.last_row);
        let controls = match (last_row > 0, multiple) {
            (true, true) => " Shift+PgUp/Dn · Alt+←/→ ",
            (true, false) => " Shift+PgUp/PgDn ",
            (false, true) => " Alt+←/→ ",
            (false, false) => "",
        };
        if !controls.is_empty() {
            block = block.title_bottom(controls.dim());
        }
        frame.render_widget(paragraph.block(block).scroll((self.scroll, 0)), area);
    }
}

/// Bottom-align short transcripts and clamp scroll in wrapped rows. Return to follow mode
/// when the viewport reaches the latest row.
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

/// Draw the user's input as literal lines and render the assistant's text as Markdown.
fn append_turn<'a>(lines: &mut Vec<Line<'a>>, turn: &'a Turn, width: u16) {
    if !lines.is_empty() {
        lines.push(Line::default());
    }
    lines.push(Line::raw("You").cyan().bold());
    lines.extend(display_lines(&turn.prompt));
    lines.push(Line::raw("Arta").green().bold());
    lines.extend(markdown::render(&turn.reply, width));
}

/// Split literal text at line boundaries and strip terminal controls from each line.
fn display_lines(text: &str) -> impl Iterator<Item = Line<'_>> {
    text.lines()
        .map(|line| Line::raw(markdown::clean_terminal_text(line)))
}

/// Single-line buffer with a UTF-8 byte cursor kept on an extended grapheme boundary.
#[derive(Default)]
struct Editor<'a> {
    /// Editable single-line prompt buffer.
    text: String,
    /// Byte offset at a grapheme boundary within the buffer.
    cursor: usize,
    /// Index within the current prefix's sorted completion matches.
    selected: usize,
    /// Borrowed parser entries available for prefix matching.
    candidates: Vec<&'a suggestion::Candidate>,
}

impl<'a> Editor<'a> {
    /// Submit the entire buffer and reset the cursor and completion selection.
    fn take(&mut self) -> String {
        self.cursor = 0;
        self.selected = 0;
        mem::take(&mut self.text)
    }

    /// Insert at the cursor, then account for graphemes that merged across the insertion
    /// boundary.
    fn insert(&mut self, text: &str) {
        self.text.insert_str(self.cursor, text);
        self.cursor = self.boundary_at_or_after(self.cursor + text.len());
    }

    /// Find the next valid grapheme boundary or the end of the string.
    fn boundary_at_or_after(&self, offset: usize) -> usize {
        self.text
            .grapheme_indices(true)
            .map(|(index, _)| index)
            .find(|&index| index >= offset)
            .unwrap_or(self.text.len())
    }

    /// Look up completions for the token intersecting the current cursor position.
    fn suggestions(&self) -> Option<suggestion::Suggestions<'a>> {
        suggestion::matches(&self.candidates, &self.text, self.cursor)
    }

    /// Move or delete by grapheme cluster and accept completion by whole token. Ordinary
    /// edits reset the selected candidate.
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

    /// Walk backwards from the cursor in terminal columns to select a visible substring and
    /// position the hardware cursor.
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
    fn model_input_pane_pages_and_switches_between_inference_steps() -> Result<()> {
        let mut terminal = Terminal::new(TestBackend::new(60, 12))?;
        let mut tui = Tui::default();
        let mut render = |tui: &mut Tui<'_>, row, expected| -> Result<()> {
            terminal.draw(|frame| tui.render(frame, &[], None))?;
            let buffer = terminal.backend().buffer();
            let line: String = (31..59)
                .map(|column| buffer[(column, row)].symbol())
                .collect();
            assert_eq!(line.trim_end(), expected);
            assert!(tui.scroll.is_none());
            Ok(())
        };

        let key = |code, modifiers| Event::Key(KeyEvent::new(code, modifiers));
        render(&mut tui, 1, "No model input yet")?;
        tui.model_inputs.push(
            "<system>\n\n<tools>lookup</tools>\n\nCompare each claim with the source notes.".into(),
        );
        render(&mut tui, 1, "<system>")?;
        render(&mut tui, 3, "<tools>lookup</tools>")?;
        tui.handle(key(KeyCode::PageDown, KeyModifiers::SHIFT));
        render(&mut tui, 4, "source notes.")?;
        tui.handle(key(KeyCode::PageUp, KeyModifiers::SHIFT));
        render(&mut tui, 1, "<system>")?;

        tui.model_inputs.push("<tool>Blue</tool>".into());
        render(&mut tui, 1, "<tool>Blue</tool>")?;
        tui.handle(key(KeyCode::Left, KeyModifiers::ALT));
        tui.model_inputs.push("<tool>Green</tool>".into());
        render(&mut tui, 1, "<system>")?;
        tui.handle(key(KeyCode::Right, KeyModifiers::ALT));
        render(&mut tui, 1, "<tool>Blue</tool>")?;
        tui.handle(key(KeyCode::Right, KeyModifiers::ALT));
        render(&mut tui, 1, "<tool>Green</tool>")?;
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
        assert_ne!(buffer[(25, 19)].symbol(), "x");
        assert_eq!(buffer[(40, 19)].symbol(), "x");
        let first: String = (0..5).map(|column| buffer[(column, 17)].symbol()).collect();
        let second: String = (0..6).map(|column| buffer[(column, 18)].symbol()).collect();
        let third: String = (0..6).map(|column| buffer[(column, 19)].symbol()).collect();
        assert_eq!(first, "/exit");
        assert_eq!(second, "/model");
        assert_eq!(third, "/reset");
        assert!(
            buffer[(0, 17)]
                .modifier
                .contains(ratatui::style::Modifier::REVERSED)
        );

        tui.handle(key(KeyCode::Down));
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

    #[test]
    fn workflow_pane_highlights_the_active_node_and_keeps_its_output_in_chat() -> Result<()> {
        use arta::workflow::{LoopIteration, node::Node};
        use ratatui::style::Modifier;

        let nodes = [
            Node::Loop {
                id: "review-loop".into(),
                nodes: vec![
                    Node::Agent {
                        id: "write".into(),
                        agent: "writer".into(),
                    },
                    Node::Agent {
                        id: "qa".into(),
                        agent: "tester".into(),
                    },
                ],
            },
            Node::Agent {
                id: "report".into(),
                agent: "report".into(),
            },
        ];
        let mut terminal = Terminal::new(TestBackend::new(96, 24))?;
        let mut pane = WorkflowPane::from_nodes("sample", &nodes);
        pane.start();
        let mut tui = Tui {
            workflow: Some(pane),
            workflow_history: vec![WorkflowTranscript::new(0, "Review this".into())],
            ..Tui::default()
        };
        tui.workflow_event(WorkflowEvent::NodeStarted {
            id: "qa".into(),
            agent: "tester".into(),
            loops: vec![LoopIteration {
                id: "review-loop".into(),
                number: 2,
            }],
        });
        tui.workflow_event(WorkflowEvent::ModelLoading {
            previous: Some("llama.cpp/writer".into()),
            next: "llama.cpp/reviewer".into(),
        });
        tui.workflow_event(WorkflowEvent::ModelInput("QA system prompt".into()));
        tui.workflow_event(WorkflowEvent::LoopToolCall {
            iteration: LoopIteration {
                id: "review-loop".into(),
                number: 2,
            },
            name: "break".into(),
            arguments: [("message".into(), "Checking the work".into())]
                .into_iter()
                .collect(),
            error: None,
        });
        tui.workflow_event(WorkflowEvent::Output("**Checking** the work".into()));
        terminal.draw(|frame| tui.render(frame, &[], None))?;
        let buffer = terminal.backend().buffer();
        let text = |row, columns: std::ops::Range<u16>| {
            columns
                .map(|column| buffer[(column, row)].symbol())
                .collect::<String>()
        };
        assert!(text(1, 1..47).contains("Switching to llama.cpp/reviewer"));
        assert!(text(2, 1..47).contains("review-loop · iteration 2"));
        assert!(text(3, 1..47).contains("write (writer)"));
        assert!(text(4, 1..47).contains("▶ qa (tester)"));
        assert!(text(5, 1..47).contains("report"));
        assert!(buffer[(5, 4)].modifier.contains(Modifier::REVERSED));
        assert!(text(1, 49..95).contains("QA system prompt"));
        assert!(text(18, 0..96).contains("qa (tester) · review-loop iteration 2"));
        assert!(text(19, 0..96).contains(
            "Tool call break({\"message\":\"Checking the work\"}) · review-loop iteration 2"
        ));
        assert!(text(20, 0..96).contains("Checking the work"));
        Ok(())
    }
}

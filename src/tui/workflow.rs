//! Two projections of one workflow. The outline flattens the static tree for navigation. The
//! transcript appends every actual visit, including repeated visits to the same node.

use crate::markdown::{clean_terminal_text, render};
use arta::workflow::{LoopIteration, Workflow, node::Node};
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::Stylize,
    text::Line,
    widgets::{Block, BorderType, Paragraph, Wrap},
};

/// Scrollable definition tree annotated with the current visit, loop stack, model, and run
/// status.
pub(super) struct WorkflowPane {
    /// YAML workflow label used in the pane title.
    id: String,
    /// Preorder outline containing one row per declared node.
    rows: Vec<Row>,
    /// Ready before submission, then running, complete, or failed.
    status: &'static str,
    /// Index of the active node row, retained when a visit fails.
    current: Option<usize>,
    /// Iteration path used to annotate active ancestor rows.
    loops: Vec<LoopIteration>,
    /// Display label built from model-loading and model-loaded events.
    pub(super) model: Option<String>,
    /// Index of the first visible outline row.
    scroll: usize,
    /// Number of outline rows available after reserving space for the model label.
    page: usize,
}

/// One flattened tree entry retaining indentation and an optional leaf label.
struct Row {
    /// Definition ID used to locate the row on a node-start event.
    id: String,
    /// Tree depth, rendered as two spaces per enclosing loop.
    indent: usize,
    /// Agent profile or image node type. Absence identifies a loop heading.
    leaf: Option<String>,
}

impl WorkflowPane {
    /// Build an outline from the validated tree before the first agent visit.
    pub(super) fn new(workflow: &Workflow) -> Self {
        Self::from_nodes(workflow.id(), workflow.nodes())
    }

    /// Flatten the definition in preorder and initialize a ready, unselected outline.
    pub(super) fn from_nodes(id: &str, nodes: &[Node]) -> Self {
        let mut rows = Vec::new();
        Self::append_nodes(&mut rows, nodes, 0);
        Self {
            id: id.into(),
            rows,
            status: "Ready",
            current: None,
            loops: Vec::new(),
            model: None,
            scroll: 0,
            page: 1,
        }
    }

    /// Add each loop before its more deeply indented children and retain agent names on
    /// leaves.
    fn append_nodes(rows: &mut Vec<Row>, nodes: &[Node], indent: usize) {
        for node in nodes {
            match node {
                Node::Agent { id, agent } => rows.push(Row {
                    id: id.clone(),
                    indent,
                    leaf: Some(agent.clone()),
                }),
                Node::Image { id, .. } => rows.push(Row {
                    id: id.clone(),
                    indent,
                    leaf: Some("image".into()),
                }),
                Node::Loop { id, nodes } => {
                    rows.push(Row {
                        id: id.clone(),
                        indent,
                        leaf: None,
                    });
                    Self::append_nodes(rows, nodes, indent + 1);
                }
            }
        }
    }

    /// Change the outline's status from ready to running.
    pub(super) fn start(&mut self) {
        self.status = "Running";
    }

    /// Select the row for a node start and reveal it within the current page.
    pub(super) fn node(&mut self, id: &str, loops: Vec<LoopIteration>) {
        self.current = self.rows.iter().position(|row| row.id == id);
        self.loops = loops;
        self.reveal_current();
    }

    /// Move the viewport the minimum distance needed to include the active row.
    fn reveal_current(&mut self) {
        if let Some(current) = self.current {
            self.scroll = self
                .scroll
                .clamp((current + 1).saturating_sub(self.page), current);
        }
    }

    /// Clear the active path on success and preserve it on failure for inspection.
    pub(super) fn finish(&mut self, success: bool) {
        self.status = if success { "Complete" } else { "Failed" };
        if success {
            self.current = None;
            self.loops.clear();
        }
    }

    /// Move by one viewport of definition rows. Rendering clamps the last page.
    pub(super) fn page(&mut self, down: bool) {
        self.scroll = if down {
            self.scroll.saturating_add(self.page)
        } else {
            self.scroll.saturating_sub(self.page)
        };
    }

    /// Reserve space for the model label, clamp the visible rows, and annotate active loop
    /// iterations.
    pub(super) fn render(&mut self, frame: &mut Frame<'_>, area: Rect) {
        let title = format!(" {} · {} ", clean_terminal_text(&self.id), self.status);
        let mut block = Block::bordered()
            .border_type(BorderType::Rounded)
            .title(title.bold());
        let inner = block.inner(area);
        let model_status = self.model.as_deref().map(|message| {
            Paragraph::new(Line::raw(clean_terminal_text(message).into_owned()).cyan())
                .wrap(Wrap { trim: true })
        });
        let status_height = model_status.as_ref().map_or(0, |status| {
            status
                .line_count(inner.width)
                .min(2)
                .min(usize::from(inner.height.saturating_sub(1))) as u16
        });
        let [model, nodes] =
            Layout::vertical([Constraint::Length(status_height), Constraint::Min(0)]).areas(inner);
        let page = usize::from(nodes.height).max(1);
        if page != self.page {
            self.page = page;
            self.reveal_current();
        }
        let last = self.rows.len().saturating_sub(self.page);
        self.scroll = self.scroll.min(last);
        if last > 0 {
            block = block.title_bottom(" Ctrl+PgUp/PgDn ".dim());
        }
        frame.render_widget(block, area);
        if let Some(status) = model_status {
            frame.render_widget(status, model);
        }
        let rows = self
            .rows
            .iter()
            .enumerate()
            .skip(self.scroll)
            .take(self.page)
            .map(|(index, row)| {
                let active = self.current == Some(index);
                let marker = if active { "▶ " } else { "  " };
                let mut label = format!(
                    "{}{marker}{}",
                    "  ".repeat(row.indent),
                    clean_terminal_text(&row.id)
                );
                match &row.leaf {
                    Some(agent) if agent != &row.id => {
                        label.push_str(&format!(" ({})", clean_terminal_text(agent)))
                    }
                    None => {
                        if let Some(iteration) = self.loops.iter().find(|item| item.id == row.id) {
                            label.push_str(&format!(" · iteration {}", iteration.number));
                        } else {
                            label.push_str(" · loop");
                        }
                    }
                    _ => {}
                }
                let line = Line::raw(label);
                if active {
                    line.cyan().bold().reversed()
                } else {
                    line
                }
            })
            .collect::<Vec<_>>();
        frame.render_widget(Paragraph::new(rows), nodes);
    }
}

/// Display record anchored at the number of ordinary chat turns present when the workflow
/// began.
pub(super) struct WorkflowTranscript {
    /// Number of completed ordinary chat turns when this workflow started. Anchors it in the
    /// display timeline.
    pub(super) history_index: usize,
    /// Original workflow request displayed once above its visits.
    prompt: String,
    /// Chronological visits including repeated entries for loop iterations.
    steps: Vec<Step>,
    /// Terminal execution error shown after any partial output.
    pub(super) error: Option<String>,
    /// Draw the activity indicator while awaiting the current visit's first nonempty output.
    pub(super) waiting: bool,
}

/// One chronological visit, retaining its label, control calls, and accumulated reply.
struct Step {
    /// Node, agent and loop path formatted when the visit starts.
    label: String,
    /// Rendered loop-control attempts belonging to this visit.
    tool_calls: Vec<String>,
    /// Concatenated output chunks, retained for Markdown rendering on redraw.
    reply: String,
}

impl WorkflowTranscript {
    /// Anchor an empty workflow transcript at the current position in the conversation.
    pub(super) fn new(history_index: usize, prompt: String) -> Self {
        Self {
            history_index,
            prompt,
            steps: Vec::new(),
            error: None,
            waiting: true,
        }
    }

    /// Start a new transcript entry even when the same node ID has run before.
    pub(super) fn node(&mut self, label: String) {
        self.steps.push(Step {
            label,
            tool_calls: Vec::new(),
            reply: String::new(),
        });
        self.waiting = true;
    }

    /// Attach a control call label to the current visit.
    pub(super) fn tool_call(&mut self, label: String) {
        if let Some(step) = self.steps.last_mut() {
            step.tool_calls.push(label);
        }
    }

    /// Append a model chunk to the current visit and end its waiting animation on nonempty
    /// text.
    pub(super) fn output(&mut self, chunk: &str) {
        if let Some(step) = self.steps.last_mut() {
            step.reply.push_str(chunk);
        }
        self.waiting &= chunk.is_empty();
    }

    /// Render the original request, visits, and any terminal error at the recorded chat
    /// position.
    pub(super) fn append<'a>(&'a self, lines: &mut Vec<Line<'a>>, width: u16, busy: &str) {
        if !lines.is_empty() {
            lines.push(Line::default());
        }
        lines.push(Line::raw("You").cyan().bold());
        lines.extend(super::display_lines(&self.prompt));
        for step in &self.steps {
            lines.push(Line::raw(clean_terminal_text(&step.label)).green().bold());
            lines.extend(
                step.tool_calls
                    .iter()
                    .map(|call| Line::raw(clean_terminal_text(call)).cyan()),
            );
            lines.extend(render(&step.reply, width));
        }
        if self.waiting {
            lines.push(Line::raw(busy.to_owned()).green());
        }
        if let Some(error) = &self.error {
            lines.push(Line::raw("Error").red().bold());
            lines.extend(super::display_lines(error).map(|line| line.red()));
        }
    }
}

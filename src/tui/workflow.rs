use crate::markdown::{clean_terminal_text, render};
use arta::workflow::{LoopIteration, Workflow, node::Node};
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::Stylize,
    text::Line,
    widgets::{Block, BorderType, Paragraph, Wrap},
};

pub(super) struct WorkflowPane {
    id: String,
    rows: Vec<Row>,
    status: &'static str,
    current: Option<usize>,
    loops: Vec<LoopIteration>,
    pub(super) model: Option<String>,
    scroll: usize,
    page: usize,
}

struct Row {
    id: String,
    indent: usize,
    agent: Option<String>,
}

impl WorkflowPane {
    pub(super) fn new(workflow: &Workflow) -> Self {
        Self::from_nodes(workflow.id(), workflow.nodes())
    }

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

    fn append_nodes(rows: &mut Vec<Row>, nodes: &[Node], indent: usize) {
        for node in nodes {
            match node {
                Node::Agent { id, agent } => rows.push(Row {
                    id: id.clone(),
                    indent,
                    agent: Some(agent.clone()),
                }),
                Node::Loop { id, nodes } => {
                    rows.push(Row {
                        id: id.clone(),
                        indent,
                        agent: None,
                    });
                    Self::append_nodes(rows, nodes, indent + 1);
                }
            }
        }
    }

    pub(super) fn start(&mut self) {
        self.status = "Running";
    }

    pub(super) fn node(&mut self, id: &str, loops: Vec<LoopIteration>) {
        self.current = self.rows.iter().position(|row| row.id == id);
        self.loops = loops;
        self.reveal_current();
    }

    fn reveal_current(&mut self) {
        if let Some(current) = self.current {
            self.scroll = self
                .scroll
                .clamp((current + 1).saturating_sub(self.page), current);
        }
    }

    pub(super) fn finish(&mut self, success: bool) {
        self.status = if success { "Complete" } else { "Failed" };
        if success {
            self.current = None;
            self.loops.clear();
        }
    }

    pub(super) fn page(&mut self, down: bool) {
        self.scroll = if down {
            self.scroll.saturating_add(self.page)
        } else {
            self.scroll.saturating_sub(self.page)
        };
    }

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
                match &row.agent {
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

pub(super) struct WorkflowTranscript {
    pub(super) history_index: usize,
    prompt: String,
    steps: Vec<Step>,
    pub(super) error: Option<String>,
    pub(super) waiting: bool,
}

struct Step {
    label: String,
    tool_calls: Vec<String>,
    reply: String,
}

impl WorkflowTranscript {
    pub(super) fn new(history_index: usize, prompt: String) -> Self {
        Self {
            history_index,
            prompt,
            steps: Vec::new(),
            error: None,
            waiting: true,
        }
    }

    pub(super) fn node(&mut self, label: String) {
        self.steps.push(Step {
            label,
            tool_calls: Vec::new(),
            reply: String::new(),
        });
        self.waiting = true;
    }

    pub(super) fn tool_call(&mut self, label: String) {
        if let Some(step) = self.steps.last_mut() {
            step.tool_calls.push(label);
        }
    }

    pub(super) fn output(&mut self, chunk: &str) {
        if let Some(step) = self.steps.last_mut() {
            step.reply.push_str(chunk);
        }
        self.waiting &= chunk.is_empty();
    }

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

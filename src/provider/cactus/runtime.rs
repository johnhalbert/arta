//! JSON lines transport for a Cactus SDK worker. The owner controls process lifetime.

use super::RuntimeConfig;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    env,
    io::{BufRead, Write},
    path::PathBuf,
    process::Stdio,
    sync::OnceLock,
};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
};

pub(super) struct Worker {
    _child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}

pub(super) const WORKER_FLAG: &str = "--internal-cactus-worker";
static CLI_EXE: OnceLock<PathBuf> = OnceLock::new();

pub(super) fn register_cli() {
    if let Ok(path) = env::current_exe() {
        let _ = CLI_EXE.set(path);
    }
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum Response<T> {
    Ok { result: T },
    Error { message: String },
}

impl Worker {
    /// Start the native worker for one model conversation.
    pub(super) fn start(config: &RuntimeConfig, model: &str) -> Result<Self> {
        tokio::runtime::Handle::try_current()
            .context("Cactus inference requires a Tokio runtime")?;
        let path = match &config.worker_path {
            Some(path) => std::path::absolute(path)?,
            None => match CLI_EXE.get() {
                Some(path) => path.clone(),
                None => {
                    env::current_exe()?.with_file_name(format!("arta{}", env::consts::EXE_SUFFIX))
                }
            },
        };
        let mut command = Command::new(&path);
        command
            .args([WORKER_FLAG, model])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .kill_on_drop(true);
        if let Some(path) = &config.library_path {
            command.env("CACTUS_LIB_PATH", path);
        }
        if config.offline {
            command.env("HF_HUB_OFFLINE", "1");
        }
        let mut child = command.spawn().with_context(|| {
            format!(
                "failed to start Cactus worker '{}'; set providers.cactus.runtime.worker_path to the Arta executable",
                path.display()
            )
        })?;
        let input = child
            .stdin
            .take()
            .context("Cactus worker stdin is unavailable")?;
        let output = child
            .stdout
            .take()
            .context("Cactus worker stdout is unavailable")?;
        Ok(Self {
            _child: child,
            input,
            output: BufReader::new(output),
        })
    }

    pub(super) async fn request<T: DeserializeOwned>(
        &mut self,
        request: impl Serialize,
    ) -> Result<T> {
        let mut input = serde_json::to_vec(&request).context("failed to encode Cactus request")?;
        input.push(b'\n');
        self.input
            .write_all(&input)
            .await
            .context("failed to write to Cactus worker")?;

        let mut line = String::new();
        if self
            .output
            .read_line(&mut line)
            .await
            .context("failed to read from Cactus worker")?
            == 0
        {
            bail!("Cactus worker exited before sending a response");
        }
        match serde_json::from_str(&line).context("invalid response from Cactus worker")? {
            Response::Ok { result } => Ok(result),
            Response::Error { message } => bail!("Cactus worker: {message}"),
        }
    }
}

/// Serve one isolated model process until EOF or the first error.
pub(super) fn serve<Request, Reply>(
    input: impl BufRead,
    mut output: impl Write,
    mut handle: impl FnMut(Request) -> Result<Reply>,
) -> Result<()>
where
    Request: DeserializeOwned,
    Reply: Serialize,
{
    for line in input.lines() {
        let reply = serde_json::from_str(&line?)
            .context("invalid Cactus worker request")
            .and_then(&mut handle);
        let response = match reply {
            Ok(result) => Response::Ok { result },
            Err(error) => Response::Error {
                message: format!("{error:#}"),
            },
        };
        serde_json::to_writer(&mut output, &response)?;
        output.write_all(b"\n")?;
        output.flush()?;
        if matches!(response, Response::Error { .. }) {
            break;
        }
    }
    Ok(())
}

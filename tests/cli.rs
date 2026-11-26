//! Process-level CLI contracts. These tests launch the built binary and inspect exit status
//! and stderr for mode selection, invalid arguments, empty stdin, and model-loading failures.

use std::{
    fs,
    io::Write,
    path::Path,
    process::{Command, Stdio},
};

#[test]
fn cactus_worker_mode_reports_protocol_errors_without_starting_the_cli() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_arta"))
        .args(["--internal-cactus-worker", "needle"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    writeln!(
        child.stdin.take().unwrap(),
        r#"{{"operation":"complete","text":"hello","max_tokens":32}}"#
    )
    .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&output.stdout).unwrap(),
        serde_json::json!({"status":"error", "message":"Needle worker is not initialized"})
    );
}

#[test]
fn no_arguments_starts_repl_setup() {
    let output = Command::new(env!("CARGO_BIN_EXE_arta"))
        .current_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests"))
        .stdin(Stdio::null())
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("failed to start the REPL"));
}

#[test]
fn missing_prompt_reports_usage() {
    let output = Command::new(env!("CARGO_BIN_EXE_arta"))
        .args(["--model", "unused.gguf"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("--prompt"));
}

#[test]
fn agent_selection_allows_interactive_startup_and_rejects_paths() {
    let output = Command::new(env!("CARGO_BIN_EXE_arta"))
        .args(["--agent", "../reviewer"])
        .stdin(Stdio::null())
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("invalid agent name"));
}

#[test]
fn empty_stdin_is_rejected_before_opening_model() {
    let output = Command::new(env!("CARGO_BIN_EXE_arta"))
        .args(["--model", "unused.gguf", "--prompt", "-"])
        .stdin(Stdio::null())
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("prompt cannot be empty"));
}

#[test]
fn missing_model_reports_an_error_without_panicking() {
    let output = Command::new(env!("CARGO_BIN_EXE_arta"))
        .args(["--model", "tests/missing-model.gguf", "--prompt", "hi"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("cannot access model"));
}

#[test]
fn image_workflow_loads_the_library_staged_by_cargo() -> anyhow::Result<()> {
    let fixture = tempfile::tempdir()?;
    let workflow = fixture.path().join("workflows/render");
    fs::create_dir_all(&workflow)?;
    fs::write(
        workflow.join("workflow.yaml"),
        "id: render\nnodes:\n  - { id: first, type: image }\n",
    )?;
    fs::write(
        fixture.path().join("config.toml"),
        r#"
[workflows]
path = "workflows"
[[providers."stable-diffusion.cpp".models]]
id = "image-model"
path = "missing.safetensors"
"#,
    )?;
    let output = Command::new(env!("CARGO_BIN_EXE_arta"))
        .current_dir(fixture.path())
        .args([
            "--config",
            "config.toml",
            "--workflow",
            "render",
            "--prompt",
            "A tree",
        ])
        .output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(output.stdout.is_empty());
    assert!(
        stderr.contains("image model file") && stderr.contains("missing.safetensors"),
        "{stderr}"
    );
    Ok(())
}

#[test]
fn audio_workflow_loads_the_library_staged_by_cargo() -> anyhow::Result<()> {
    let fixture = tempfile::tempdir()?;
    let workflow = fixture.path().join("workflows/narrate");
    fs::create_dir_all(&workflow)?;
    fs::write(
        workflow.join("workflow.yaml"),
        "id: narrate\nnodes:\n  - { id: first, type: audio }\n",
    )?;
    fs::write(
        fixture.path().join("config.toml"),
        r#"
[workflows]
path = "workflows"
[[providers."audio.cpp".models]]
id = "speech"
path = "missing.gguf"
"#,
    )?;
    let output = Command::new(env!("CARGO_BIN_EXE_arta"))
        .current_dir(fixture.path())
        .args([
            "--config",
            "config.toml",
            "--workflow",
            "narrate",
            "--prompt",
            "Hello",
        ])
        .output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(output.stdout.is_empty());
    assert!(
        stderr.contains("[first (audio · audio.cpp/speech)]"),
        "{stderr}"
    );
    assert!(
        stderr.contains("audio model path") && stderr.contains("missing.gguf"),
        "{stderr}"
    );
    assert!(!fixture.path().join("audio").exists());
    Ok(())
}

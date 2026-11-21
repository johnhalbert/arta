use std::process::{Command, Stdio};

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

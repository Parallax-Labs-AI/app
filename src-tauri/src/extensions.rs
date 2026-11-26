use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{path::Path, process::Stdio, time::Duration};
use tokio::{
    io::{AsyncReadExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Program {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    #[serde(default)]
    pub enabled: bool,
}

pub fn validate(programs: &[Program], uvx: bool) -> Result<()> {
    ensure!(programs.len() <= 12, "Add at most 12 entries.");
    let mut names = std::collections::HashSet::new();
    for program in programs {
        ensure!(
            !program.name.trim().is_empty()
                && program.name.len() <= 60
                && !program.name.chars().any(char::is_control)
                && names.insert(program.name.trim().to_ascii_lowercase()),
            "Use a unique name for each entry."
        );
        ensure!(
            (Path::new(&program.command).is_absolute() || uvx && program.command == "uvx")
                && program.command.len() <= 1024
                && !program.command.chars().any(char::is_control),
            "Enter an absolute executable path or use uvx for an MCP server."
        );
        ensure!(
            program.args.len() <= 32
                && program
                    .args
                    .iter()
                    .all(|arg| arg.len() <= 4096 && !arg.chars().any(char::is_control)),
            "Use at most 32 plain-text arguments."
        );
    }
    Ok(())
}

fn command(program: &Program, executable: &Path, directory: &Path) -> Command {
    let mut command = crate::hardware::runtime_command(executable);
    command
        .env_clear()
        .args(&program.args)
        .current_dir(directory);
    for name in [
        "PATH",
        "SystemRoot",
        "WINDIR",
        "TEMP",
        "TMP",
        "TMPDIR",
        "LANG",
    ] {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    if program.command == "uvx" {
        command
            .env("UV_CACHE_DIR", directory.join("cache"))
            .env("UV_PYTHON_INSTALL_DIR", directory.join("python"))
            .env("UV_PYTHON_PREFERENCE", "only-managed");
    }
    let mut command = Command::from(command);
    command.kill_on_drop(true);
    command
}

pub async fn run_command(
    program: &Program,
    directory: &Path,
    cancel: &CancellationToken,
) -> Result<String> {
    let mut child = command(program, Path::new(&program.command), directory)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("Could not start the configured command.")?;
    let mut stdout = child.stdout.take().context("Missing stdout.")?.take(32769);
    let mut stderr = child.stderr.take().context("Missing stderr.")?.take(32769);
    let run = async {
        let mut output = Vec::new();
        let mut errors = Vec::new();
        tokio::try_join!(
            stdout.read_to_end(&mut output),
            stderr.read_to_end(&mut errors)
        )?;
        ensure!(
            output.len() <= 32768 && errors.len() <= 32768,
            "Command output exceeded its limit."
        );
        let status = child.wait().await?;
        let text = format!(
            "Exit status: {status}\n{}\n{}",
            String::from_utf8_lossy(&output),
            String::from_utf8_lossy(&errors)
        );
        Ok(text)
    };
    tokio::select! {
        _ = cancel.cancelled() => anyhow::bail!("Command cancelled."),
        result = tokio::time::timeout(Duration::from_secs(45), run) => result.context("Command timed out.")?,
    }
}

pub struct McpSession {
    _child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    pub name: String,
    pub tools: Vec<Value>,
}

impl McpSession {
    pub async fn connect(
        program: &Program,
        root: &Path,
        client: &reqwest::Client,
        cancel: &CancellationToken,
    ) -> Result<Self> {
        let directory = root.join("custom-mcp");
        tokio::fs::create_dir_all(&directory).await?;
        let executable = if program.command == "uvx" {
            crate::search::install(&directory.join("uv"), client, cancel).await?
        } else {
            program.command.clone().into()
        };
        let mut child = command(program, &executable, &directory)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("Could not start the MCP server.")?;
        let mut session = Self {
            input: child.stdin.take().context("Missing MCP input.")?,
            output: BufReader::new(child.stdout.take().context("Missing MCP output.")?),
            _child: child,
            name: program.name.clone(),
            tools: Vec::new(),
        };
        let initialize = async {
            crate::search::send(&mut session.input, json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{
                "protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"Parallax App","version":env!("CARGO_PKG_VERSION")}
            }})).await?;
            let hello = crate::search::receive(&mut session.output, 1).await?;
            ensure!(
                matches!(
                    hello["protocolVersion"].as_str(),
                    Some("2025-06-18" | "2025-03-26" | "2025-11-25")
                ),
                "Unsupported MCP protocol version."
            );
            crate::search::send(
                &mut session.input,
                json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
            )
            .await?;
            crate::search::send(
                &mut session.input,
                json!({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}),
            )
            .await?;
            let result = crate::search::receive(&mut session.output, 2).await?;
            session.tools = result["tools"]
                .as_array()
                .context("Missing MCP tool list.")?
                .iter()
                .filter(|tool| {
                    tool["name"].as_str().is_some_and(|name| {
                        !name.is_empty() && name.len() <= 128 && !name.chars().any(char::is_control)
                    }) && tool["inputSchema"]["type"] == "object"
                        && tool["inputSchema"].to_string().len() <= 4096
                })
                .take(8)
                .cloned()
                .collect();
            Ok(session)
        };
        tokio::select! {
            _ = cancel.cancelled() => anyhow::bail!("MCP setup cancelled."),
            result = tokio::time::timeout(Duration::from_secs(25), initialize) => result.context("MCP setup timed out.")?,
        }
    }

    pub async fn call(
        &mut self,
        index: usize,
        id: u32,
        arguments: Value,
        cancel: &CancellationToken,
    ) -> Result<String> {
        let name = &self.tools.get(index).context("Unknown MCP tool.")?["name"];
        let request = async {
            crate::search::send(&mut self.input, json!({"jsonrpc":"2.0","id":id,"method":"tools/call","params":{"name":name,"arguments":arguments}})).await?;
            let result = crate::search::receive(&mut self.output, id).await?;
            ensure!(
                result["isError"] != true,
                "The MCP server reported a tool error."
            );
            let mut text: String = result["content"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|block| block["type"] == "text")
                .filter_map(|block| block["text"].as_str())
                .flat_map(|text| text.chars().chain(std::iter::once('\n')))
                .take(12000)
                .collect();
            if text.trim().is_empty() && result["structuredContent"].is_object() {
                text = result["structuredContent"]
                    .to_string()
                    .chars()
                    .take(12000)
                    .collect();
            }
            ensure!(!text.trim().is_empty(), "The MCP tool returned no text.");
            Ok(text)
        };
        tokio::select! {
            _ = cancel.cancelled() => anyhow::bail!("MCP tool cancelled."),
            result = tokio::time::timeout(Duration::from_secs(45), request) => result.context("MCP tool timed out.")?,
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[tokio::test]
    async fn configured_command_keeps_arguments_literal() {
        let root = tempfile::tempdir().unwrap();
        let program = Program {
            name: "Print".into(),
            command: "/usr/bin/printf".into(),
            args: vec!["%s".into(), "hello; still one argument".into()],
            enabled: true,
        };
        let result = run_command(&program, root.path(), &CancellationToken::new())
            .await
            .unwrap();
        assert!(result.contains("hello; still one argument"));
    }

    #[tokio::test]
    async fn custom_stdio_server_lists_tools_and_returns_call_output() {
        let root = tempfile::tempdir().unwrap();
        let script = root.path().join("fixture.sh");
        std::fs::write(&script, r#"#!/bin/sh
IFS= read -r frame
printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-06-18"}}'
IFS= read -r frame
IFS= read -r frame
printf '%s\n' '{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"echo","inputSchema":{"type":"object","properties":{"text":{"type":"string"}}}}]}}'
IFS= read -r frame
case "$frame" in
    *hello*) printf '%s\n' '{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"MCP received hello"}]}}' ;;
    *) exit 1 ;;
esac
"#).unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).unwrap();
        let program = Program {
            name: "Fixture".into(),
            command: script.display().to_string(),
            args: Vec::new(),
            enabled: true,
        };
        let cancel = CancellationToken::new();
        let mut server =
            McpSession::connect(&program, root.path(), &reqwest::Client::new(), &cancel)
                .await
                .unwrap();
        assert_eq!(server.tools[0]["name"], "echo");
        assert_eq!(
            server
                .call(0, 3, json!({"text":"hello"}), &cancel)
                .await
                .unwrap()
                .trim(),
            "MCP received hello"
        );
    }
}

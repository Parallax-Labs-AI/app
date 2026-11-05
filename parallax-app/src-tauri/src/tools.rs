use crate::{config, AppState};
use serde_json::{json, Value};
use std::{path::Path, time::Duration};

/// Truncates to at most `n` characters.
pub fn clip(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn tool(name: &str, desc: &str, props: Value, required: &[&str]) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": { "type": "object", "properties": props, "required": required }
        }
    })
}

/// OpenAI-style schemas for the built-in tools.
pub fn schemas() -> Vec<Value> {
    vec![
        tool("read_file", "Read a file.", json!({"path": {"type": "string"}}), &["path"]),
        tool(
            "write_file",
            "Write to a file. Will overwrite.",
            json!({"path": {"type": "string"}, "content": {"type": "string"}}),
            &["path", "content"],
        ),
        tool("create_dir", "Create a directory.", json!({"path": {"type": "string"}}), &["path"]),
        tool(
            "run_command",
            "Run a shell command inside the workspace.",
            json!({"command": {"type": "string"}}),
            &["command"],
        ),
        tool("delete_file", "Delete a single file.", json!({"path": {"type": "string"}}), &["path"]),
        tool(
            "delete_dir",
            "Delete a directory and all its contents recursively.",
            json!({"path": {"type": "string"}}),
            &["path"],
        ),
    ]
}

/// Runs a tool call. MCP tools (prefixed `mcp__`) are routed to their server.
pub async fn execute(name: &str, args: &Value, state: &AppState) -> String {
    if name.starts_with("mcp__") {
        let found = state.mcp.lock().await.tools.get(name).cloned();
        return match found {
            Some((client, real)) => client.call_tool(&real, args).await,
            None => format!("Unknown MCP tool: {name}"),
        };
    }

    let ws = {
        let c = state.cfg.lock().unwrap();
        config::workspace_path(&c).display().to_string()
    };
    let s = |k: &str| args[k].as_str().unwrap_or_default().to_string();
    let path = Path::new(&ws).join(s("path"));

    match name {
        "read_file" => std::fs::read_to_string(&path).unwrap_or_else(|e| format!("Error: {e}")),

        "write_file" => {
            let res = path
                .parent()
                .map_or(Ok(()), |p| std::fs::create_dir_all(p))
                .and_then(|_| std::fs::write(&path, s("content")));
            match res {
                Ok(_) => format!("Successfully wrote to {}", s("path")),
                Err(e) => format!("Error: {e}"),
            }
        }

        "create_dir" => match std::fs::create_dir_all(&path) {
            Ok(_) => format!("Created directory {}", s("path")),
            Err(e) => format!("Error: {e}"),
        },

        "run_command" => {
            let command = s("command");
            let mut cmd = if cfg!(windows) {
                let mut c = tokio::process::Command::new("cmd");
                c.args(["/C", command.as_str()]);
                c
            } else {
                let mut c = tokio::process::Command::new("sh");
                c.args(["-c", command.as_str()]);
                c
            };
            cmd.current_dir(&ws).kill_on_drop(true);
            match tokio::time::timeout(Duration::from_secs(120), cmd.output()).await {
                Ok(Ok(out)) => {
                    let text = format!(
                        "{}{}",
                        String::from_utf8_lossy(&out.stdout),
                        String::from_utf8_lossy(&out.stderr)
                    );
                    if text.is_empty() { "Command executed.".to_string() } else { text }
                }
                Ok(Err(e)) => format!("Error: {e}"),
                Err(_) => "Error: command timed out after 120s".to_string(),
            }
        }

        "delete_file" => {
            if path.exists() {
                match std::fs::remove_file(&path) {
                    Ok(_) => format!("Successfully deleted file {}", s("path")),
                    Err(e) => format!("Error: {e}"),
                }
            } else {
                format!("File not found: {}", s("path"))
            }
        }

        "delete_dir" => {
            if path.exists() {
                match std::fs::remove_dir_all(&path) {
                    Ok(_) => format!("Successfully deleted directory {}", s("path")),
                    Err(e) => format!("Error: {e}"),
                }
            } else {
                format!("Directory not found: {}", s("path"))
            }
        }

        other => format!("Unknown tool: {other}"),
    }
}
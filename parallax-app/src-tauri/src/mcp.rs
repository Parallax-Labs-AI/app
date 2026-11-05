use crate::{config, tools::clip, AppState};
use serde::Serialize;
use serde_json::{json, Map, Value};
use std::{
    collections::{BTreeMap, HashMap},
    process::Stdio,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};
use tauri::{AppHandle, Manager, State};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, Command},
    sync::{oneshot, Mutex},
    time::timeout,
};

// ---------- Client ----------

pub struct McpClient {
    stdin: Mutex<ChildStdin>,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>>,
    next_id: AtomicU64,
    _child: Child, // kill_on_drop: the server process ends with the client
    pub tools: Vec<Value>,
}

impl McpClient {
    pub async fn start(argv: &[String], env: &HashMap<String, String>, cwd: &str) -> Result<Self, String> {
        let (program, args) = argv
            .split_first()
            .ok_or_else(|| "The command is empty.".to_string())?;

        let mut child = Command::new(program)
            .args(args)
            .envs(env)
            .current_dir(cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| format!("Could not start '{program}': {e}"))?;

        let stdin = child.stdin.take().ok_or_else(|| "No stdin for MCP server.".to_string())?;
        let stdout = child.stdout.take().ok_or_else(|| "No stdout for MCP server.".to_string())?;

        let pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>> = Default::default();
        let reader_pending = pending.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let Ok(msg) = serde_json::from_str::<Value>(&line) else {
                    continue;
                };
                if let Some(id) = msg.get("id").and_then(Value::as_u64) {
                    let waiter = reader_pending.lock().await.remove(&id);
                    if let Some(tx) = waiter {
                        let _ = tx.send(msg);
                    }
                }
            }
        });

        let mut client = McpClient {
            stdin: Mutex::new(stdin),
            pending,
            next_id: AtomicU64::new(1),
            _child: child,
            tools: vec![],
        };

        client
            .request(
                "initialize",
                json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": { "name": "parallax", "version": "1.0" }
                }),
                60,
            )
            .await?;
        client.notify("notifications/initialized", json!({})).await?;
        let res = client.request("tools/list", json!({}), 30).await?;
        client.tools = res["tools"].as_array().cloned().unwrap_or_default();
        Ok(client)
    }

    async fn send(&self, msg: Value) -> Result<(), String> {
        let mut stdin = self.stdin.lock().await;
        stdin
            .write_all(format!("{msg}\n").as_bytes())
            .await
            .map_err(|e| e.to_string())?;
        stdin.flush().await.map_err(|e| e.to_string())
    }

    async fn notify(&self, method: &str, params: Value) -> Result<(), String> {
        self.send(json!({ "jsonrpc": "2.0", "method": method, "params": params })).await
    }

    async fn request(&self, method: &str, params: Value, secs: u64) -> Result<Value, String> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id, tx);
        self.send(json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }))
            .await?;

        let msg = timeout(Duration::from_secs(secs), rx)
            .await
            .map_err(|_| format!("The MCP server didn't answer {method} in time."))?
            .map_err(|_| "The MCP server closed the connection.".to_string())?;

        if let Some(err) = msg.get("error") {
            return Err(err["message"].as_str().unwrap_or("MCP error").to_string());
        }
        Ok(msg["result"].clone())
    }

    pub async fn call_tool(&self, tool: &str, args: &Value) -> String {
        match self
            .request("tools/call", json!({ "name": tool, "arguments": args }), 120)
            .await
        {
            Ok(res) => {
                let text: Vec<&str> = res["content"]
                    .as_array()
                    .map(|items| {
                        items
                            .iter()
                            .filter(|c| c["type"].as_str() == Some("text"))
                            .filter_map(|c| c["text"].as_str())
                            .collect()
                    })
                    .unwrap_or_default();
                if text.is_empty() { res.to_string() } else { text.join("\n") }
            }
            Err(e) => format!("Error: {e}"),
        }
    }
}

// ---------- Manager ----------

#[derive(Serialize, Clone)]
pub struct McpToolInfo {
    pub server: String,
    pub id: String,          // unique name the model sees, e.g. mcp__web-search__search
    pub name: String,        // the tool's real name on the server
    pub description: String,
}

#[derive(Default)]
pub struct McpManager {
    pub clients: Vec<Arc<McpClient>>,
    pub tools: HashMap<String, (Arc<McpClient>, String)>, // id -> (client, real name)
    pub schemas: Vec<Value>,
    pub catalog: Vec<McpToolInfo>,
    pub status: BTreeMap<String, String>,
}

fn safe_name(n: &str) -> String {
    n.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .take(32)
        .collect()
}

/// Checks the MCP JSON and returns the `mcpServers` map.
pub fn validate(text: &str) -> Result<Map<String, Value>, String> {
    let v: Value = serde_json::from_str(text)
        .map_err(|e| format!("Invalid MCP JSON (line {}): {e}", e.line()))?;
    let servers = v
        .get("mcpServers")
        .and_then(Value::as_object)
        .ok_or_else(|| "The MCP JSON must contain an \"mcpServers\" object.".to_string())?;

    for (name, spec) in servers {
        let has_command = spec
            .get("command")
            .and_then(Value::as_str)
            .map_or(false, |c| !c.trim().is_empty());
        if !has_command {
            return Err(format!("\"{name}\" needs a \"command\" string."));
        }
        if spec.get("args").map_or(false, |a| !a.is_array()) {
            return Err(format!("\"{name}\": \"args\" must be a list."));
        }
        if spec.get("env").map_or(false, |e| !e.is_object()) {
            return Err(format!("\"{name}\": \"env\" must be an object."));
        }
    }
    Ok(servers.clone())
}

/// Validates and pretty-prints the MCP JSON for storage.
pub fn normalize(text: &str) -> Result<String, String> {
    let servers = validate(text)?;
    serde_json::to_string_pretty(&json!({ "mcpServers": servers })).map_err(|e| e.to_string())
}

/// Starts (or restarts) every enabled server from the current config.
pub async fn reload(app: AppHandle) {
    let state = app.state::<AppState>();
    let (text, cwd) = {
        let c = crate::lock(&state.cfg);
        (c.mcp_json.clone(), config::workspace_path(&c).display().to_string())
    };

    let servers = match validate(&text) {
        Ok(s) => s,
        Err(e) => {
            let mut m = state.mcp.lock().await;
            *m = McpManager::default();
            m.status.insert("_config".into(), e);
            return;
        }
    };

    {
        let mut m = state.mcp.lock().await;
        *m = McpManager::default();
        for name in servers.keys() {
            m.status.insert(name.clone(), "connecting".into());
        }
    }

    let mut built = McpManager::default();
    for (name, spec) in &servers {
        if spec.get("disabled").and_then(Value::as_bool) == Some(true) {
            built.status.insert(name.clone(), "disabled".into());
            continue;
        }

        let argv: Vec<String> = std::iter::once(spec["command"].as_str().unwrap_or_default().to_string())
            .chain(
                spec.get("args")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(|a| a.as_str().map(str::to_string)),
            )
            .collect();
        let env: HashMap<String, String> = spec
            .get("env")
            .and_then(Value::as_object)
            .into_iter()
            .flatten()
            .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
            .collect();

        let status = match McpClient::start(&argv, &env, &cwd).await {
            Ok(client) => {
                let client = Arc::new(client);
                let safe = safe_name(name);
                for t in &client.tools {
                    let real = t["name"].as_str().unwrap_or_default().to_string();
                    let description = clip(t["description"].as_str().unwrap_or(""), 1024);
                    let id: String = format!("mcp__{safe}__{real}").chars().take(64).collect();
                    built.tools.insert(id.clone(), (client.clone(), real.clone()));
                    built.catalog.push(McpToolInfo {
                        server: name.clone(),
                        id: id.clone(),
                        name: real.clone(),
                        description: description.clone(),
                    });
                    built.schemas.push(json!({
                        "type": "function",
                        "function": {
                            "name": id,
                            "description": description,
                            "parameters": t
                                .get("inputSchema")
                                .cloned()
                                .unwrap_or_else(|| json!({"type": "object", "properties": {}}))
                        }
                    }));
                }
                let n = client.tools.len();
                built.clients.push(client);
                format!("connected · {n} tools")
            }
            Err(e) => format!("error: {}", clip(&e, 160)),
        };
        built.status.insert(name.clone(), status);
    }

    *state.mcp.lock().await = built;
}

#[tauri::command]
pub async fn list_mcp_tools(state: State<'_, AppState>) -> Result<Vec<McpToolInfo>, String> {
    Ok(state.mcp.lock().await.catalog.clone())
}
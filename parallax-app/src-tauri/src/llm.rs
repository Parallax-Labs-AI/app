use crate::config::Config;
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};

/// llama-server ignores the key, but OpenAI-style clients expect one.
const API_KEY: &str = "not-needed";

pub struct Completion {
    pub content: String,
    pub reasoning: String,
    pub tool_calls: Vec<Value>,
}

/// Streams a chat completion from the local llama-server.
/// `on_token` receives ("content" | "reasoning", text) for each fragment as it arrives.
pub async fn stream_chat(
    cfg: &Config,
    messages: Vec<Value>,
    tools: Option<Vec<Value>>,
    stop: &AtomicBool,
    mut on_token: impl FnMut(&str, &str) + Send,
) -> Result<Completion, String> {
    let mut body = json!({
        "model": cfg.model_id,
        "messages": messages,
        "temperature": cfg.temperature,
        "stream": true,
    });
    if let Some(t) = tools {
        if !t.is_empty() {
            body["tools"] = json!(t);
            body["tool_choice"] = json!("auto");
        }
    }

    let url = format!("http://127.0.0.1:{}/v1/chat/completions", cfg.server_port);
    let resp = reqwest::Client::new()
        .post(&url)
        .bearer_auth(API_KEY)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Cannot reach the model server: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("The model server returned {status}: {text}"));
    }

    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    let mut content = String::new();
    let mut reasoning = String::new();
    // (id, name, arguments) per tool call index
    let mut calls: Vec<(String, String, String)> = Vec::new();

    'outer: while let Some(chunk) = stream.next().await {
        if stop.load(Ordering::SeqCst) {
            break;
        }
        let chunk = chunk.map_err(|e| format!("Lost connection to the model: {e}"))?;
        buf.extend_from_slice(&chunk);

        while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
            let line_bytes: Vec<u8> = buf.drain(..=pos).collect();
            let line = String::from_utf8_lossy(&line_bytes);
            let Some(data) = line.trim().strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data == "[DONE]" {
                break 'outer;
            }
            let Ok(v) = serde_json::from_str::<Value>(data) else {
                continue;
            };
            let delta = &v["choices"][0]["delta"];

            // llama-server sends thinking models' reasoning in its own field.
            let think = delta["reasoning_content"]
                .as_str()
                .or_else(|| delta["reasoning"].as_str());
            if let Some(r) = think {
                if !r.is_empty() {
                    reasoning.push_str(r);
                    on_token("reasoning", r);
                }
            }

            if let Some(t) = delta["content"].as_str() {
                if !t.is_empty() {
                    content.push_str(t);
                    on_token("content", t);
                }
            }

            if let Some(arr) = delta["tool_calls"].as_array() {
                for tc in arr {
                    let idx = tc["index"].as_u64().unwrap_or(0) as usize;
                    if calls.len() <= idx {
                        calls.resize(idx + 1, Default::default());
                    }
                    let slot = &mut calls[idx];
                    if let Some(id) = tc["id"].as_str() {
                        slot.0 = id.to_string();
                    }
                    if let Some(n) = tc["function"]["name"].as_str() {
                        slot.1 = n.to_string();
                    }
                    if let Some(a) = tc["function"]["arguments"].as_str() {
                        slot.2.push_str(a);
                    }
                }
            }
        }
    }

    let tool_calls = calls
        .into_iter()
        .enumerate()
        .filter(|(_, c)| !c.1.is_empty())
        .map(|(i, (id, name, args))| {
            let id = if id.is_empty() { format!("call_{i}") } else { id };
            let args = if args.is_empty() { "{}".to_string() } else { args };
            json!({
                "id": id,
                "type": "function",
                "function": { "name": name, "arguments": args }
            })
        })
        .collect();

    Ok(Completion {
        content,
        reasoning,
        tool_calls,
    })
}
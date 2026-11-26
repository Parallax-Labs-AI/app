use crate::server::Endpoint;
use anyhow::{bail, ensure, Result};
use futures_util::StreamExt;
use reqwest::Client;
use serde::Serialize;
use serde_json::Value;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

#[derive(Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatEvent {
    Started { message: i64 },
    Delta { text: String },
    Reasoning { text: String },
    SearchStatus { status: SearchStatus },
    ToolNotice { text: String },
    AgentApproval { request: crate::agent::Approval },
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchStatus {
    Searching,
    Done,
    Failed,
}

#[derive(Debug)]
pub struct RejectedRequest;

impl std::fmt::Display for RejectedRequest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("The model rejected this conversation. Try a shorter conversation, a larger context, or a different system prompt.")
    }
}

impl std::error::Error for RejectedRequest {}

pub async fn stream(
    client: &Client,
    endpoint: &Endpoint,
    body: Value,
    cancel: &CancellationToken,
    mut on_text: impl FnMut(&str, bool) -> Result<()>,
) -> Result<Vec<ToolCall>> {
    let request = client
        .post(format!("{}/v1/chat/completions", endpoint.base))
        .bearer_auth(&endpoint.secret)
        .json(&body)
        .timeout(Duration::from_secs(15 * 60));
    let response = tokio::select! {
        _ = cancel.cancelled() => return Ok(Vec::new()),
        response = request.send() => response.map_err(|_| anyhow::anyhow!("Could not reach the local model. Reload it and retry."))?,
    };
    if response.status() == reqwest::StatusCode::BAD_REQUEST {
        return Err(RejectedRequest.into());
    }
    ensure!(
        response.status().is_success(),
        "Local inference returned HTTP {}. Reload the model and retry.",
        response.status()
    );
    let mut stream = response.bytes_stream();
    let mut sse = Sse {
        text_tool: body["tools"].as_array().map(|tools| TextToolCall {
            names: tools
                .iter()
                .filter_map(|tool| tool["function"]["name"].as_str().map(str::to_owned))
                .collect(),
            ..TextToolCall::default()
        }),
        ..Sse::default()
    };
    loop {
        let chunk = tokio::select! {
            _ = cancel.cancelled() => return Ok(Vec::new()),
            next = stream.next() => next,
        };
        let Some(chunk) = chunk else {
            break;
        };
        let chunk = chunk.map_err(|_| {
            anyhow::anyhow!("The local model stopped responding. Your partial reply was saved.")
        })?;
        if sse.push(&chunk, &mut on_text)? {
            return Ok(sse.into_calls());
        }
    }
    bail!("The model stopped before finishing the reply. Try regenerating it.")
}

#[derive(Default)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

impl ToolCall {
    pub fn message(&self) -> Value {
        serde_json::json!({"id":self.id,"type":"function","function":{"name":self.name,"arguments":self.arguments}})
    }
}

#[derive(Default)]
struct Sse {
    pending: Vec<u8>,
    tags: ThinkStream,
    calls: std::collections::BTreeMap<u64, ToolCall>,
    text_tool: Option<TextToolCall>,
}

impl Sse {
    fn into_calls(self) -> Vec<ToolCall> {
        if self.calls.is_empty() {
            return self
                .text_tool
                .and_then(|tool| tool.call)
                .into_iter()
                .collect();
        }
        self.calls.into_values().collect()
    }

    fn push(
        &mut self,
        bytes: &[u8],
        on_text: &mut impl FnMut(&str, bool) -> Result<()>,
    ) -> Result<bool> {
        self.pending.extend_from_slice(bytes);
        ensure!(
            self.pending.len() <= 4 * 1024 * 1024,
            "Unexpectedly large inference event."
        );
        let mut consumed = 0;
        for line in self.pending.split_inclusive(|byte| *byte == b'\n') {
            if !line.ends_with(b"\n") {
                break;
            }
            consumed += line.len();
            let line = std::str::from_utf8(line)?.trim_end_matches(['\r', '\n']);
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            let mut emit = |text: &str, reasoning: bool| match &mut self.text_tool {
                Some(tool) if !reasoning => tool.push(text, on_text),
                _ => on_text(text, reasoning),
            };
            if data == "[DONE]" {
                self.tags.flush(&mut emit)?;
                if let Some(tool) = &mut self.text_tool {
                    tool.finish(on_text)?;
                }
                return Ok(true);
            }
            let event: Value = serde_json::from_str(data)?;
            ensure!(
                event.get("error").is_none(),
                "The local model reported a generation error."
            );
            let delta = &event["choices"][0]["delta"];
            if let Some(calls) = delta["tool_calls"].as_array() {
                for next in calls {
                    let index = next["index"].as_u64().unwrap_or(0);
                    ensure!(index < 4, "Too many tool calls.");
                    let call = self.calls.entry(index).or_default();
                    if let Some(id) = next["id"].as_str() {
                        call.id.push_str(id);
                    }
                    if let Some(name) = next["function"]["name"].as_str() {
                        call.name.push_str(name);
                    }
                    if let Some(arguments) = next["function"]["arguments"].as_str() {
                        call.arguments.push_str(arguments);
                    }
                    ensure!(
                        call.id.len() <= 128
                            && call.name.len() <= 128
                            && call.arguments.len() <= 32_768,
                        "Model tool call exceeded its size limit."
                    );
                }
            }
            if let Some(text) = delta["reasoning_content"]
                .as_str()
                .filter(|text| !text.is_empty())
            {
                emit(text, true)?;
            }
            if let Some(text) = delta["content"].as_str() {
                self.tags.push(text, &mut emit)?;
            }
        }
        self.pending.drain(..consumed);
        Ok(false)
    }
}

// Tool markup can arrive in delta.content. Inspect only the start of the
// answer and only recognize tool names advertised on this request.
const TOOL_OPEN: &str = "<tool_call>";
const TOOL_CLOSE: &str = "</tool_call>";
const FUNCTION_OPEN: &str = "<function=";

struct TextToolCall {
    pending: Option<String>,
    call: Option<ToolCall>,
    names: Vec<String>,
}

impl Default for TextToolCall {
    fn default() -> Self {
        Self {
            pending: Some(String::new()),
            call: None,
            names: vec!["search".into()],
        }
    }
}

impl TextToolCall {
    fn push(&mut self, text: &str, emit: &mut impl FnMut(&str, bool) -> Result<()>) -> Result<()> {
        if self.call.is_some() {
            return Ok(());
        }
        let Some(pending) = &mut self.pending else {
            return emit(text, false);
        };
        pending.push_str(text);
        let start = pending.trim_start();
        let function = start.strip_prefix(TOOL_OPEN).map(str::trim_start);
        let header = function.and_then(|text| text.strip_prefix(FUNCTION_OPEN));
        if let Some((name, parameters)) = header
            .and_then(|text| text.split_once('>'))
            .filter(|(name, _)| self.names.iter().any(|allowed| allowed == name))
        {
            let captured = if let Some((parameters, _)) = parameters.split_once(TOOL_CLOSE) {
                Some(Self::parameters(name, parameters))
            } else {
                (pending.len() > 32_768).then_some(Value::Null)
            };
            if let Some(arguments) = captured {
                let name = name.to_owned();
                self.capture(name, arguments);
            }
        } else if pending.len() > 128
            || !function.map_or_else(
                || TOOL_OPEN.starts_with(start),
                |text| {
                    FUNCTION_OPEN.starts_with(text)
                        || header.is_some_and(|prefix| {
                            self.names.iter().any(|name| name.starts_with(prefix))
                        })
                },
            )
        {
            let answer = std::mem::take(pending);
            self.pending = None;
            emit(&answer, false)?;
        }
        Ok(())
    }

    fn finish(&mut self, emit: &mut impl FnMut(&str, bool) -> Result<()>) -> Result<()> {
        if let Some(pending) = self.pending.take() {
            let name = pending
                .trim_start()
                .strip_prefix(TOOL_OPEN)
                .and_then(|text| text.trim_start().strip_prefix(FUNCTION_OPEN))
                .and_then(|text| text.split_once('>'))
                .map(|(name, _)| name);
            if let Some(name) = name.filter(|name| self.names.iter().any(|allowed| allowed == name))
            {
                self.capture(name.into(), Value::Null);
            } else if !pending.is_empty() {
                emit(&pending, false)?;
            }
        }
        Ok(())
    }

    fn capture(&mut self, name: String, arguments: Value) {
        self.pending = None;
        let arguments = if arguments.is_null() {
            serde_json::json!({})
        } else {
            arguments
        };
        self.call = Some(ToolCall {
            id: format!("call_{}", uuid::Uuid::new_v4().simple()),
            name,
            arguments: arguments.to_string(),
        });
    }

    fn parameters(name: &str, text: &str) -> Value {
        let text = text.trim();
        let text = text.strip_suffix("</function>").unwrap_or(text).trim_end();
        if name == "search" {
            let Some(query) = text.strip_prefix("<parameter=query>") else {
                return Value::Null;
            };
            let query = query
                .trim()
                .strip_suffix("</parameter>")
                .unwrap_or(query.trim())
                .trim();
            if query.contains(['<', '>']) || query.len() > 302 {
                return Value::Null;
            }
            return serde_json::json!({"query":query.trim_matches('"').trim()});
        }
        if name == "move_file" || name == "run_configured_command" || name.starts_with("mcp_") {
            let mut args = serde_json::Map::new();
            let mut rest = text;
            while !rest.trim().is_empty() {
                let Some((key, next)) = rest
                    .trim_start()
                    .strip_prefix("<parameter=")
                    .and_then(|text| text.split_once('>'))
                else {
                    return Value::Null;
                };
                let Some((value, next)) = next.split_once("</parameter>") else {
                    return Value::Null;
                };
                let value = value.trim();
                let value = if name.starts_with("mcp_") {
                    serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.into()))
                } else {
                    Value::String(value.into())
                };
                if key.is_empty() || args.insert(key.into(), value).is_some() {
                    return Value::Null;
                }
                rest = next;
            }
            return Value::Object(args);
        }
        if name == "workspace_command" {
            let mut args = serde_json::Map::new();
            let mut rest = text;
            for _ in 0..2 {
                let Some((key, next)) = rest
                    .trim_start()
                    .strip_prefix("<parameter=")
                    .and_then(|text| text.split_once('>'))
                else {
                    return Value::Null;
                };
                let Some((value, next)) = next.split_once("</parameter>") else {
                    return Value::Null;
                };
                if !matches!(key, "command" | "path")
                    || args
                        .insert(key.into(), serde_json::json!(value.trim()))
                        .is_some()
                {
                    return Value::Null;
                }
                rest = next;
            }
            return if rest.trim().is_empty() {
                Value::Object(args)
            } else {
                Value::Null
            };
        }
        let Some(text) = text.strip_prefix("<parameter=path>") else {
            return Value::Null;
        };
        if !matches!(name, "write_file" | "create_file") {
            let path = text
                .trim()
                .strip_suffix("</parameter>")
                .unwrap_or(text.trim())
                .trim();
            return if !path.contains(['<', '>']) {
                serde_json::json!({"path":path})
            } else {
                Value::Null
            };
        }
        let Some((path, rest)) = text.split_once("</parameter>") else {
            return Value::Null;
        };
        let Some(content) = rest
            .trim_start()
            .strip_prefix("<parameter=content>")
            .and_then(|text| text.trim_end().strip_suffix("</parameter>"))
        else {
            return Value::Null;
        };
        serde_json::json!({"path":path.trim(),"content":content.strip_prefix('\n').unwrap_or(content)})
    }
}

pub async fn conversation_tokens(
    client: &Client,
    endpoint: &Endpoint,
    content: &str,
) -> Result<usize> {
    let tokens: Value = client
        .post(format!("{}/tokenize", endpoint.base))
        .bearer_auth(&endpoint.secret)
        .timeout(Duration::from_secs(10))
        .json(&serde_json::json!({"content":content,"add_special":false,"parse_special":false}))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    Ok(tokens["tokens"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("The model returned no tokens."))?
        .len())
}

#[derive(Default)]
struct ThinkStream {
    pending: String,
    phase: ThinkPhase,
}

#[derive(Default, PartialEq)]
enum ThinkPhase {
    #[default]
    Prefix,
    Thinking,
    Answer,
}

impl ThinkStream {
    fn push(&mut self, text: &str, emit: &mut impl FnMut(&str, bool) -> Result<()>) -> Result<()> {
        if self.phase == ThinkPhase::Answer {
            return emit(text, false);
        }
        self.pending.push_str(text);
        if self.phase == ThinkPhase::Prefix {
            let start = self.pending.trim_start();
            if let Some(rest) = start.strip_prefix("<think>") {
                self.pending = rest.to_owned();
                self.phase = ThinkPhase::Thinking;
            } else if "<think>".starts_with(start) && self.pending.len() <= 64 {
                return Ok(());
            } else {
                self.phase = ThinkPhase::Answer;
                return self.flush(emit);
            }
        }
        const CLOSE: &str = "</think>";
        if let Some((thought, answer)) = self.pending.split_once(CLOSE) {
            emit(thought, true)?;
            emit(answer, false)?;
            self.pending.clear();
            self.phase = ThinkPhase::Answer;
        } else {
            let keep = (1..CLOSE.len())
                .rev()
                .find(|&length| self.pending.ends_with(&CLOSE[..length]))
                .unwrap_or(0);
            let end = self.pending.len() - keep;
            emit(&self.pending[..end], true)?;
            self.pending.drain(..end);
        }
        Ok(())
    }

    fn flush(&mut self, emit: &mut impl FnMut(&str, bool) -> Result<()>) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        emit(
            &std::mem::take(&mut self.pending),
            self.phase == ThinkPhase::Thinking,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_the_split_search_markup_seen_in_chat() {
        let mut sse = Sse {
            text_tool: Some(TextToolCall::default()),
            ..Sse::default()
        };
        let mut output = String::new();
        let mut reasoning = String::new();
        let mut emit = |text: &str, thought| {
            if thought {
                reasoning.push_str(text);
            } else {
                output.push_str(text);
            }
            Ok(())
        };
        for chunk in [
            "<think>Check current reports.</think>\n<tool_",
            "call> <function=se",
            "arch> <parameter=query> MD-11 plane crash Miami recent news\" </tool",
            "_call>",
        ] {
            let event = serde_json::json!({"choices":[{"delta":{"content":chunk}}]});
            assert!(!sse
                .push(format!("data: {event}\n").as_bytes(), &mut emit)
                .unwrap());
        }
        assert!(sse.push(b"data: [DONE]\n", &mut emit).unwrap());
        let calls = sse.into_calls();
        assert_eq!(reasoning, "Check current reports.");
        assert!(output.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        assert_eq!(
            crate::search::query(&calls[0].arguments).unwrap(),
            "MD-11 plane crash Miami recent news"
        );
    }

    #[test]
    fn raw_search_requires_a_complete_envelope_and_keeps_markdown_examples_literal() {
        let complete = "<tool_call>\n<function=search>\n<parameter=query>\nApollo 11\n</parameter>\n</function>\n</tool_call>";
        let mut enabled = TextToolCall::default();
        let mut literal = TextToolCall::default();
        let example = format!("```xml\n{complete}\n```");
        enabled
            .push(complete, &mut |_, _| panic!("Tool markup leaked"))
            .unwrap();
        assert_eq!(
            crate::search::query(&enabled.call.unwrap().arguments).unwrap(),
            "Apollo 11"
        );
        let mut output = String::new();
        literal
            .push(&example, &mut |text, _| {
                output.push_str(text);
                Ok(())
            })
            .unwrap();
        assert!(literal.call.is_none());
        assert_eq!(output, example);

        let mut incomplete = TextToolCall::default();
        incomplete
            .push(complete.trim_end_matches(TOOL_CLOSE), &mut |_, _| {
                panic!("Tool markup leaked")
            })
            .unwrap();
        incomplete
            .finish(&mut |_, _| panic!("Tool markup leaked"))
            .unwrap();
        assert!(crate::search::query(&incomplete.call.unwrap().arguments).is_err());
    }

    #[test]
    fn streamed_tool_call_arguments_are_assembled() {
        let mut sse = Sse::default();
        for delta in [
            serde_json::json!({"index":0,"id":"call_1","function":{"name":"search","arguments":"{\"query\":"}}),
            serde_json::json!({"index":0,"function":{"arguments":"\"Apollo 11\"}"}}),
        ] {
            let event = serde_json::json!({"choices":[{"delta":{"tool_calls":[delta]}}]});
            sse.push(format!("data: {event}\n").as_bytes(), &mut |_, _| Ok(()))
                .unwrap();
        }
        let call = &sse.calls[&0];
        assert_eq!(call.name, "search");
        assert_eq!(crate::search::query(&call.arguments).unwrap(), "Apollo 11");
    }

    #[test]
    fn streaming_reassembles_chunked_utf8_and_separates_reasoning() {
        let mut out = String::new();
        let mut thought = String::new();
        let bytes = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"quiet\",\"content\":\"héllo\"}}]}\r\n\r\ndata: [DONE]\n".as_bytes();
        let mut sse = Sse::default();
        let mut callback = |delta: &str, reasoning| {
            if reasoning {
                thought.push_str(delta);
            } else {
                out.push_str(delta);
            }
            Ok(())
        };
        for byte in &bytes[..bytes.len() - 1] {
            assert!(!sse.push(&[*byte], &mut callback).unwrap());
        }
        assert!(sse.push(&bytes[bytes.len() - 1..], &mut callback).unwrap());
        assert_eq!(out, "héllo");
        assert_eq!(thought, "quiet");
    }

    #[test]
    fn split_think_tags_never_leak_to_the_answer() {
        let mut tags = ThinkStream::default();
        let mut answer = String::new();
        let mut reasoning = String::new();
        let mut emit = |text: &str, thought| {
            if thought {
                reasoning.push_str(text);
            } else {
                answer.push_str(text);
            }
            Ok(())
        };
        for chunk in ["\n<thi", "nk>plan", "ning</th", "ink>Answer.", " `<think>`"] {
            tags.push(chunk, &mut emit).unwrap();
        }
        tags.flush(&mut emit).unwrap();
        assert_eq!(reasoning, "planning");
        assert_eq!(answer, "Answer. `<think>`");
    }
}

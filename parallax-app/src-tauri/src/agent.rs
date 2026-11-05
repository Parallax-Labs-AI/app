use crate::{history, llm, lock, runtime, tools::{self, clip}, AppState};
use serde_json::{json, Value};
use std::{
    collections::HashSet,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
use tauri::{AppHandle, Emitter, Manager, State};
use tokio::sync::oneshot;

const MAX_STEPS: usize = 25;
const DESTRUCTIVE: [&str; 2] = ["delete_file", "delete_dir"];
const DENIED_REPLY: &str = "Understood. I won't run that tool. Tell me how you'd like to continue.";
const NEW_PROJECT_NOTE: &str = "\n\nThis is a new project. Before writing code, ask the user for a project name and a directory name. Once they reply, use create_dir and then start building.";
const EXTENDED_THINKING_NOTE: &str = r#"

## Extended thinking mode (ENABLED)
The user has turned on Extended Thinking. Think as much as possible before
answering: reason step by step at length, explore multiple approaches, weigh
trade-offs, and consider edge cases before settling on a conclusion. Do not
rush, and do not shorten your reasoning to save tokens — thoroughness is the
priority in this mode. This applies to every reply, and especially to coding
work: reason through the design, the failure modes, and the verification steps
before and while writing code."#;

/// One open conversation. Kept in memory while the app runs and saved to disk after each turn.
pub struct Conv {
    messages: Vec<Value>,
    mcp_disabled: Vec<String>,
    model: String,
    extended_thinking: bool,
    allowed: HashSet<String>,
}

impl Conv {
    fn load(kind: &str, id: &str) -> Conv {
        let saved = history::load(kind, id);
        let field = |k: &str| saved.as_ref().and_then(|v| v.get(k));
        let messages: Vec<Value> = field("messages")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mcp_disabled: Vec<String> = field("mcp_disabled")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
            .unwrap_or_default();
        let model = field("model").and_then(Value::as_str).unwrap_or_default().to_string();
        let extended_thinking = field("extended_thinking").and_then(Value::as_bool).unwrap_or(false);
        Conv { messages, mcp_disabled, model, extended_thinking, allowed: HashSet::new() }
    }
}

fn with_conv<R>(state: &AppState, kind: &str, id: &str, f: impl FnOnce(&mut Conv) -> R) -> R {
    let mut convs = lock(&state.convs);
    let conv = convs.entry(id.to_string()).or_insert_with(|| Conv::load(kind, id));
    f(conv)
}

fn emit(app: &AppHandle, event: &str, payload: Value) {
    let _ = app.emit(event, payload);
}

fn system_prompt(kind: &str) -> String {
    let os = match std::env::consts::OS {
        "windows" => "Windows (use dir, del, mkdir, type, powershell, cmd)",
        "macos"   => "macOS (use ls, rm, mkdir, cat, bash, zsh)",
        _         => "Linux (use ls, rm, mkdir, cat, bash, sh)",
    };

    if kind == "agent" {
        format!(
r#"You are Parallax, an autonomous senior software engineer built by Parallax Labs.
You operate locally on the user's machine ({os}) with direct access to the filesystem,
a shell, and a suite of tools. You are not a chatbot in this mode — you are a colleague
who plans, executes, verifies, and reports.

## Operating principles
1. Understand before acting. Read the relevant files, list directories, and inspect
   existing patterns before writing or modifying anything. Never guess at file contents
   or project structure when a tool can tell you.
2. Plan, then execute. For any non-trivial task, outline a short plan (internally or in
   your reply), then carry it out step by step. Prefer small, verifiable increments over
   giant blind rewrites.
3. Verify your work. After writing or changing code, run the project's tests, linters,
   type-checkers, or build commands when available. Read the output. If something fails,
   diagnose and fix it before reporting success. Never claim a task is done if you have
   not verified it.
4. Match the codebase. Follow existing style, naming, structure, dependencies, and
   conventions. Do not introduce new frameworks, packages, or patterns unless the task
   requires it and the user would expect it.
5. Be surgical. Change only what is needed. Avoid unrelated refactors, formatting churn,
   or speculative "improvements" that were not requested.
6. Leave the workspace clean. No stray debug prints, no commented-out dead code, no
   temporary files unless the user asked for them.

## Tools and how to use them
- read_file / list_dir: inspect before editing. Use them liberally.
- write_file: create or overwrite files. This is how you deliver code — never paste
  full file contents into chat as a substitute for writing them.
- run_command: execute shell commands (build, test, lint, git, package managers).
  Prefer non-destructive commands; ask before anything irreversible (rm -rf, force push,
  database drops, global installs).
- git_*: use for status, diff, log, branch, commit. Do not commit or push unless asked.
- run_tests: run the project's test suite and read the results.
If a tool you need is not available, say so clearly and propose the closest alternative.

## Workflow for a typical task
1. Inspect: list the project, read the files involved, understand the current state.
2. Plan: decide the minimal set of changes.
3. Edit: write or modify files with tools.
4. Verify: run tests/build/lint; read errors; fix; repeat until green.
5. Report: summarize what changed, why, and how you verified it. Mention any remaining
   risks or follow-ups.

## Communication
- In chat replies, be concise and factual. Do not dump full file contents into the reply;
  the user can see the files you wrote. Summarize diffs and reasoning instead.
- Do not put code blocks in your chat reply when you have already written the code with
  tools. Use chat for explanation, not duplication.
- If a task is ambiguous, state your assumption and proceed — or ask one focused question
  if the ambiguity is genuinely blocking.
- If you are blocked, say exactly what is blocking you and what you tried.

## Safety and boundaries
- Never exfiltrate data, embed secrets, or run commands that could damage the user's system
  without explicit confirmation.
- Do not install global packages, modify system configuration, or change permissions unless
  the task clearly requires it.
- Treat the user's repository and machine as production. Be careful.

Operating system: {os}"#
        )
    } else {
        format!(
r#"You are Parallax, a helpful, knowledgeable AI assistant built by Parallax Labs.
You are clear, accurate, and genuinely useful. You can help with coding, reasoning,
writing, analysis, planning, and general questions — and you should do all of them well.

## Core behavior
- Be accurate. If you are unsure, say so and explain what you do and don't know. Never
  invent facts, APIs, citations, or file contents.
- Be clear. Prefer plain language, concrete examples, and well-structured answers. Use
  headings, lists, and code blocks when they genuinely help.
- Be useful. Answer the question the user actually asked, then add helpful context only
  when it adds value. Avoid filler, flattery, and unnecessary preamble.
- Be honest about limits. If a task is outside your knowledge or capabilities, say so and
  suggest what would help.

## Coding
When writing or explaining code:
- Produce correct, runnable, idiomatic code in the requested language and framework.
- Follow the conventions of the language and the existing project when context is given.
- Include error handling, edge cases, and brief comments where they matter.
- Explain non-obvious decisions. Do not over-explain trivia.
- When debugging, identify the root cause before proposing a fix. Show the reasoning.
- Prefer complete, working examples over fragments when the user needs to run something.

## Reasoning
For analytical, mathematical, or planning questions:
- Break the problem into parts, state assumptions, and work through the logic.
- Consider alternatives and edge cases before settling on an answer.
- Distinguish between what is certain, what is likely, and what is speculative.
- If the problem is underspecified, state the assumption you're making or ask.

## Tools
Use tools only when they genuinely help the user — e.g., reading a file they reference,
running a command they asked for, or looking something up. Do not use tools to show off,
and do not claim to have used a tool you did not use.

## Communication style
- Match the user's level of detail and tone.
- Be direct. Skip "Great question!" and similar filler.
- For long answers, lead with the conclusion or the key point, then the details.
- Use formatting to aid comprehension, not to decorate.

Operating system: {os}"#
        )
    }
}

fn tool_msg(id: &str, name: &str, content: &str) -> Value {
    json!({ "role": "tool", "tool_call_id": id, "name": name, "content": content })
}

fn valid_session_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 64 && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn parse_kind(kind: &str) -> Result<&'static str, String> {
    match kind {
        "chat" => Ok("chat"),
        "agent" => Ok("agent"),
        _ => Err("Unknown mode.".into()),
    }
}

// ---------- Sessions ----------

#[tauri::command]
pub fn list_sessions(kind: String) -> Vec<history::SessionMeta> {
    history::list(&kind)
}

/// Returns a chat's messages, its MCP settings, and the model it was started with.
#[tauri::command]
pub fn open_session(kind: String, id: String, state: State<AppState>) -> Result<Value, String> {
    let kind = parse_kind(&kind)?;
    if !valid_session_id(&id) {
        return Err("Invalid chat id.".into());
    }
    Ok(with_conv(&state, kind, &id, |c| {
        json!({ "messages": c.messages, "mcp_disabled": c.mcp_disabled, "model": c.model, "extended_thinking": c.extended_thinking })
    }))
}

#[tauri::command]
pub fn delete_session(kind: String, id: String, state: State<AppState>) -> Result<(), String> {
    if lock(&state.running).contains_key(&id) {
        return Err("This chat is still responding.".into());
    }
    lock(&state.convs).remove(&id);
    let _ = history::delete(&kind, &id);
    Ok(())
}

// ---------- Turns ----------

/// Sends a user message and starts the reply in the background.
/// Progress arrives as events: stream-start, token, stream-end, log, approval-request, turn-end.
#[tauri::command]
pub async fn send_message(
    app: AppHandle,
    kind: String,
    session_id: String,
    text: String,
    mcp_disabled: Vec<String>,
    extended_thinking: bool,
) -> Result<(), String> {
    let kind = parse_kind(&kind)?;
    let text = text.trim().to_string();
    if text.is_empty() {
        return Err("Type a message first.".into());
    }
    if !valid_session_id(&session_id) {
        return Err("Invalid chat id.".into());
    }
    if !runtime::is_ready(&app).await {
        return Err("The model isn't ready. Select a model and wait for it to load.".into());
    }

    let state = app.state::<AppState>();
    let cancel = Arc::new(AtomicBool::new(false));
    {
        let mut running = lock(&state.running);
        if running.contains_key(&session_id) {
            return Err("This chat is still responding.".into());
        }
        running.insert(session_id.clone(), cancel.clone());
    }

    let model = {
        let c = lock(&state.cfg);
        format!("{}|{}", c.model_id, c.quant)
    };
    with_conv(&state, kind, &session_id, |c| {
        c.mcp_disabled = mcp_disabled;
        c.extended_thinking = extended_thinking;
        if c.model.is_empty() {
            c.model = model;
        }
        c.messages.push(json!({ "role": "user", "content": text }));
    });

    let app2 = app.clone();
    tauri::async_runtime::spawn(async move {
        run_turn(app2, kind, session_id, cancel).await;
    });
    Ok(())
}

#[tauri::command]
pub fn stop_message(session_id: String, state: State<AppState>) {
    if let Some(flag) = lock(&state.running).get(&session_id) {
        flag.store(true, Ordering::SeqCst);
    }
    // If the reply is waiting on a tool approval, treat stop as "no".
    if let Some(tx) = lock(&state.approval).take() {
        let _ = tx.send((false, false));
    }
}

#[tauri::command]
pub fn approve_tool(approved: bool, always_allow: bool, state: State<AppState>) {
    if let Some(tx) = lock(&state.approval).take() {
        let _ = tx.send((approved, always_allow));
    }
}

async fn run_turn(app: AppHandle, kind: &'static str, id: String, cancel: Arc<AtomicBool>) {
    let state = app.state::<AppState>();
    if let Err(e) = turn_loop(&app, &state, kind, &id, &cancel).await {
        emit(&app, "turn-error", json!({ "session": id, "error": e }));
    }
    if let Err(e) = save_conv(&state, kind, &id) {
        emit(&app, "turn-error", json!({ "session": id, "error": e }));
    }
    lock(&state.running).remove(&id);
    emit(&app, "turn-end", json!({ "session": id }));
}

async fn turn_loop(
    app: &AppHandle,
    state: &AppState,
    kind: &'static str,
    id: &str,
    cancel: &AtomicBool,
) -> Result<(), String> {
    let cfg = lock(&state.cfg).clone();

    for _ in 0..MAX_STEPS {
        if cancel.load(Ordering::SeqCst) {
            return Ok(());
        }

        let (messages, disabled) = with_conv(state, kind, id, |c| {
            let mut system = system_prompt(kind);
            if kind == "agent" && c.messages.len() <= 1 {
                system.push_str(NEW_PROJECT_NOTE);
            }
            let mut msgs = vec![json!({ "role": "system", "content": system })];
            msgs.extend(c.messages.iter().cloned());
            (msgs, c.mcp_disabled.clone())
        });

        // Built-in file tools are for agent mode only. MCP tools work in both modes,
        // minus any the user switched off for this chat.
        let mut tool_list: Vec<Value> = if kind == "agent" { tools::schemas() } else { Vec::new() };
        let mcp: Vec<Value> = state
            .mcp
            .lock()
            .await
            .schemas
            .iter()
            .filter(|s| {
                let name = s["function"]["name"].as_str();
                !disabled.iter().any(|d| Some(d.as_str()) == name)
            })
            .cloned()
            .collect();
        tool_list.extend(mcp);
        let tools_opt = if tool_list.is_empty() { None } else { Some(tool_list) };

        emit(app, "stream-start", json!({ "session": id }));
        let tok_app = app.clone();
        let sid = id.to_string();
        let result = llm::stream_chat(&cfg, messages, tools_opt, cancel, move |part, text| {
            let _ = tok_app.emit("token", json!({ "session": sid, "kind": part, "token": text }));
        })
        .await;
        emit(app, "stream-end", json!({ "session": id }));
        let completion = result?;

        // If a model put its whole reply in the reasoning field, keep that text as the answer.
        let stored = if completion.content.is_empty() {
            completion.reasoning.clone()
        } else {
            completion.content.clone()
        };
        let content_val = if stored.is_empty() {
            Value::Null
        } else {
            Value::String(stored)
        };
        let mut assistant = json!({ "role": "assistant", "content": content_val });
        if !completion.tool_calls.is_empty() {
            assistant["tool_calls"] = json!(completion.tool_calls.clone());
        }
        with_conv(state, kind, id, |c| c.messages.push(assistant));

        if completion.tool_calls.is_empty() {
            if completion.content.is_empty() && completion.reasoning.is_empty() {
                emit(
                    app,
                    "log",
                    json!({ "session": id, "kind": "error", "text": "The model returned an empty reply." }),
                );
            }
            return Ok(());
        }

        let mut denied = false;
        for tc in &completion.tool_calls {
            let call_id = tc["id"].as_str().unwrap_or_default().to_string();
            let name = tc["function"]["name"].as_str().unwrap_or_default().to_string();

            if denied || cancel.load(Ordering::SeqCst) {
                let why = if denied {
                    "Skipped: an earlier tool call was denied."
                } else {
                    "Skipped: the response was stopped."
                };
                with_conv(state, kind, id, |c| c.messages.push(tool_msg(&call_id, &name, why)));
                continue;
            }

            let args: Value = tc["function"]["arguments"]
                .as_str()
                .and_then(|s| serde_json::from_str::<Value>(s).ok())
                .unwrap_or_else(|| json!({}));
            let destructive = DESTRUCTIVE.contains(&name.as_str());
            let pre_allowed = !destructive && with_conv(state, kind, id, |c| c.allowed.contains(&name));

            if !pre_allowed {
                let (tx, rx) = oneshot::channel::<(bool, bool)>();
                *lock(&state.approval) = Some(tx);
                emit(
                    app,
                    "approval-request",
                    json!({
                        "session": id,
                        "id": call_id,
                        "name": name,
                        "args": args.clone(),
                        "destructive": destructive,
                    }),
                );
                let (approved, always) = rx.await.unwrap_or((false, false));

                if !approved {
                    emit(app, "log", json!({ "session": id, "kind": "error", "text": format!("Denied: {name}") }));
                    with_conv(state, kind, id, |c| {
                        c.messages.push(tool_msg(&call_id, &name, "User denied permission to run this tool."))
                    });
                    denied = true;
                    continue;
                }
                if always && !destructive {
                    with_conv(state, kind, id, |c| c.allowed.insert(name.clone()));
                }
            }

            emit(app, "log", json!({ "session": id, "kind": "system", "text": format!("Running {name}…") }));
            let result = tools::execute(&name, &args, state).await;
            emit(app, "log", json!({ "session": id, "kind": "result", "text": clip(&result, 1000) }));
            let stored = clip(&result, 8000);
            with_conv(state, kind, id, |c| c.messages.push(tool_msg(&call_id, &name, &stored)));
        }

        if denied {
            emit(app, "stream-start", json!({ "session": id }));
            emit(app, "token", json!({ "session": id, "kind": "content", "token": DENIED_REPLY }));
            emit(app, "stream-end", json!({ "session": id }));
            with_conv(state, kind, id, |c| {
                c.messages.push(json!({ "role": "assistant", "content": DENIED_REPLY }))
            });
            return Ok(());
        }
    }

    Err("Stopped after too many tool steps. Ask again with a smaller goal.".into())
}

fn save_conv(state: &AppState, kind: &str, id: &str) -> Result<(), String> {
    let (messages, disabled, model) = with_conv(state, kind, id, |c| {
        (c.messages.clone(), c.mcp_disabled.clone(), c.model.clone())
    });
    if messages.is_empty() {
        return Ok(());
    }
    let title = messages
        .iter()
        .find(|m| m["role"].as_str() == Some("user"))
        .and_then(|m| m["content"].as_str())
        .map(|s| clip(s.trim(), 40))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "Untitled".to_string());
    history::save(
        kind,
        id,
        &json!({ "title": title, "model": model, "mcp_disabled": disabled, "messages": messages }),
    )
}
// Prevents an extra console window on Windows release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod agent;
mod config;
mod hardware;
mod history;
mod llm;
mod mcp;
mod runtime;
mod tools;

use config::{Config, SettingsPatch};
use mcp::McpManager;
use serde::Serialize;
use std::{
    collections::{BTreeMap, HashMap},
    sync::{atomic::AtomicBool, Arc, Mutex, MutexGuard},
};
use tauri::{AppHandle, Manager, State};
use tokio::sync::oneshot;

/// Locks a std mutex and recovers from poisoning instead of panicking.
pub fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

pub struct AppState {
    pub cfg: Mutex<Config>,
    pub convs: Mutex<HashMap<String, agent::Conv>>,
    pub running: Mutex<HashMap<String, Arc<AtomicBool>>>, // session id -> cancel flag
    pub approval: Mutex<Option<oneshot::Sender<(bool, bool)>>>, // (approved, always_allow)
    pub engine: Mutex<runtime::EngineStatus>,
    pub server: tokio::sync::Mutex<Option<runtime::ServerProc>>,
    pub runtime_install: tokio::sync::Mutex<()>,
    pub model_download: Mutex<Option<Arc<AtomicBool>>>,
    pub mcp: tokio::sync::Mutex<McpManager>,
}

/// Errors if a reply is running. Used before anything that would stop the server.
pub fn ensure_idle(state: &AppState) -> Result<(), String> {
    if lock(&state.running).is_empty() {
        Ok(())
    } else {
        Err("Wait for the current response to finish first.".into())
    }
}

#[derive(Serialize)]
struct ConfigDto {
    #[serde(flatten)]
    config: Config,
    mcp_status: BTreeMap<String, String>,
}

async fn snapshot(state: &AppState) -> ConfigDto {
    let config = lock(&state.cfg).clone();
    let mcp_status = state.mcp.lock().await.status.clone();
    ConfigDto { config, mcp_status }
}

// ---------- Settings ----------

#[tauri::command]
async fn get_config(state: State<'_, AppState>) -> Result<ConfigDto, String> {
    Ok(snapshot(&state).await)
}

#[tauri::command]
async fn save_config(
    mut patch: SettingsPatch,
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<ConfigDto, String> {
    if let Some(raw) = patch.mcp_json.take() {
        patch.mcp_json = Some(mcp::normalize(&raw)?);
    }
    let server_changed = patch.affects_server(&lock(&state.cfg));
    if server_changed {
        ensure_idle(&state)?;
    }
    let mcp_changed = patch.mcp_json.is_some();

    // Apply to a copy first, so an invalid value changes nothing.
    let mut next = lock(&state.cfg).clone();
    patch.apply(&mut next)?;
    config::save(&next)?;
    *lock(&state.cfg) = next;

    if mcp_changed {
        tauri::async_runtime::spawn(mcp::reload(app.clone()));
    }
    if server_changed {
        tauri::async_runtime::spawn(runtime::restart_if_running(app.clone()));
    }
    Ok(snapshot(&state).await)
}

#[tauri::command]
fn set_onboarded(onboarded: bool, state: State<AppState>) -> Result<(), String> {
    let mut c = lock(&state.cfg);
    c.onboarded = onboarded;
    config::save(&c)
}

#[tauri::command]
fn acknowledge_update(state: State<AppState>) -> Result<(), String> {
    let mut c = lock(&state.cfg);
    c.update_seen = true;
    config::save(&c)
}

#[tauri::command]
async fn set_workspace(path: String, state: State<'_, AppState>) -> Result<ConfigDto, String> {
    let trimmed = path.trim().to_string();
    if !std::path::Path::new(&trimmed).is_dir() {
        return Err(format!("That folder doesn't exist: {trimmed}"));
    }
    {
        let mut c = lock(&state.cfg);
        c.workspace_dir = trimmed;
        config::save(&c)?;
    }
    Ok(snapshot(&state).await)
}

fn main() {
    let cfg = config::load();
    if !config::path().exists() {
        let _ = config::save(&cfg);
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState {
            cfg: Mutex::new(cfg),
            convs: Mutex::new(HashMap::new()),
            running: Mutex::new(HashMap::new()),
            approval: Mutex::new(None),
            engine: Mutex::new(runtime::EngineStatus::default()),
            server: tokio::sync::Mutex::new(None),
            runtime_install: tokio::sync::Mutex::new(()),
            model_download: Mutex::new(None),
            mcp: tokio::sync::Mutex::new(McpManager::default()),
        })
        .setup(|app| {
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(mcp::reload(handle.clone()));
            // Prepare the local engine immediately on a new install. This runs
            // in the background and does not force the user to pick a model.
            tauri::async_runtime::spawn(async move {
                let _ = runtime::ensure_runtime(handle.clone()).await;
                runtime::autostart(handle).await;
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_config,
            save_config,
            set_onboarded,
            acknowledge_update,
            set_workspace,
            agent::list_sessions,
            agent::open_session,
            agent::delete_session,
            agent::send_message,
            agent::stop_message,
            agent::approve_tool,
            mcp::list_mcp_tools,
            hardware::detect_hardware,
            hardware::assess_model,
            runtime::install_runtime,
            runtime::download_model,
            runtime::cancel_model_download,
            runtime::delete_model,
            runtime::list_models,
            runtime::select_model,
            runtime::unload_model,
            runtime::retry_engine,
            runtime::engine_status,
            runtime::context_info,
        ])
        .build(tauri::generate_context!())
        .expect("error while building Parallax App")
        .run(|app, event| {
            // Stop the child as soon as the app is asked to exit (not merely
            // after the event loop has gone away). `stop_server_inner` is
            // idempotent, so the final Exit event is safe as well.
            if matches!(event, tauri::RunEvent::Exit | tauri::RunEvent::ExitRequested { .. }) {
                let state = app.state::<AppState>();
                tauri::async_runtime::block_on(runtime::stop_server_inner(&state));
            }
        });
}

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod agent;
mod app;
mod catalog;
mod db;
mod extensions;
mod hardware;
mod inference;
mod memory;
mod models;
mod runtime;
mod search;
mod server;
mod settings;
mod transfer;

use app::{App, Snapshot};
use db::{Conversation, Installed, Thread, Turn};
use inference::ChatEvent;
use settings::{ModelPreferences, Settings};
use std::{path::PathBuf, sync::Arc};
use tauri::{ipc::Channel, Manager, State};

type AppState<'a> = State<'a, Arc<App>>;
type Api<T> = Result<T, String>;
fn api<T>(value: anyhow::Result<T>) -> Api<T> {
    value.map_err(|error| error.to_string())
}

#[tauri::command]
fn open_chat_link(app: tauri::AppHandle, url: String) -> Api<()> {
    use tauri_plugin_opener::OpenerExt;
    let parsed = reqwest::Url::parse(&url).map_err(|_| "Invalid link")?;
    if !matches!(parsed.scheme(), "http" | "https")
        || parsed.host().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
    {
        return Err("Only HTTP and HTTPS links can be opened.".into());
    }
    app.opener()
        .open_url(parsed.as_str(), None::<&str>)
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn snapshot(state: AppState<'_>) -> Api<Snapshot> {
    api(state.snapshot())
}

#[tauri::command]
fn model_catalog() -> &'static [catalog::CatalogModel] {
    catalog::catalog()
}

#[tauri::command]
fn save_settings(state: AppState<'_>, settings: Settings) -> Api<Settings> {
    let was_enabled = api(state.db.settings())?.web_search;
    let saved = api(state.save_settings(settings, None))?;
    if saved.web_search && !was_enabled {
        let app = state.inner().clone();
        tauri::async_runtime::spawn(async move {
            app.prepare_search().await;
        });
    }
    Ok(saved)
}

#[tauri::command]
fn change_storage(state: AppState<'_>, directory: PathBuf) -> Api<Settings> {
    let settings = api(state.db.settings())?;
    api(state.save_settings(settings, Some(directory)))
}

#[tauri::command]
async fn setup_runtime(state: AppState<'_>, repair: bool) -> Api<Option<String>> {
    api(state.setup_runtime(repair).await)
}

#[tauri::command]
fn start_download(state: AppState<'_>, id: String) -> Api<()> {
    api(state.inner().begin_download(&id))
}

#[tauri::command]
fn control_download(state: AppState<'_>, id: String, discard: bool) -> Api<()> {
    api(state.control_download(&id, discard))
}

#[tauri::command]
async fn import_model(state: AppState<'_>, source: PathBuf) -> Api<Installed> {
    api(state.inner().import(source).await)
}

#[tauri::command]
async fn check_model_memory(
    state: AppState<'_>,
    id: String,
    custom: Option<u32>,
) -> Api<memory::Profile> {
    api(state.check_model_memory(&id, custom).await)
}

#[tauri::command]
fn save_model_preferences(
    state: AppState<'_>,
    id: String,
    preferences: ModelPreferences,
) -> Api<()> {
    api(state.save_model_preferences(&id, &preferences))
}

#[tauri::command]
async fn load_model(state: AppState<'_>, id: String, retry_gpu: bool) -> Api<()> {
    api(state.load(&id, retry_gpu).await)
}

#[tauri::command]
async fn unload_model(state: AppState<'_>) -> Api<()> {
    api(state.unload())
}

#[tauri::command]
async fn delete_model(state: AppState<'_>, id: String) -> Api<()> {
    api(state.delete_model(&id))
}

#[tauri::command]
fn set_thinking(state: AppState<'_>, id: String, enabled: bool) -> Api<()> {
    api(state.set_thinking(&id, enabled))
}

#[tauri::command]
fn create_chat(state: AppState<'_>) -> Api<Conversation> {
    api(state.db.create_chat())
}

#[tauri::command]
fn list_chats(state: AppState<'_>, search: String) -> Api<Vec<Conversation>> {
    api(state.db.list(&search))
}

#[tauri::command]
fn get_thread(state: AppState<'_>, id: String) -> Api<Thread> {
    api(state.db.thread(&id))
}

#[tauri::command]
fn update_chat(state: AppState<'_>, id: String, title: String, system_prompt: String) -> Api<()> {
    let _guard = state
        .gate
        .try_lock()
        .map_err(|_| "Wait for the current model or reply to finish.".to_owned())?;
    api(state.db.update_chat(&id, &title, &system_prompt))
}

#[tauri::command]
fn delete_chat(state: AppState<'_>, id: Option<String>) -> Api<()> {
    let _guard = state
        .gate
        .try_lock()
        .map_err(|_| "Wait for the current model or reply to finish.".to_owned())?;
    api(state.db.delete_chat(id.as_deref()))
}

#[tauri::command]
async fn export_chats(state: AppState<'_>, path: PathBuf) -> Api<()> {
    let app = state.inner().clone();
    let result = tokio::task::spawn_blocking(move || app.export(&path))
        .await
        .map_err(|error| error.to_string())?;
    api(result)
}

#[tauri::command]
async fn chat(
    state: AppState<'_>,
    id: String,
    turn: Turn,
    project: Option<agent::Project>,
    events: Channel<ChatEvent>,
) -> Api<()> {
    api(state
        .chat_with_project(&id, turn, project, |event| {
            events.send(event).map_err(Into::into)
        })
        .await)
}

#[tauri::command]
async fn chat_context(state: AppState<'_>, id: Option<String>) -> Api<usize> {
    api(state.chat_context(id.as_deref()).await)
}

#[tauri::command]
fn agent_decide(state: AppState<'_>, id: String, decision: agent::Decision) -> Api<()> {
    api(state.agent_access.decide(&id, decision))
}

#[tauri::command]
fn clear_agent_permissions(state: AppState<'_>) {
    state.agent_access.reset();
}

#[tauri::command]
fn stop_chat(state: AppState<'_>, id: String) {
    state.stop(&id);
}

#[tauri::command]
fn diagnostic_logs(state: AppState<'_>) -> Vec<String> {
    state.logs()
}

fn main() {
    let mut args = std::env::args_os().skip(1);
    if args.next().is_some_and(|arg| arg == "--supervise-runtime") {
        std::process::exit(server::supervise(args));
    }

    let application = tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _, _| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.unminimize();
                let _ = window.set_focus();
            }
        }))
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|handle| {
            let root = handle.path().app_local_data_dir()?;
            let app = App::open(root, std::env::current_exe()?, hardware::detect())?;
            app.log("Parallax App started.");
            let preferences = app.db.settings()?;
            handle.manage(app.clone());
            if preferences.web_search {
                let search_app = app.clone();
                tauri::async_runtime::spawn(async move {
                    search_app.prepare_search().await;
                });
            }
            if preferences.autoload {
                if let Some(id) = preferences.default_model {
                    tauri::async_runtime::spawn(async move {
                        let _ = app.load(&id, false).await;
                    });
                }
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            open_chat_link,
            snapshot,
            model_catalog,
            save_settings,
            change_storage,
            setup_runtime,
            start_download,
            control_download,
            import_model,
            check_model_memory,
            save_model_preferences,
            load_model,
            unload_model,
            delete_model,
            set_thinking,
            create_chat,
            list_chats,
            get_thread,
            update_chat,
            delete_chat,
            export_chats,
            chat,
            stop_chat,
            chat_context,
            agent_decide,
            clear_agent_permissions,
            diagnostic_logs
        ])
        .build(tauri::generate_context!())
        .expect("Could not start Parallax App");

    application.run(|handle, event| {
        if matches!(
            event,
            tauri::RunEvent::Exit | tauri::RunEvent::ExitRequested { .. }
        ) {
            if let Some(app) = handle.try_state::<Arc<App>>() {
                app.shutdown();
            }
        }
    });
}

use serde::Serialize;
use serde_json::Value;
use std::{
    fs,
    path::PathBuf,
    time::{Duration, UNIX_EPOCH},
};

#[derive(Serialize)]
pub struct SessionMeta {
    pub id: String,
    pub title: String,
    pub timestamp: u64,
}

fn dir() -> PathBuf {
    let d = crate::config::data_dir().join("history");
    let _ = fs::create_dir_all(&d);
    migrate_legacy(&d);
    d
}

/// Pull old history into ~/.parallax/history without overwriting a newer file.
/// Files with colliding names stay in the legacy folder for manual recovery.
fn migrate_legacy(target: &PathBuf) {
    let legacy = dirs::home_dir().unwrap_or_default().join(".parallax_history");
    let Ok(entries) = fs::read_dir(&legacy) else { return; };
    for entry in entries.flatten() {
        let from = entry.path();
        let to = target.join(entry.file_name());
        if from.is_file() && !to.exists() {
            let _ = fs::rename(&from, &to);
        }
    }
    if fs::read_dir(&legacy).map(|mut e| e.next().is_none()).unwrap_or(false) {
        let _ = fs::remove_dir(&legacy);
    }
}

fn file(kind: &str, id: &str) -> Option<PathBuf> {
    if kind != "chat" && kind != "agent" {
        return None;
    }
    // Keep only safe characters so ids can't escape the folder.
    let clean: String = id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect();
    if clean.is_empty() {
        return None;
    }
    Some(dir().join(format!("{kind}_{clean}.json")))
}

pub fn save(kind: &str, id: &str, data: &Value) -> Result<(), String> {
    let path = file(kind, id).ok_or_else(|| "Invalid session id".to_string())?;
    let text = serde_json::to_string(data).map_err(|e| e.to_string())?;
    fs::write(path, text).map_err(|e| e.to_string())
}

pub fn load(kind: &str, id: &str) -> Option<Value> {
    let path = file(kind, id)?;
    serde_json::from_str(&fs::read_to_string(path).ok()?).ok()
}

pub fn delete(kind: &str, id: &str) -> bool {
    file(kind, id).map(|p| fs::remove_file(p).is_ok()).unwrap_or(false)
}

pub fn list(kind: &str) -> Vec<SessionMeta> {
    let prefix = format!("{kind}_");
    let mut out: Vec<SessionMeta> = fs::read_dir(dir())
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            let id = name.strip_prefix(&prefix)?.strip_suffix(".json")?.to_string();
            let timestamp = entry
                .metadata()
                .ok()?
                .modified()
                .ok()?
                .duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_secs();
            let title = fs::read_to_string(entry.path())
                .ok()
                .and_then(|t| serde_json::from_str::<Value>(&t).ok())
                .and_then(|v| v["title"].as_str().map(str::to_string))
                .unwrap_or_else(|| "Untitled".to_string());
            Some(SessionMeta { id, title, timestamp })
        })
        .collect();
    out.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    out
}

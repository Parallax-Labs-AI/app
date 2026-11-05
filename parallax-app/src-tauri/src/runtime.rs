use crate::{config, ensure_idle, hardware, lock, AppState};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    fs,
    path::{Path, PathBuf},
    process::Stdio,
    sync::{atomic::{AtomicBool, Ordering}, Arc},
    time::Duration,
};
use tauri::{AppHandle, Emitter, Manager, State};
use tokio::{io::AsyncWriteExt, process::Child};

const LLAMA_REPO: &str = "ggml-org/llama.cpp";
const LLAMA_TAG: &str = "b11871";
const USER_AGENT: &str = "parallax-app";

// ---------- Minimal GGUF header reader ----------
//
// Replaces the missing `crate::gguf` module. It reads only what the context
// recommendation needs: layer count, KV head count, and embedding width.

mod gguf {
    use std::{
        collections::HashMap,
        fs::File,
        io::{self, BufReader, Read},
        path::Path,
    };

    /// What the context estimate needs. Only attention layers hold a KV cache, so
    /// hybrid models (such as Gated DeltaNet) count just their attention layers.
    #[derive(Debug, Clone, Default)]
    pub struct Arch {
        pub attention_layers: u64,
        pub kv_heads_total: u64, // KV heads summed over the attention layers
        pub key_len: u64,
        pub value_len: u64,
    }

    impl Arch {
        /// Bytes of KV cache per token: K and V in f16 (2 bytes each) for every attention layer.
        pub fn kv_bytes_per_token(&self) -> u64 {
            self.kv_heads_total * (self.key_len + self.value_len) * 2
        }
    }

    #[derive(Clone)]
    enum Val {
        Int(u64),
        Text(String),
        List(Vec<Val>),
        Other,
    }

    impl Val {
        fn as_u64(&self) -> Option<u64> {
            match self {
                Val::Int(v) => Some(*v),
                _ => None,
            }
        }
    }

    fn invalid(msg: String) -> io::Error {
        io::Error::new(io::ErrorKind::InvalidData, msg)
    }

    fn bytes<const N: usize>(r: &mut impl Read) -> io::Result<[u8; N]> {
        let mut b = [0u8; N];
        r.read_exact(&mut b)?;
        Ok(b)
    }

    fn u32_(r: &mut impl Read) -> io::Result<u32> {
        Ok(u32::from_le_bytes(bytes::<4>(r)?))
    }

    fn u64_(r: &mut impl Read) -> io::Result<u64> {
        Ok(u64::from_le_bytes(bytes::<8>(r)?))
    }

    fn string(r: &mut impl Read) -> io::Result<String> {
        let n = u64_(r)?;
        if n > 1 << 24 {
            return Err(invalid(format!("a {n}-byte string in the header")));
        }
        let mut buf = vec![0u8; n as usize];
        r.read_exact(&mut buf)?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    /// One value of a GGUF scalar type. Each type has its own width, so each is read exactly.
    fn scalar(r: &mut impl Read, ty: u32) -> io::Result<Val> {
        Ok(match ty {
            0 | 1 => Val::Int(bytes::<1>(r)?[0] as u64),
            2 | 3 => Val::Int(u16::from_le_bytes(bytes::<2>(r)?) as u64),
            4 | 5 => Val::Int(u32_(r)? as u64),
            6 => {
                bytes::<4>(r)?;
                Val::Other
            }
            7 => {
                bytes::<1>(r)?;
                Val::Other
            }
            8 => Val::Text(string(r)?),
            10 | 11 => Val::Int(u64_(r)?),
            12 => {
                bytes::<8>(r)?;
                Val::Other
            }
            other => return Err(invalid(format!("unknown GGUF value type {other}"))),
        })
    }

    fn value(r: &mut impl Read, ty: u32) -> io::Result<Val> {
        if ty != 9 {
            return scalar(r, ty);
        }
        let elem = u32_(r)?;
        let n = u64_(r)?;
        let mut items = Vec::new();
        for _ in 0..n {
            items.push(scalar(r, elem)?);
        }
        Ok(Val::List(items))
    }

    pub fn read_arch(path: &Path) -> Result<Arch, String> {
        let file = File::open(path).map_err(|e| format!("Cannot open the model file: {e}"))?;
        let mut r = BufReader::new(file);
        let header = |what: &str, e: io::Error| format!("Could not read the model header ({what}): {e}");

        if bytes::<4>(&mut r).map_err(|e| header("magic", e))? != *b"GGUF" {
            return Err("This isn't a GGUF model file.".into());
        }
        let version = u32_(&mut r).map_err(|e| header("version", e))?;
        let kv_count = if version >= 2 {
            let _tensors = u64_(&mut r).map_err(|e| header("tensor count", e))?;
            u64_(&mut r).map_err(|e| header("metadata count", e))?
        } else {
            let _tensors = u32_(&mut r).map_err(|e| header("tensor count", e))?;
            u32_(&mut r).map_err(|e| header("metadata count", e))? as u64
        };

        let mut meta: HashMap<String, Val> = HashMap::new();
        for i in 0..kv_count {
            let key = string(&mut r).map_err(|e| header(&format!("name of entry {i}"), e))?;
            let ty = u32_(&mut r).map_err(|e| header(&format!("type of '{key}'"), e))?;
            let v = value(&mut r, ty).map_err(|e| header(&format!("value of '{key}' (type {ty})"), e))?;
            meta.insert(key, v);
        }

        let arch_name = match meta.get("general.architecture") {
            Some(Val::Text(s)) => s.clone(),
            _ => return Err("The model file doesn't name its architecture.".into()),
        };
        let get = |suffix: &str| meta.get(&format!("{arch_name}.{suffix}"));

        let n_layer = get("block_count")
            .and_then(Val::as_u64)
            .ok_or("The model file doesn't give its layer count.")?;
        let head_count = get("attention.head_count").and_then(Val::as_u64).unwrap_or(0);
        let embd = get("embedding_length").and_then(Val::as_u64).unwrap_or(0);

        // Hybrid models list KV heads per layer, with 0 for layers that have no KV cache.
        let (kv_heads_total, attention_layers) = match get("attention.head_count_kv") {
            Some(Val::List(per_layer)) => {
                let counts: Vec<u64> = per_layer.iter().map(|v| v.as_u64().unwrap_or(0)).collect();
                let total: u64 = counts.iter().sum();
                let layers = counts.iter().filter(|&&c| c > 0).count() as u64;
                (total, layers)
            }
            Some(Val::Int(n)) => (n * n_layer, n_layer),
            _ => return Err("The model file doesn't say how many KV heads it uses.".into()),
        };

        let key_len = get("attention.key_length")
            .and_then(Val::as_u64)
            .unwrap_or(if head_count > 0 { embd / head_count } else { 0 });
        if key_len == 0 {
            return Err("The model file doesn't give the attention head size.".into());
        }
        let value_len = get("attention.value_length")
            .and_then(Val::as_u64)
            .unwrap_or(key_len);

        Ok(Arch {
            attention_layers,
            kv_heads_total,
            key_len,
            value_len,
        })
    }
}

// ---------- Types ----------

/// A running llama-server and the model it was started with.
pub struct ServerProc {
    pub child: Child,
    pub model_id: String,
    pub quant: String,
    pub file: String,
    pub port: u16,
    pub pid: Option<u32>,
    pub log_path: PathBuf,
}

/// What the engine is doing right now. Sent to the UI as "engine-status".
#[derive(Clone, Serialize)]
pub struct EngineStatus {
    pub state: String, // stopped | installing | downloading | starting | ready | error
    pub message: String,
    pub percent: i32,  // download progress, or -1 when unknown
}

impl Default for EngineStatus {
    fn default() -> Self {
        EngineStatus {
            state: "stopped".into(),
            message: String::new(),
            percent: -1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub enum Backend {
    Cuda,
    Vulkan,
    Metal,
    Cpu,
}

#[derive(Serialize)]
pub struct ModelChoice {
    pub model_id: String,
    pub model_name: String,
    pub quant: String,
    pub required_gb: f64,
    pub fits: bool,
    pub downloaded: bool,
    pub selected: bool,
    pub loaded: bool,
}

#[derive(Clone, Copy)]
enum Progress<'a> {
    Install,
    Model(&'a str),
}

// ---------- Paths and file names ----------

pub fn runtime_dir() -> PathBuf {
    config::data_dir().join("runtime")
}

pub fn models_dir() -> PathBuf {
    config::data_dir().join("models")
}

pub fn runtime_installed() -> bool {
    server_exe().is_file()
}

fn server_exe() -> PathBuf {
    runtime_dir().join(if cfg!(windows) { "llama-server.exe" } else { "llama-server" })
}

fn pid_file() -> PathBuf {
    runtime_dir().join("server.pid")
}

fn model_file(model_id: &str, quant: &str) -> Result<String, String> {
    match (model_id, quant) {
        ("parallax-2", "BF16") => Ok("parallax-2-bf16.gguf".into()),
        ("parallax-2", q) => Ok(format!("parallax-2-{q}.gguf")),
        ("parallax-2-mini-preview", "Q4_K_M") => Ok("parallax-labs-parallax-2-mini-preview-Q4_K_M.gguf".into()),
        (m, q) => Err(format!("No model file for {m} {q}.")),
    }
}

fn model_repo(model_id: &str) -> Result<&'static str, String> {
    match model_id {
        "parallax-2" => Ok("parallaxlabs/parallax-2"),
        "parallax-2-mini-preview" => Ok("parallaxlabs/parallax-2-mini-preview"),
        other => Err(format!("Unknown model: {other}")),
    }
}

pub fn display_name(model_id: &str, quant: &str) -> String {
    let name = hardware::catalog()
        .iter()
        .find(|m| m.id == model_id)
        .map(|m| m.name)
        .unwrap_or(model_id);
    format!("{name} {quant}")
}

// ---------- Status ----------

fn publish(app: &AppHandle, status: EngineStatus) {
    if let Some(s) = app.try_state::<AppState>() {
        *lock(&s.engine) = status.clone();
    }
    let _ = app.emit("engine-status", status);
}

fn set_engine(app: &AppHandle, state: &str, message: impl Into<String>) {
    publish(
        app,
        EngineStatus {
            state: state.to_string(),
            message: message.into(),
            percent: -1,
        },
    );
}

fn set_download(app: &AppHandle, label: &str, percent: i64) {
    publish(
        app,
        EngineStatus {
            state: "downloading".into(),
            message: label.into(),
            percent: percent as i32,
        },
    );
}

fn fail(app: &AppHandle, msg: String) -> String {
    set_engine(app, "error", msg.clone());
    msg
}

// ---------- Process helpers ----------

fn kill_pid(pid: u32) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let p = pid.to_string();
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", p.as_str(), "/T", "/F"])
            .creation_flags(0x0800_0000)
            .status();
    }
    #[cfg(not(windows))]
    {
        let _ = std::process::Command::new("kill").arg(pid.to_string()).status();
    }
}

fn port_free(port: u16) -> bool {
    std::net::TcpListener::bind(("127.0.0.1", port)).is_ok()
}

async fn health_ok(port: u16) -> bool {
    reqwest::Client::new()
        .get(format!("http://127.0.0.1:{port}/health"))
        .timeout(Duration::from_secs(2))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

async fn clear_orphan(port: u16) {
    if port_free(port) {
        return;
    }
    if let Ok(text) = fs::read_to_string(pid_file()) {
        if let Ok(pid) = text.trim().parse::<u32>() {
            kill_pid(pid);
        }
    }
    let _ = fs::remove_file(pid_file());
    for _ in 0..20 {
        if port_free(port) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}

pub async fn stop_server_inner(state: &AppState) {
    let taken = state.server.lock().await.take();
    if let Some(mut proc) = taken {
        let _ = proc.child.kill().await;
        append_log(&proc.log_path, "stopped by Parallax");
    }
    let _ = fs::remove_file(pid_file());
}

pub async fn is_ready(app: &AppHandle) -> bool {
    let state = app.state::<AppState>();
    let port = lock(&state.cfg).server_port;
    let alive = match state.server.lock().await.as_mut() {
        Some(p) => matches!(p.child.try_wait(), Ok(None)),
        None => false,
    };
    alive && health_ok(port).await
}

// ---------- Downloads ----------

async fn download(
    app: &AppHandle,
    url: &str,
    dest: &Path,
    label: &str,
    progress: Progress<'_>,
    sha256: Option<String>,
    cancel: Option<&AtomicBool>,
) -> Result<(), String> {
    use sha2::{Digest, Sha256};

    let resp = reqwest::Client::new()
        .get(url)
        .header("User-Agent", USER_AGENT)
        .send()
        .await
        .map_err(|e| format!("Could not reach the download server: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("Download failed for {label} (HTTP {}).", resp.status()));
    }

    let total = resp.content_length().unwrap_or(0);
    let tmp = dest.with_extension("part");
    let mut file = tokio::fs::File::create(&tmp).await.map_err(|e| e.to_string())?;
    let mut hasher = Sha256::new();
    let mut done: u64 = 0;
    let mut last: i64 = -2;
    let mut stream = resp.bytes_stream();

    while let Some(chunk) = stream.next().await {
        if cancel.map(|flag| flag.load(Ordering::SeqCst)).unwrap_or(false) {
            drop(file);
            let _ = tokio::fs::remove_file(&tmp).await;
            return Err("Download cancelled.".into());
        }
        let chunk = chunk.map_err(|e| format!("Download interrupted: {e}"))?;
        hasher.update(&chunk);
        file.write_all(&chunk).await.map_err(|e| e.to_string())?;
        done += chunk.len() as u64;

        let pct = if total > 0 { (done * 100 / total) as i64 } else { -1 };
        if pct != last {
            last = pct;
            match progress {
                Progress::Install => {
                    publish(
                        app,
                        EngineStatus {
                            state: "installing".into(),
                            message: label.to_string(),
                            percent: pct as i32,
                        },
                    );
                    let _ = app.emit(
                        "install-progress",
                        json!({ "label": label, "percent": pct, "done": done, "total": total }),
                    );
                }
                Progress::Model(name) => set_download(app, name, pct),
            }
        }
    }
    if cancel.map(|flag| flag.load(Ordering::SeqCst)).unwrap_or(false) {
        drop(file);
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err("Download cancelled.".into());
    }
    file.flush().await.map_err(|e| e.to_string())?;
    drop(file);

    if let Some(expected) = sha256 {
        let actual = format!("{:x}", hasher.finalize());
        if actual != expected {
            let _ = tokio::fs::remove_file(&tmp).await;
            return Err(format!("Checksum mismatch for {label}. The download was discarded."));
        }
    }
    tokio::fs::rename(&tmp, dest).await.map_err(|e| e.to_string())
}

async fn download_model_file(
    app: &AppHandle,
    model_id: &str,
    quant: &str,
    cancel: Option<&AtomicBool>,
) -> Result<(), String> {
    let file = model_file(model_id, quant)?;
    let repo = model_repo(model_id)?;
    fs::create_dir_all(models_dir()).map_err(|e| e.to_string())?;
    let dest = models_dir().join(&file);
    if dest.exists() {
        return Ok(());
    }
    let label = display_name(model_id, quant);
    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    download(app, &url, &dest, &file, Progress::Model(&label), None, cancel).await
}

// ---------- Engine install ----------

fn nvidia_driver_major() -> Option<u32> {
    let out = hardware::run("nvidia-smi", &["--query-gpu=driver_version", "--format=csv,noheader"])?;
    out.lines().next()?.trim().split('.').next()?.parse().ok()
}

pub fn choose_backend(hw: &hardware::Hardware) -> Backend {
    if hw.unified_memory {
        return Backend::Metal;
    }
    let has_nvidia = hw.gpus.iter().any(|g| g.name.to_lowercase().contains("nvidia"));
    if cfg!(windows) && has_nvidia && nvidia_driver_major().map_or(false, |d| d >= 525) {
        return Backend::Cuda;
    }
    if !hw.gpus.is_empty() {
        return Backend::Vulkan;
    }
    Backend::Cpu
}

fn fallback_chain(backend: Backend) -> Vec<Backend> {
    match backend {
        Backend::Cuda => vec![Backend::Cuda, Backend::Vulkan, Backend::Cpu],
        Backend::Vulkan => vec![Backend::Vulkan, Backend::Cpu],
        Backend::Metal => vec![Backend::Metal, Backend::Cpu],
        Backend::Cpu => vec![Backend::Cpu],
    }
}

fn wanted_patterns(backend: Backend) -> &'static [&'static str] {
    if cfg!(windows) {
        match backend {
            Backend::Cuda => &["bin-win-cuda-12"],
            Backend::Vulkan => &["bin-win-vulkan-x64"],
            _ => &["bin-win-cpu-x64"],
        }
    } else if cfg!(target_os = "macos") {
        if cfg!(target_arch = "aarch64") {
            &["bin-macos-arm64"]
        } else {
            &["bin-macos-x64"]
        }
    } else if backend == Backend::Vulkan {
        &["bin-ubuntu-vulkan-x64"]
    } else {
        &["bin-ubuntu-x64"]
    }
}

#[derive(Deserialize)]
struct Release {
    tag_name: String,
    assets: Vec<Asset>,
}

#[derive(Deserialize, Clone)]
struct Asset {
    name: String,
    browser_download_url: String,
    digest: Option<String>,
}

fn is_archive(name: &str) -> bool {
    name.ends_with(".zip") || name.ends_with(".tar.gz")
}

fn pick_engine(assets: &[Asset], backend: Backend) -> Option<(Asset, Backend)> {
    for b in fallback_chain(backend) {
        let pats = wanted_patterns(b);
        let found = assets.iter().find(|a| {
            is_archive(&a.name) && !a.name.starts_with("cudart") && pats.iter().any(|p| a.name.contains(p))
        });
        if let Some(a) = found {
            return Some((a.clone(), b));
        }
    }
    None
}

fn pick_cudart(assets: &[Asset]) -> Option<Asset> {
    assets
        .iter()
        .find(|a| a.name.starts_with("cudart") && a.name.contains("win-cuda-12") && a.name.ends_with(".zip"))
        .cloned()
}

fn extract_flat(archive: &Path, dest: &Path) -> Result<(), String> {
    let name = archive.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    if name.ends_with(".zip") {
        extract_zip(archive, dest)
    } else {
        extract_tar_gz(archive, dest)
    }
}

fn extract_zip(archive: &Path, dest: &Path) -> Result<(), String> {
    let file = fs::File::open(archive).map_err(|e| e.to_string())?;
    let mut zip = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).map_err(|e| e.to_string())?;
        if entry.is_dir() {
            continue;
        }
        let Some(name) = Path::new(entry.name()).file_name().map(|n| n.to_owned()) else {
            continue;
        };
        let mut out = fs::File::create(dest.join(name)).map_err(|e| e.to_string())?;
        std::io::copy(&mut entry, &mut out).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn extract_tar_gz(archive: &Path, dest: &Path) -> Result<(), String> {
    let file = fs::File::open(archive).map_err(|e| e.to_string())?;
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(file));
    for entry in tar.entries().map_err(|e| e.to_string())? {
        let mut entry = entry.map_err(|e| e.to_string())?;
        let kind = entry.header().entry_type();
        if kind.is_dir() {
            continue;
        }
        let path = entry.path().map_err(|e| e.to_string())?.into_owned();
        let Some(fname) = path.file_name().map(|n| n.to_owned()) else {
            continue;
        };
        let target = dest.join(&fname);

        if kind.is_symlink() {
            if let Some(link) = entry.link_name().map_err(|e| e.to_string())? {
                if let Some(link_target) = link.file_name() {
                    make_link(Path::new(link_target), &target).map_err(|e| e.to_string())?;
                }
            }
            continue;
        }
        if kind.is_file() {
            entry.unpack(&target).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

#[cfg(unix)]
fn make_link(target_name: &Path, link: &Path) -> std::io::Result<()> {
    let _ = fs::remove_file(link);
    std::os::unix::fs::symlink(target_name, link)
}

#[cfg(not(unix))]
fn make_link(_target_name: &Path, _link: &Path) -> std::io::Result<()> {
    Ok(())
}

// ---------- Logs ----------

pub fn logs_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_default().join(".parallax").join("logs")
}

const KEEP_LOGS: usize = 30;

fn timestamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + if month <= 2 { 1 } else { 0 };
    format!(
        "{year:04}-{month:02}-{day:02}_{:02}-{:02}-{:02}",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

fn open_run_log(model_file: &str, exe: &Path, args: &[String]) -> Result<(fs::File, PathBuf), String> {
    let dir = logs_dir();
    fs::create_dir_all(&dir).map_err(|e| format!("Cannot create the logs folder: {e}"))?;
    prune_logs(&dir);

    let stem = model_file.trim_end_matches(".gguf");
    let path = dir.join(format!("llama-server_{}_{}.log", timestamp(), stem));
    let mut file = fs::File::create(&path).map_err(|e| format!("Cannot write the server log: {e}"))?;

    let header = format!(
        "[parallax] {} llama-server starting\n[parallax] model:   {}\n[parallax] command: {} {}\n\n",
        timestamp(),
        model_file,
        exe.display(),
        args.join(" ")
    );
    std::io::Write::write_all(&mut file, header.as_bytes())
        .map_err(|e| format!("Cannot write the server log: {e}"))?;
    Ok((file, path))
}

fn prune_logs(dir: &Path) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut logs: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map_or(false, |n| n.starts_with("llama-server_") && n.ends_with(".log"))
        })
        .collect();
    logs.sort();
    let excess = logs.len().saturating_sub(KEEP_LOGS);
    for p in logs.into_iter().take(excess) {
        let _ = fs::remove_file(p);
    }
}

fn append_log(path: &Path, line: &str) {
    if let Ok(mut f) = fs::OpenOptions::new().append(true).create(true).open(path) {
        let text = format!("[parallax] {} {line}\n", timestamp());
        let _ = std::io::Write::write_all(&mut f, text.as_bytes());
    }
}

/// Creates the log for one start attempt. Called before anything else can fail, so every
/// attempt leaves a record. The file is opened in append mode so Parallax's lines and the
/// engine's output never overwrite each other.
fn create_run_log(model_file: &str) -> Result<(fs::File, PathBuf), String> {
    let dir = logs_dir();
    fs::create_dir_all(&dir).map_err(|e| format!("Cannot create {}: {e}", dir.display()))?;
    prune_logs(&dir);

    let stem = model_file.trim_end_matches(".gguf");
    let path = dir.join(format!("llama-server_{}_{}.log", timestamp(), stem));
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|e| format!("Cannot create {}: {e}", path.display()))?;

    let header = format!(
        "[parallax] {} start attempt\n[parallax] model: {model_file}\n",
        timestamp()
    );
    std::io::Write::write_all(&mut file, header.as_bytes())
        .map_err(|e| format!("Cannot write {}: {e}", path.display()))?;
    Ok((file, path))
}

// ---------- Context and memory ----------

const SAFETY_MIB: f64 = 256.0;
const UNMEASURED_BUFFERS_MIB: f64 = 1024.0;
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

#[derive(Serialize)]
pub struct ContextInfo {
    pub model_name: String,
    pub quant: String,
    pub cap: u32,
    pub recommended: u32,
    pub weights_gb: f64,
    pub kv_kb_per_token: f64,
    pub memory_gb: f64,
    pub memory_source: String,
    pub buffers_gb: f64,
    pub buffers_source: String,
    pub engine_kv_mib: Option<f64>,
    pub engine_ctx: Option<u32>,
    pub message: String,
}

#[derive(Serialize, Deserialize, Clone, Copy, Default)]
struct Profile {
    recurrent_mib: f64,
    compute_mib: f64,
    kv_mib: f64,
    ctx: u32,
}

fn profile_file() -> PathBuf {
    runtime_dir().join("memory_profile.json")
}

fn read_profiles() -> std::collections::HashMap<String, Profile> {
    fs::read_to_string(profile_file())
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

fn write_profile(file: &str, profile: Profile) {
    let mut all = read_profiles();
    all.insert(file.to_string(), profile);
    if let Ok(text) = serde_json::to_string_pretty(&all) {
        let _ = fs::write(profile_file(), text);
    }
}

fn mib_after_size(line: &str) -> Option<f64> {
    let rest = line.split("size =").nth(1)?.trim_start();
    let num: String = rest.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
    let value: f64 = num.parse().ok()?;
    rest.contains("mib").then_some(value)
}

fn parse_engine_memory(log: &str) -> Profile {
    let mut p = Profile::default();
    for line in log.lines() {
        let lower = line.to_lowercase();
        let Some(mib) = mib_after_size(&lower) else {
            continue;
        };
        if lower.contains("recurrent") {
            p.recurrent_mib += mib;
        } else if lower.contains("compute buffer") {
            p.compute_mib += mib;
        } else if lower.contains("kv") {
            p.kv_mib += mib;
        }
    }
    p
}

fn gpu_free_gb() -> Option<f64> {
    let out = hardware::run("nvidia-smi", &["--query-gpu=memory.free", "--format=csv,noheader,nounits"])?;
    let mib: f64 = out.lines().next()?.trim().parse().ok()?;
    Some(mib / 1024.0)
}

fn gpu_used_by_pid_gb(pid: u32) -> f64 {
    let Some(out) = hardware::run(
        "nvidia-smi",
        &["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
    ) else {
        return 0.0;
    };
    out.lines()
        .filter_map(|line| {
            let mut parts = line.split(',');
            let p: u32 = parts.next()?.trim().parse().ok()?;
            let mib: f64 = parts.next()?.trim().parse().ok()?;
            (p == pid).then_some(mib)
        })
        .sum::<f64>()
        / 1024.0
}

async fn recommend(model_id: &str, quant: &str, server_pid: Option<u32>) -> Result<ContextInfo, String> {
    let file = model_file(model_id, quant)?;
    let path = models_dir().join(&file);
    let name = display_name(model_id, quant);
    if !path.exists() {
        return Err(format!("{name} isn't downloaded yet, so its context limit can't be worked out."));
    }

    let probe = path.clone();
    let (arch, weights_bytes, hw, free_gb, used_gb) = tokio::task::spawn_blocking(move || {
        (
            gguf::read_arch(&probe),
            fs::metadata(&probe).map(|m| m.len()).unwrap_or(0),
            hardware::cached_hardware(),
            gpu_free_gb(),
            server_pid.map_or(0.0, gpu_used_by_pid_gb),
        )
    })
    .await
    .map_err(|e| e.to_string())?;
    let arch = arch?;

    let (memory_gb, memory_source) = match free_gb {
        Some(free) if used_gb > 0.0 => (free + used_gb, "free GPU memory plus this server's usage (nvidia-smi)"),
        Some(free) => (free, "free GPU memory (nvidia-smi)"),
        None => (hw.budget_gb, "budget from the hardware scan"),
    };

    let profile = read_profiles().get(&file).copied();
    let (buffers_mib, buffers_source, engine_kv_mib, engine_ctx) = match profile {
        Some(p) if p.recurrent_mib + p.compute_mib > 0.0 => (
            p.recurrent_mib + p.compute_mib,
            "reported by the engine on its last load",
            Some(p.kv_mib),
            Some(p.ctx),
        ),
        _ => (
            UNMEASURED_BUFFERS_MIB,
            "not measured yet; the first load will measure it",
            None,
            None,
        ),
    };

    let weights_gb = weights_bytes as f64 / GIB;
    let buffers_gb = buffers_mib / 1024.0;
    let kv_per_token = arch.kv_bytes_per_token() as f64;
    let spare_gb = memory_gb - weights_gb - buffers_gb - SAFETY_MIB / 1024.0;

    let max_tokens = if kv_per_token <= 0.0 {
        config::CONTEXT_CAP as f64
    } else {
        (spare_gb * GIB / kv_per_token).floor().max(0.0)
    };
    let recommended = if spare_gb <= 0.0 {
        0
    } else {
        let rounded = (max_tokens.min(config::CONTEXT_CAP as f64) as u32) / 1024 * 1024;
        if rounded < config::CONTEXT_MIN { 0 } else { rounded }
    };

    let message = if recommended == 0 {
        format!(
            "{name} doesn't leave enough memory for a {}-token context on this computer. Try a smaller quantization.",
            config::CONTEXT_MIN
        )
    } else {
        format!("Recommended context: {recommended} tokens for {memory_gb:.1} GB of memory.")
    };

    Ok(ContextInfo {
        model_name: name.clone(),
        quant: quant.into(),
        cap: config::CONTEXT_CAP,
        recommended,
        weights_gb,
        kv_kb_per_token: kv_per_token / 1024.0,
        memory_gb,
        memory_source: memory_source.into(),
        buffers_gb,
        buffers_source: buffers_source.into(),
        engine_kv_mib,
        engine_ctx,
        message,
    })
}

#[tauri::command]
pub async fn context_info(state: State<'_, AppState>) -> Result<ContextInfo, String> {
    let (model_id, quant) = {
        let c = lock(&state.cfg);
        (c.model_id.clone(), c.quant.clone())
    };
    if model_id.is_empty() {
        return Err("Select a model to see its context limit.".into());
    }
    let file = model_file(&model_id, &quant)?;
    let pid = state
        .server
        .lock()
        .await
        .as_ref()
        .filter(|p| p.file == file)
        .and_then(|p| p.pid);
    recommend(&model_id, &quant, pid).await
}

// ---------- Starting, stopping, watching ----------

pub async fn autostart(app: AppHandle) {
    let (onboarded, auto, model_id, quant) = {
        let state = app.state::<AppState>();
        let c = lock(&state.cfg);
        (c.onboarded, c.auto_start, c.model_id.clone(), c.quant.clone())
    };
    if !onboarded || !auto || model_id.is_empty() {
        set_engine(&app, "stopped", "");
        return;
    }
    let _ = ensure_and_start(&app, &model_id, &quant).await;
}

async fn ensure_and_start(app: &AppHandle, model_id: &str, quant: &str) -> Result<(), String> {
    let file = model_file(model_id, quant).map_err(|e| fail(app, e))?;
    if !models_dir().join(&file).exists() {
        return Err(fail(app, format!(
            "{} is not downloaded. Open Settings → Models & downloads to get it.",
            display_name(model_id, quant)
        )));
    }
    start_server_with(app, model_id, quant).await
}

fn watch(app: AppHandle, pid: u32) {
    tauri::async_runtime::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(2)).await;
            let state = app.state::<AppState>();
            let exit = {
                let mut g = state.server.lock().await;
                match g.as_mut() {
                    Some(p) if p.pid == Some(pid) => p
                        .child
                        .try_wait()
                        .ok()
                        .flatten()
                        .map(|code| (code, p.log_path.clone())),
                    _ => return,
                }
            };
            if let Some((code, log_path)) = exit {
                let _gone = state.server.lock().await.take();
                let _ = fs::remove_file(pid_file());
                append_log(&log_path, &format!("process exited unexpectedly: {code}"));
                set_engine(&app, "error", format!("The model server stopped unexpectedly ({code}). Details: {}", log_path.display()));
                return;
            }
        }
    });
}

/// Picks the context size to start with. Never blocks loading: if the model can't be
/// sized, the saved value or a safe default is used and the reason goes to the run log.
async fn choose_context(model_id: &str, quant: &str, requested: u32, mode: &str) -> (u32, Vec<String>) {
    let mut notes = Vec::new();
    match recommend(model_id, quant, None).await {
        Ok(info) => {
            notes.push(format!(
                "context: recommended {} tokens (memory {:.1} GB from {})",
                info.recommended, info.memory_gb, info.memory_source
            ));
            let automatic = if mode == "balanced" {
                info.recommended.min(32_768)
            } else {
                info.recommended
            };
            let mut ctx = match requested {
                0 => automatic,
                n if info.recommended > 0 && n > info.recommended => {
                    notes.push(format!(
                        "requested {n} is above the recommended {}; using the recommended size",
                        info.recommended
                    ));
                    info.recommended
                }
                n => n,
            };
            if ctx == 0 {
                notes.push(format!(
                    "not enough memory for the minimum context; trying {} anyway",
                    config::CONTEXT_MIN
                ));
                ctx = config::CONTEXT_MIN;
            }
            (ctx, notes)
        }
        Err(e) => {
            notes.push(format!("could not size the context, so sizing was skipped: {e}"));
            (if requested == 0 {
                if mode == "max" { config::CONTEXT_CAP } else { 32_768 }
            } else {
                requested
            }, notes)
        }
    }
}

/// Starts llama-server for a model. Does nothing if that model is already running.
pub async fn start_server_with(app: &AppHandle, model_id: &str, quant: &str) -> Result<(), String> {
    let state = app.state::<AppState>();
    let label = display_name(model_id, quant);
    let file = model_file(model_id, quant).map_err(|e| fail(app, e))?;
    let (port, requested_ctx, context_mode, ngl) = {
        let c = lock(&state.cfg);
        (c.server_port, c.context_size, c.context_mode.clone(), c.gpu_layers)
    };

    let mut guard = state.server.lock().await;
    if let Some(p) = guard.as_mut() {
        if p.file == file && p.port == port && matches!(p.child.try_wait(), Ok(None)) && health_ok(port).await {
            set_engine(app, "ready", label);
            return Ok(());
        }
    }
    if let Some(mut old) = guard.take() {
        let _ = old.child.kill().await;
        append_log(&old.log_path, "stopped (replaced by another model or setting)");
    }
    let _ = fs::remove_file(pid_file());

    // From here on every attempt has a log, and every error is written to it.
    let (log, log_path) = create_run_log(&file).map_err(|e| fail(app, e))?;
    let report = |msg: String| {
        append_log(&log_path, &format!("error: {msg}"));
        fail(app, msg)
    };
    append_log(
        &log_path,
        &format!("request: {label} on port {port}, context setting {requested_ctx}, gpu layers {ngl}"),
    );

    let model_path = models_dir().join(&file);
    if !model_path.exists() {
        return Err(report(format!("The model file is missing: {file}")));
    }
    if !server_exe().exists() {
        return Err(report("The engine isn't installed. Run setup again.".into()));
    }

    set_engine(app, "starting", label.clone());
    let (ctx, notes) = choose_context(model_id, quant, requested_ctx, &context_mode).await;
    for note in &notes {
        append_log(&log_path, note);
    }
    append_log(&log_path, &format!("context: starting at {ctx} tokens"));

    clear_orphan(port).await;
    if !port_free(port) {
        return Err(report(format!("Port {port} is in use by another program. Change it in Settings.")));
    }

    let args: Vec<String> = vec![
        "-m".into(),
        model_path.display().to_string(),
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        port.to_string(),
        "-c".into(),
        ctx.to_string(),
        "-np".into(),
        "1".into(),
        "-ngl".into(),
        ngl.to_string(),
    ];
    let exe = server_exe();
    append_log(&log_path, &format!("command: {} {}", exe.display(), args.join(" ")));

    let log_err = log.try_clone().map_err(|e| report(e.to_string()))?;
    let mut cmd = tokio::process::Command::new(&exe);
    cmd.args(&args)
        .current_dir(runtime_dir())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(log_err))
        .kill_on_drop(true);
    #[cfg(windows)]
    cmd.creation_flags(0x0800_0000);

    let child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            return Err(report(format!(
                "Could not start llama-server: {e}. Details: {}",
                log_path.display()
            )));
        }
    };
    let pid = child.id();
    if let Some(p) = pid {
        let _ = fs::write(pid_file(), p.to_string());
    }
    *guard = Some(ServerProc {
        child,
        model_id: model_id.into(),
        quant: quant.into(),
        file: file.clone(),
        port,
        pid,
        log_path: log_path.clone(),
    });
    drop(guard);

    for _ in 0..180 {
        let exit = {
            let mut g = state.server.lock().await;
            match g.as_mut() {
                None => return Err(report("Loading was cancelled.".into())),
                Some(p) => p.child.try_wait().ok().flatten(),
            }
        };
        if let Some(status) = exit {
            let _gone = state.server.lock().await.take();
            let _ = fs::remove_file(pid_file());
            append_log(&log_path, &format!("exited during startup: {status}"));
            return Err(report(format!(
                "llama-server exited ({status}). Details: {}",
                log_path.display()
            )));
        }
        if health_ok(port).await {
            let mut profile = parse_engine_memory(&fs::read_to_string(&log_path).unwrap_or_default());
            profile.ctx = ctx;
            write_profile(&file, profile);
            append_log(&log_path, "ready: accepting requests");
            set_engine(app, "ready", label);
            if let Some(p) = pid {
                watch(app.clone(), p);
            }
            return Ok(());
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    append_log(&log_path, "timed out waiting for the model to load");
    Err(report(format!(
        "The model didn't finish loading. Details: {}",
        log_path.display()
    )))
}

pub async fn restart_if_running(app: AppHandle) {
    let state = app.state::<AppState>();
    let running = state.server.lock().await.is_some();
    if !running {
        return;
    }
    let (model_id, quant) = {
        let c = lock(&state.cfg);
        (c.model_id.clone(), c.quant.clone())
    };
    if model_id.is_empty() {
        return;
    }
    stop_server_inner(&state).await;
    let _ = ensure_and_start(&app, &model_id, &quant).await;
}

// ---------- Commands ----------

async fn install_runtime_inner(app: &AppHandle) -> Result<String, String> {
    let hw = hardware::detect_hardware();
    let wanted_backend = choose_backend(&hw);
    let dir = runtime_dir();
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let release: Release = reqwest::Client::new()
        .get(format!("https://api.github.com/repos/{LLAMA_REPO}/releases/tags/{LLAMA_TAG}"))
        .header("User-Agent", USER_AGENT)
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .map_err(|e| format!("Could not reach GitHub: {e}"))?
        .error_for_status()
        .map_err(|e| e.to_string())?
        .json()
        .await
        .map_err(|e| e.to_string())?;

    let (asset, backend) = match pick_engine(&release.assets, wanted_backend) {
        Some(found) => found,
        None => {
            let archives: Vec<&str> = release
                .assets
                .iter()
                .map(|a| a.name.as_str())
                .filter(|n| is_archive(n))
                .collect();
            return Err(format!(
                "No llama.cpp build in release {} matches this computer ({:?}, {}). Archives in this release: {}",
                release.tag_name,
                wanted_backend,
                std::env::consts::OS,
                if archives.is_empty() { "none".to_string() } else { archives.join(", ") }
            ));
        }
    };

    if backend != wanted_backend {
        let _ = app.emit(
            "install-progress",
            json!({
                "label": format!("No {:?} build found, using {:?} instead", wanted_backend, backend),
                "percent": -1
            }),
        );
    }

    let marker = format!("{} {:?}", release.tag_name, backend);
    let installed = fs::read_to_string(dir.join("VERSION")).ok();
    if installed.as_deref() == Some(marker.as_str()) && server_exe().exists() {
        return Ok(marker);
    }

    let mut wanted = vec![asset];
    if backend == Backend::Cuda {
        wanted.push(pick_cudart(&release.assets).ok_or_else(|| {
            format!("CUDA build found, but no cudart package in release {}.", release.tag_name)
        })?);
    }

    for asset in wanted {
        let archive_path = dir.join(&asset.name);
        let sha = asset
            .digest
            .as_deref()
            .and_then(|d| d.strip_prefix("sha256:"))
            .map(str::to_string);
        download(app, &asset.browser_download_url, &archive_path, &asset.name, Progress::Install, sha, None).await?;

        publish(
            app,
            EngineStatus {
                state: "installing".into(),
                message: format!("extracting {}", asset.name),
                percent: -1,
            },
        );
        let (ap, d) = (archive_path.clone(), dir.clone());
        tokio::task::spawn_blocking(move || extract_flat(&ap, &d))
            .await
            .map_err(|e| e.to_string())??;
        let _ = fs::remove_file(&archive_path);
    }

    fs::write(dir.join("VERSION"), &marker).map_err(|e| e.to_string())?;
    Ok(marker)
}

/// Installs the local llama.cpp runtime only when it is missing. This is safe
/// to call at launch; it never replaces a working runtime behind the user's back.
pub async fn ensure_runtime(app: AppHandle) -> Result<String, String> {
    if runtime_installed() {
        return Ok("already installed".into());
    }
    let state = app.state::<AppState>();
    let _guard = state.runtime_install.lock().await;
    if runtime_installed() {
        return Ok("already installed".into());
    }
    set_engine(&app, "installing", "Preparing local AI engine");
    let result = install_runtime_inner(&app).await;
    match &result {
        Ok(_) => set_engine(&app, "stopped", ""),
        Err(error) => set_engine(&app, "error", error.clone()),
    }
    result
}

#[tauri::command]
pub async fn install_runtime(app: AppHandle) -> Result<String, String> {
    if runtime_installed() {
        return Ok("already installed".into());
    }
    let state = app.state::<AppState>();
    let _guard = state.runtime_install.lock().await;
    if runtime_installed() {
        return Ok("already installed".into());
    }
    install_runtime_inner(&app).await
}

#[tauri::command]
pub async fn list_models(state: State<'_, AppState>) -> Result<Vec<ModelChoice>, String> {
    let (sel_model, sel_quant) = {
        let c = lock(&state.cfg);
        (c.model_id.clone(), c.quant.clone())
    };
    let running: Option<String> = state.server.lock().await.as_ref().map(|p| p.file.clone());
    let ready = lock(&state.engine).state == "ready";

    let hw = hardware::cached_hardware();

    let mut out = Vec::new();
    for def in hardware::catalog() {
        let a = hardware::assess_model(def.id.to_string(), hw.budget_gb)?;
        for q in a.quants {
            let file = model_file(def.id, &q.name).ok();
            out.push(ModelChoice {
                model_id: def.id.to_string(),
                model_name: a.model_name.clone(),
                quant: q.name.clone(),
                required_gb: q.required_gb,
                fits: q.fits,
                downloaded: file.as_ref().map_or(false, |f| models_dir().join(f).exists()),
                selected: def.id == sel_model && q.name == sel_quant,
                loaded: ready && file.is_some() && running == file,
            });
        }
    }
    Ok(out)
}

/// Downloads a model without changing the currently loaded model. Keeping this
/// separate from `select_model` makes model management predictable in Settings.
#[tauri::command]
pub async fn download_model(
    app: AppHandle,
    state: State<'_, AppState>,
    model_id: String,
    quant: String,
) -> Result<(), String> {
    let (model_id, quant) = hardware::validate_model(&model_id, &quant)?;
    let file = model_file(&model_id, &quant)?;
    if models_dir().join(&file).exists() {
        return Ok(());
    }
    let cancel = {
        let mut active = lock(&state.model_download);
        if active.is_some() {
            return Err("A model download is already in progress.".into());
        }
        let flag = Arc::new(AtomicBool::new(false));
        *active = Some(flag.clone());
        flag
    };
    let label = display_name(&model_id, &quant);
    set_download(&app, &label, -1);
    let result = download_model_file(&app, &model_id, &quant, Some(&cancel)).await;
    {
        let mut active = lock(&state.model_download);
        if active.as_ref().is_some_and(|flag| Arc::ptr_eq(flag, &cancel)) {
            *active = None;
        }
    }
    match result {
        Ok(()) => {
            set_engine(&app, "stopped", "");
            Ok(())
        }
        Err(_) if cancel.load(Ordering::SeqCst) => {
            set_engine(&app, "stopped", "Download cancelled");
            Err("Download cancelled.".into())
        }
        Err(error) => Err(fail(&app, error)),
    }
}

#[tauri::command]
pub fn cancel_model_download(state: State<'_, AppState>) -> bool {
    if let Some(flag) = lock(&state.model_download).as_ref() {
        flag.store(true, Ordering::SeqCst);
        true
    } else {
        false
    }
}

#[tauri::command]
pub async fn delete_model(
    state: State<'_, AppState>,
    model_id: String,
    quant: String,
) -> Result<(), String> {
    let (model_id, quant) = hardware::validate_model(&model_id, &quant)?;
    if lock(&state.model_download).is_some() {
        return Err("Cancel the active download before removing a model.".into());
    }
    let file = model_file(&model_id, &quant)?;
    if state.server.lock().await.as_ref().is_some_and(|p| p.file == file) {
        return Err("Unload this model before removing it.".into());
    }
    let path = models_dir().join(file);
    if path.exists() {
        fs::remove_file(path).map_err(|e| format!("Could not remove the model: {e}"))?;
    }
    Ok(())
}

#[tauri::command]
pub async fn select_model(
    app: AppHandle,
    state: State<'_, AppState>,
    model_id: String,
    quant: String,
) -> Result<(), String> {
    let (model_id, quant) = hardware::validate_model(&model_id, &quant)?;
    ensure_idle(&state)?;
    stop_server_inner(&state).await;
    {
        let mut c = lock(&state.cfg);
        c.model_id = model_id.clone();
        c.quant = quant.clone();
        c.context_size = 0;
        config::save(&c)?;
    }
    ensure_and_start(&app, &model_id, &quant).await
}

#[tauri::command]
pub fn engine_status(state: State<'_, AppState>) -> EngineStatus {
    lock(&state.engine).clone()
}

/// Stops the model server and clears the engine status to "stopped".
#[tauri::command]
pub async fn unload_model(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    ensure_idle(&state)?;
    stop_server_inner(&state).await;
    set_engine(&app, "stopped", "Model unloaded");
    Ok(())
}

/// Retries starting the currently selected model after an error.
#[tauri::command]
pub async fn retry_engine(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    let (model_id, quant) = {
        let c = lock(&state.cfg);
        (c.model_id.clone(), c.quant.clone())
    };
    if model_id.is_empty() {
        return Err("No model is selected.".into());
    }
    ensure_idle(&state)?;
    stop_server_inner(&state).await;
    ensure_and_start(&app, &model_id, &quant).await
}

//! Reads the header of a GGUF model file: metadata and tensor names only.
//! The weights themselves are never read.

use std::{
    collections::{BTreeSet, HashMap},
    fs::File,
    io::{self, BufReader, Read},
    path::Path,
};

/// Architecture facts needed to size the KV cache exactly.
pub struct Arch {
    pub attention_layers: usize,
    pub kv_heads_total: u64, // KV heads summed over every full-attention layer
    pub key_len: u64,
    pub value_len: u64,
}

impl Arch {
    /// Bytes per token of context, with the engine's default f16 cache (2 bytes per value).
    pub fn kv_bytes_per_token(&self) -> u64 {
        self.kv_heads_total * (self.key_len + self.value_len) * 2
    }
}

#[derive(Clone)]
enum Value {
    Int(u64),
    Float(f64),
    Bool(bool),
    Text(String),
    List(Vec<Value>),
}

impl Value {
    fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Int(v) => Some(*v),
            Value::Bool(b) => Some(*b as u64),
            _ => None,
        }
    }

    fn as_text(&self) -> Option<&str> {
        match self {
            Value::Text(s) => Some(s),
            _ => None,
        }
    }
}

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
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

fn text(r: &mut impl Read) -> io::Result<String> {
    let n = u64_(r)?;
    if n > 1 << 24 {
        return Err(invalid("a string in the model header is too long"));
    }
    let mut b = vec![0u8; n as usize];
    r.read_exact(&mut b)?;
    Ok(String::from_utf8_lossy(&b).into_owned())
}

fn value(r: &mut impl Read, ty: u32) -> io::Result<Value> {
    Ok(match ty {
        0 | 1 => Value::Int(bytes::<1>(r)?[0] as u64),
        2 | 3 => Value::Int(u16::from_le_bytes(bytes::<2>(r)?) as u64),
        4 | 5 => Value::Int(u32_(r)? as u64),
        6 => Value::Float(f32::from_le_bytes(bytes::<4>(r)?) as f64),
        7 => Value::Bool(bytes::<1>(r)?[0] != 0),
        8 => Value::Text(text(r)?),
        9 => {
            let elem = u32_(r)?;
            let n = u64_(r)?;
            let mut items = Vec::new();
            for _ in 0..n {
                items.push(value(r, elem)?);
            }
            Value::List(items)
        }
        10 | 11 => Value::Int(u64_(r)?),
        12 => Value::Float(f64::from_le_bytes(bytes::<8>(r)?)),
        other => return Err(invalid(&format!("unknown metadata type {other}"))),
    })
}

/// "blk.12.attn_k.weight" -> Some(12). Only full-attention layers have this tensor.
fn attention_layer(name: &str) -> Option<u64> {
    let rest = name.strip_prefix("blk.")?;
    let (num, tail) = rest.split_once('.')?;
    if tail == "attn_k.weight" {
        num.parse().ok()
    } else {
        None
    }
}

/// Picks the context size to start with. This never blocks loading. If the model
/// can't be sized, the saved value or a safe default is used, and the reason is
/// written to the run log.
async fn choose_context(model_id: &str, quant: &str, requested: u32) -> (u32, Vec<String>) {
    let mut notes = Vec::new();
    match recommend(model_id, quant, None).await {
        Ok(info) => {
            notes.push(format!(
                "context: recommended {} tokens ({})",
                info.recommended, info.memory_source
            ));
            let ctx = match requested {
                0 => info.recommended,
                n if info.recommended > 0 && n > info.recommended => {
                    notes.push(format!(
                        "requested {n} is above the recommended {}; starting at the recommended size",
                        info.recommended
                    ));
                    info.recommended
                }
                n => n,
            };
            let ctx = if ctx == 0 { config::CONTEXT_MIN } else { ctx };
            (ctx, notes)
        }
        Err(e) => {
            notes.push(format!("could not size the context, so sizing was skipped: {e}"));
            let ctx = if requested == 0 { 8192 } else { requested };
            (ctx, notes)
        }
    }
}

/// Starts llama-server for a model. Does nothing if that model is already running.
pub async fn start_server_with(app: &AppHandle, model_id: &str, quant: &str) -> Result<(), String> {
    let file = model_file(model_id, quant).map_err(|e| fail(app, e))?;
    let label = display_name(model_id, quant);
    let state = app.state::<AppState>();
    let (port, requested_ctx, ngl) = {
        let c = lock(&state.cfg);
        (c.server_port, c.context_size, c.gpu_layers)
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

    let model_path = models_dir().join(&file);
    if !model_path.exists() {
        return Err(fail(app, format!("The model file is missing: {file}")));
    }
    if !server_exe().exists() {
        return Err(fail(app, "The engine isn't installed. Run setup again.".into()));
    }

    set_engine(app, "starting", label.clone());
    let (ctx, notes) = choose_context(model_id, quant, requested_ctx).await;

    clear_orphan(port).await;
    if !port_free(port) {
        return Err(fail(app, format!("Port {port} is in use by another program. Change it in Settings.")));
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
    let (log, log_path) = open_run_log(&file, &exe, &args).map_err(|e| fail(app, e))?;
    for note in &notes {
        append_log(&log_path, note);
    }
    append_log(&log_path, &format!("context: starting at {ctx} tokens"));
    let log_err = log.try_clone().map_err(|e| fail(app, e.to_string()))?;

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
            append_log(&log_path, &format!("failed to start: {e}"));
            return Err(fail(
                app,
                format!("Could not start llama-server: {e}. Details: {}", log_path.display()),
            ));
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
                None => return Err(fail(app, "Loading was cancelled.".into())),
                Some(p) => p.child.try_wait().ok().flatten(),
            }
        };
        if let Some(status) = exit {
            let _gone = state.server.lock().await.take();
            let _ = fs::remove_file(pid_file());
            append_log(&log_path, &format!("exited during startup: {status}"));
            return Err(fail(
                app,
                format!("llama-server exited ({status}). Details: {}", log_path.display()),
            ));
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
    Err(fail(
        app,
        format!("The model didn't finish loading. Details: {}", log_path.display()),
    ))
}
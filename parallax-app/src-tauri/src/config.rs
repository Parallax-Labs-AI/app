use crate::hardware;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::PathBuf};

pub const DEFAULT_MCP: &str = r#"{
  "mcpServers": {
    "web-search": {
      "command": "uvx",
      "args": [
        "free-search-mcp"
      ]
    }
  }
}"#;

pub const DEFAULT_ACCENT: &str = "#ffffff";
/// Context limits. 0 means "auto": the largest context the model's memory allows.
pub const CONTEXT_MIN: u32 = 2_048;
pub const CONTEXT_CAP: u32 = 262_144;

pub fn valid_context(n: u32) -> bool {
    n == 0 || ((CONTEXT_MIN..=CONTEXT_CAP).contains(&n) && n % 1024 == 0)
}

#[derive(Clone, Serialize)]
pub struct Config {
    pub model_id: String, // empty means nothing is selected
    pub quant: String,
    pub onboarded: bool,
    pub update_seen: bool,
    pub auto_start: bool,
    pub server_port: u16,
    pub context_size: u32,
    pub context_mode: String,
    pub gpu_layers: u32,
    pub temperature: f32,
    pub theme: String,
    pub accent: String,
    pub font_size: u32,
    pub show_reasoning: bool,
    pub workspace_dir: String,
    pub mcp_json: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            model_id: String::new(),
            quant: String::new(),
            onboarded: false,
            update_seen: false,
            auto_start: true,
            server_port: 8080,
            context_size: 0,
            context_mode: "balanced".into(),
            gpu_layers: 999,
            temperature: 0.3,
            theme: "dark".into(),
            accent: DEFAULT_ACCENT.into(),
            font_size: 14,
            show_reasoning: true,
            workspace_dir: dirs::home_dir().unwrap_or_default().display().to_string(),
            mcp_json: DEFAULT_MCP.into(),
        }
    }
}

/// Partial update sent from the Settings panel.
#[derive(Deserialize, Default)]
pub struct SettingsPatch {
    pub theme: Option<String>,
    pub accent: Option<String>,
    pub font_size: Option<u32>,
    pub show_reasoning: Option<bool>,
    pub auto_start: Option<bool>,
    pub server_port: Option<u16>,
    pub context_size: Option<u32>,
    pub context_mode: Option<String>,
    pub gpu_layers: Option<u32>,
    pub temperature: Option<f32>,
    pub workspace_dir: Option<String>,
    pub mcp_json: Option<String>,
}

impl SettingsPatch {
    /// True if this change needs the model server restarted.
    pub fn affects_server(&self, c: &Config) -> bool {
        self.server_port.map_or(false, |v| v != c.server_port)
            || self.context_size.map_or(false, |v| v != c.context_size)
            || self.context_mode.as_ref().map_or(false, |v| v != &c.context_mode)
            || self.gpu_layers.map_or(false, |v| v != c.gpu_layers)
    }

    /// Validates and applies the patch. Call this on a copy, so nothing changes if it fails.
    pub fn apply(self, c: &mut Config) -> Result<(), String> {
        if let Some(t) = self.theme {
            if t != "dark" && t != "light" {
                return Err("Theme must be dark or light.".into());
            }
            c.theme = t;
        }
        if let Some(a) = self.accent {
            c.accent = normalize_hex(&a).ok_or_else(|| "Accent colour must look like #3b82f6.".to_string())?;
        }
        if let Some(f) = self.font_size {
            if !(12..=24).contains(&f) {
                return Err("Font size must be between 12 and 24.".into());
            }
            c.font_size = f;
        }
        if let Some(p) = self.server_port {
            if p < 1024 {
                return Err("Port must be 1024 or higher.".into());
            }
            c.server_port = p;
        }
        if let Some(ctx) = self.context_size {
            if !valid_context(ctx) {
                return Err("Context must be 0 (auto) or a multiple of 1024 between 2048 and 262144.".into());
            }
            c.context_size = ctx;
        }
        if let Some(mode) = self.context_mode {
            if mode != "balanced" && mode != "max" {
                return Err("Context preference must be Balanced or Max.".into());
            }
            c.context_mode = mode;
        }
        if let Some(n) = self.gpu_layers {
            if n > 999 {
                return Err("GPU layers must be between 0 and 999.".into());
            }
            c.gpu_layers = n;
        }
        if let Some(t) = self.temperature {
            if !t.is_finite() || !(0.0..=2.0).contains(&t) {
                return Err("Temperature must be between 0 and 2.".into());
            }
            c.temperature = t;
        }
        if let Some(v) = self.show_reasoning {
            c.show_reasoning = v;
        }
        if let Some(v) = self.auto_start {
            c.auto_start = v;
        }
        if let Some(d) = self.workspace_dir {
            let d = d.trim().to_string();
            if !d.is_empty() {
                c.workspace_dir = d;
            }
        }
        if let Some(m) = self.mcp_json {
            c.mcp_json = m;
        }
        Ok(())
    }
}

pub fn normalize_hex(s: &str) -> Option<String> {
    let s = s.trim().trim_start_matches('#');
    if s.len() == 6 && s.chars().all(|c| c.is_ascii_hexdigit()) {
        Some(format!("#{}", s.to_ascii_lowercase()))
    } else {
        None
    }
}

/// `PARALLAX_CFG` overrides the location; everything else lives in ~/.parallax.
pub fn data_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_default().join(".parallax")
}

fn legacy_path() -> PathBuf {
    dirs::home_dir().unwrap_or_default().join("parallax.cfg")
}

pub fn path() -> PathBuf {
    std::env::var_os("PARALLAX_CFG")
        .map(PathBuf::from)
        .unwrap_or_else(|| data_dir().join("config.ini"))
}

/// The workspace folder if it exists, otherwise the home folder.
pub fn workspace_path(c: &Config) -> PathBuf {
    let trimmed = c.workspace_dir.trim();
    let p = PathBuf::from(trimmed);
    if !trimmed.is_empty() && p.is_dir() {
        p
    } else {
        dirs::home_dir().unwrap_or_else(|| PathBuf::from("."))
    }
}

pub fn load() -> Config {
    let c = Config::default();
    // Move the pre-0.3 configuration into the application folder once. Keeping
    // the fallback makes the migration safe even if another program has the
    // old file open on Windows.
    let target = path();
    if std::env::var_os("PARALLAX_CFG").is_none() && !target.exists() && legacy_path().exists() {
        if fs::create_dir_all(data_dir()).is_ok() {
            let _ = fs::rename(legacy_path(), &target);
        }
    }
    let Ok(text) = fs::read_to_string(path()) else {
        if let Ok(text) = fs::read_to_string(legacy_path()) {
            return parse_config(&text, c);
        }
        return c;
    };
    parse_config(&text, c)
}

fn parse_config(text: &str, mut c: Config) -> Config {
    let ini = parse_ini(&text);

    if let Some(g) = ini.get("general") {
        let get = |k: &str| g.get(k).map(|v| v.trim().to_string());
        let num = |k: &str| get(k).and_then(|v| v.parse::<u64>().ok());

        if let Some(v) = get("model_id") { c.model_id = v; }
        if let Some(v) = get("quant") { c.quant = v; }
        if let Some(v) = get("onboarded") { c.onboarded = v == "true"; }
        if let Some(v) = get("update_seen") { c.update_seen = v == "true"; }
        if let Some(v) = get("auto_start") { c.auto_start = v == "true"; }
        if let Some(v) = num("server_port") { c.server_port = u16::try_from(v).unwrap_or(0); }
        if let Some(v) = num("context_size") { c.context_size = u32::try_from(v).unwrap_or(0); }
        if let Some(v) = get("context_mode") { c.context_mode = v; }
        if let Some(v) = num("gpu_layers") { c.gpu_layers = u32::try_from(v).unwrap_or(0); }
        if let Some(v) = get("temperature").and_then(|v| v.parse::<f32>().ok()) { c.temperature = v; }
        if let Some(v) = get("theme") { c.theme = v; }
        if let Some(v) = get("accent").and_then(|v| normalize_hex(&v)) { c.accent = v; }
        if let Some(v) = num("font_size") { c.font_size = u32::try_from(v).unwrap_or(0); }
        if let Some(v) = get("show_reasoning") { c.show_reasoning = v == "true"; }
        if let Some(v) = g.get("workspace_dir") {
            if !v.trim().is_empty() {
                c.workspace_dir = v.trim().to_string();
            }
        }
    }

    // Older configs used the short id for the mini model.
    if c.model_id == "parallax-2-mini" {
        c.model_id = "parallax-2-mini-preview".into();
    }
    if !c.model_id.is_empty() {
        match hardware::validate_model(&c.model_id, &c.quant) {
            Ok((m, q)) => {
                c.model_id = m;
                c.quant = q;
            }
            Err(_) => {
                c.model_id.clear();
                c.quant.clear();
            }
        }
    }

    if let Some(v) = ini.get("mcp").and_then(|m| m.get("config")) {
        if !v.trim().is_empty() {
            c.mcp_json = v.clone();
        }
    }

    sanitize(&mut c);
    c
}

/// Puts out-of-range values from a hand-edited file back to safe defaults.
fn sanitize(c: &mut Config) {
    if c.server_port < 1024 {
        c.server_port = 8080;
    }
    if !valid_context(c.context_size) {
        c.context_size = 0;
    }
    if c.context_mode != "balanced" && c.context_mode != "max" {
        c.context_mode = "balanced".into();
    }
    c.gpu_layers = c.gpu_layers.min(999);
    c.font_size = c.font_size.clamp(12, 24);
    if !(0.0..=2.0).contains(&c.temperature) {
        c.temperature = 0.3;
    }
    if c.theme != "dark" && c.theme != "light" {
        c.theme = "dark".into();
    }
}

pub fn save(c: &Config) -> Result<(), String> {
    let pairs: [(&str, String); 15] = [
        ("model_id", c.model_id.clone()),
        ("quant", c.quant.clone()),
        ("onboarded", c.onboarded.to_string()),
        ("update_seen", c.update_seen.to_string()),
        ("auto_start", c.auto_start.to_string()),
        ("server_port", c.server_port.to_string()),
        ("context_size", c.context_size.to_string()),
        ("context_mode", c.context_mode.clone()),
        ("gpu_layers", c.gpu_layers.to_string()),
        ("temperature", c.temperature.to_string()),
        ("theme", c.theme.clone()),
        ("accent", c.accent.clone()),
        ("font_size", c.font_size.to_string()),
        ("show_reasoning", c.show_reasoning.to_string()),
        ("workspace_dir", c.workspace_dir.clone()),
    ];
    let mut out = String::from("# Parallax configuration. Changes made in the app are saved here.\n\n[general]\n");
    for (k, v) in pairs.iter() {
        out.push_str(&format!("{k} = {v}\n"));
    }
    // Multi-line JSON: each line indented by 4 spaces under the key.
    out.push_str("\n[mcp]\nconfig =\n");
    for line in c.mcp_json.lines() {
        out.push_str(&format!("    {line}\n"));
    }

    // Write to a temp file first, so a crash can't leave a half-written config.
    let target = path();
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Could not create the Parallax folder: {e}"))?;
    }
    let tmp = target.with_extension("cfg.tmp");
    fs::write(&tmp, out).map_err(|e| format!("Could not write config.ini: {e}"))?;
    fs::rename(&tmp, &target).map_err(|e| format!("Could not save config.ini: {e}"))
}

type Ini = BTreeMap<String, BTreeMap<String, String>>;

fn flush(pending: &mut Option<(String, Vec<String>)>, section: &str, out: &mut Ini) {
    if let Some((k, lines)) = pending.take() {
        out.entry(section.to_string()).or_default().insert(k, lines.join("\n"));
    }
}

/// Minimal INI reader that supports indented continuation lines.
fn parse_ini(text: &str) -> Ini {
    let mut out: Ini = BTreeMap::new();
    let mut section = String::new();
    let mut pending: Option<(String, Vec<String>)> = None;

    for raw in text.lines() {
        if raw.starts_with(char::is_whitespace) {
            if let Some((_, lines)) = pending.as_mut() {
                lines.push(raw.strip_prefix("    ").unwrap_or(raw.trim_start()).to_string());
            }
            continue;
        }
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            flush(&mut pending, &section, &mut out);
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            flush(&mut pending, &section, &mut out);
            section = line[1..line.len() - 1].trim().to_string();
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            flush(&mut pending, &section, &mut out);
            let v = v.trim();
            let lines = if v.is_empty() { vec![] } else { vec![v.to_string()] };
            pending = Some((k.trim().to_string(), lines));
        }
    }
    flush(&mut pending, &section, &mut out);
    out
}

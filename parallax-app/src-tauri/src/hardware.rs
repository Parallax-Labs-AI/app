use serde::Serialize;
use std::{process::Command, sync::OnceLock};

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

pub struct ModelDef {
    pub id: &'static str,
    pub name: &'static str,
    pub params_b: f64,
    pub quants: &'static [&'static str],
}

const MODELS: &[ModelDef] = &[
    ModelDef {
        id: "parallax-2",
        name: "Parallax 2",
        params_b: 9.0,
        quants: &["BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q2_K"],
    },
    ModelDef {
        id: "parallax-2-mini-preview",
        name: "Parallax 2 Mini Preview",
        params_b: 0.8,
        quants: &["Q4_K_M"],
    },
];

/// Approximate bits per weight for each GGUF quant, best quality first.
const QUANT_BITS: &[(&str, f64)] = &[
    ("BF16", 16.0),
    ("Q8_0", 8.5),
    ("Q6_K", 6.56),
    ("Q5_K_M", 5.69),
    ("Q4_K_M", 4.85),
    ("Q2_K", 3.35),
];

/// All models the app knows about.
pub fn catalog() -> &'static [ModelDef] {
    MODELS
}

fn model_def(id: &str) -> Option<&'static ModelDef> {
    MODELS.iter().find(|m| m.id == id)
}

/// Checks that the model exists and the quant is offered for it.
pub fn validate_model(model_id: &str, quant: &str) -> Result<(String, String), String> {
    let m = model_def(model_id).ok_or_else(|| format!("Unknown model: {model_id}"))?;
    if !m.quants.contains(&quant) {
        return Err(format!(
            "{} is not available in {quant}. Available: {}",
            m.name,
            m.quants.join(", ")
        ));
    }
    Ok((m.id.to_string(), quant.to_string()))
}

// ---------- Hardware detection ----------

#[derive(Serialize, Clone)]
pub struct Gpu {
    pub name: String,
    pub vram_gb: f64,
}

#[derive(Serialize, Clone)]
pub struct Hardware {
    pub os: String,
    pub cpu: String,
    pub ram_gb: f64,
    pub gpus: Vec<Gpu>,
    pub unified_memory: bool,
    pub budget_gb: f64, // memory planned for the model
    pub budget_source: String,
}

/// Runs a command without flashing a console window on Windows.
pub fn run(program: &str, args: &[&str]) -> Option<String> {
    let mut cmd = Command::new(program);
    cmd.args(args);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let out = cmd.output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).to_string())
}

fn nvidia_gpus() -> Vec<Gpu> {
    run(
        "nvidia-smi",
        &["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
    )
    .map(|out| {
        out.lines()
            .filter_map(|line| {
                let mut parts = line.splitn(2, ',');
                let name = parts.next()?.trim().to_string();
                let mib: f64 = parts.next()?.trim().parse().ok()?;
                Some(Gpu { name, vram_gb: mib / 1024.0 })
            })
            .collect()
    })
    .unwrap_or_default()
}

/// Windows: reads VRAM from the display-adapter class registry key.
/// This works for AMD and Intel and avoids the 4 GB limit of WMI's AdapterRAM.
#[cfg(windows)]
fn windows_gpus() -> Vec<Gpu> {
    let Some(out) = run(
        "reg",
        &[
            "query",
            r"HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}",
            "/s",
        ],
    ) else {
        return vec![];
    };

    type Entry = (Option<String>, Option<u64>); // (name, bytes)
    fn push(cur: &mut Option<Entry>, gpus: &mut Vec<Gpu>) {
        if let Some((Some(name), Some(bytes))) = cur.take() {
            if bytes >= 512 * 1024 * 1024 {
                gpus.push(Gpu { name, vram_gb: bytes as f64 / GIB });
            }
        }
    }

    let mut gpus = vec![];
    let mut current: Option<Entry> = None;
    for line in out.lines() {
        let t = line.trim();
        if t.starts_with("HKEY") {
            push(&mut current, &mut gpus);
            current = Some((None, None));
            continue;
        }
        // reg output columns are separated by 4 spaces: name, type, data
        let cols: Vec<&str> = t.split("    ").map(str::trim).filter(|s| !s.is_empty()).collect();
        if cols.len() < 3 {
            continue;
        }
        let Some(entry) = current.as_mut() else {
            continue;
        };
        match cols[0] {
            "DriverDesc" => entry.0 = Some(cols[2].to_string()),
            "HardwareInformation.qwMemorySize" => {
                entry.1 = u64::from_str_radix(cols[2].trim_start_matches("0x"), 16).ok();
            }
            _ => {}
        }
    }
    push(&mut current, &mut gpus);
    gpus
}

/// Linux: AMD and Intel expose VRAM size under /sys/class/drm.
#[cfg(target_os = "linux")]
fn linux_drm_gpus() -> Vec<Gpu> {
    let Ok(entries) = std::fs::read_dir("/sys/class/drm") else {
        return vec![];
    };
    entries
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with("card"))
        .filter_map(|e| {
            let base = e.path().join("device");
            let bytes: u64 = std::fs::read_to_string(base.join("mem_info_vram_total"))
                .ok()?
                .trim()
                .parse()
                .ok()?;
            Some(Gpu { name: "AMD/Intel GPU".into(), vram_gb: bytes as f64 / GIB })
        })
        .collect()
}

#[cfg(windows)]
fn platform_gpus() -> Vec<Gpu> {
    windows_gpus()
}

#[cfg(target_os = "linux")]
fn platform_gpus() -> Vec<Gpu> {
    linux_drm_gpus()
}

#[cfg(not(any(windows, target_os = "linux")))]
fn platform_gpus() -> Vec<Gpu> {
    vec![]
}

static HARDWARE_CACHE: OnceLock<Hardware> = OnceLock::new();

fn scan_hardware() -> Hardware {
    // Refresh only the CPU and memory data we use. `new_all()` also scans every
    // process and is needlessly expensive when Settings refreshes the model list.
    let sys = sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::new()
            .with_cpu(sysinfo::CpuRefreshKind::new())
            .with_memory(sysinfo::MemoryRefreshKind::new().with_ram()),
    );
    let ram_gb = sys.total_memory() as f64 / GIB;
    let cpu = sys
        .cpus()
        .first()
        .map(|c| c.brand().trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "Unknown CPU".into());

    let mut gpus = nvidia_gpus();
    let has_nvidia = !gpus.is_empty();
    for g in platform_gpus() {
        // nvidia-smi already reported NVIDIA cards with accurate numbers.
        if !(has_nvidia && g.name.to_lowercase().contains("nvidia")) {
            gpus.push(g);
        }
    }

    let unified = cfg!(target_os = "macos") && cpu.to_lowercase().contains("apple");
    let best_vram = gpus.iter().map(|g| g.vram_gb).fold(0.0_f64, f64::max);

    let (budget_gb, budget_source) = if unified {
        let b = ram_gb * 0.70;
        (
            b,
            format!(
                "Apple silicon shares {ram_gb:.0} GB between CPU and GPU. About 70% ({b:.1} GB) is planned for the model."
            ),
        )
    } else if best_vram > 0.0 {
        let b = (best_vram - 0.5).max(0.0);
        (
            b,
            format!(
                "Dedicated GPU memory: {best_vram:.1} GB. Planning with {b:.1} GB to leave headroom for your desktop."
            ),
        )
    } else {
        let b = ram_gb * 0.5;
        (
            b,
            format!(
                "No dedicated GPU found. The model would run on the CPU using about half your RAM ({b:.1} GB). Responses will be slow."
            ),
        )
    };

    Hardware {
        os: std::env::consts::OS.into(),
        cpu,
        ram_gb,
        gpus,
        unified_memory: unified,
        budget_gb,
        budget_source,
    }
}

pub fn cached_hardware() -> Hardware {
    HARDWARE_CACHE.get_or_init(scan_hardware).clone()
}

#[tauri::command]
pub fn detect_hardware() -> Hardware {
    cached_hardware()
}

// ---------- Quantization recommendation ----------

#[derive(Serialize)]
pub struct QuantOption {
    pub name: String,
    pub size_gb: f64,
    pub required_gb: f64,
    pub fits: bool,
    pub recommended: bool,
}

#[derive(Serialize)]
pub struct Assessment {
    pub model_id: String,
    pub model_name: String,
    pub memory_gb: f64,
    pub quants: Vec<QuantOption>,
    pub can_run: bool,
    pub recommended: Option<String>,
    pub message: String,
}

/// Lists the quants for a model, marks which fit in `memory_gb`, and picks
/// the best-quality one that fits. If none fit, `can_run` is false.
#[tauri::command]
pub fn assess_model(model_id: String, memory_gb: f64) -> Result<Assessment, String> {
    let m = model_def(&model_id).ok_or_else(|| format!("Unknown model: {model_id}"))?;

    let mut quants: Vec<QuantOption> = QUANT_BITS
        .iter()
        .filter(|(q, _)| m.quants.contains(q))
        .map(|(q, bits)| {
            let size_gb = m.params_b * bits / 8.0;
            // Weights + ~10% runtime overhead + ~1 GB for context (KV cache).
            let required_gb = size_gb * 1.1 + 1.0;
            QuantOption {
                name: q.to_string(),
                size_gb,
                required_gb,
                fits: required_gb <= memory_gb,
                recommended: false,
            }
        })
        .collect();

    let best = quants.iter().position(|q| q.fits);
    if let Some(i) = best {
        quants[i].recommended = true;
    }

    let message = match best {
        Some(i) => format!(
            "Recommended: {} at {}. It needs about {:.1} GB and you have {:.1} GB.",
            m.name, quants[i].name, quants[i].required_gb, memory_gb
        ),
        None => {
            let smallest = quants.iter().map(|q| q.required_gb).fold(f64::MAX, f64::min);
            format!(
                "{} can't run on this hardware. Its smallest quant needs about {:.1} GB, but you have {:.1} GB.",
                m.name, smallest, memory_gb
            )
        }
    };

    let recommended = best.map(|i| quants[i].name.clone());
    Ok(Assessment {
        model_id: m.id.into(),
        model_name: m.name.into(),
        memory_gb,
        can_run: best.is_some(),
        recommended,
        quants,
        message,
    })
}

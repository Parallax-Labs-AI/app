use serde::Serialize;
use sysinfo::System;

#[derive(Serialize)]
pub struct SystemInfo {
    os: String,
    arch: String,
    cpu_cores: usize,
    total_ram_gb: u64,
    gpu_model: String,
    total_vram_gb: u64,
}

#[tauri::command]
fn get_system_info() -> SystemInfo {
    let mut sys = System::new_all();
    sys.refresh_all();

    // FIX: sysinfo returns "Darwin" for macOS. We want it to say "macOS".
    let raw_os = System::name().unwrap_or_else(|| "Unknown".to_string());
    let os = if raw_os == "Darwin" { "macOS".to_string() } else { raw_os };

    let arch = System::cpu_arch().unwrap_or_else(|| "Unknown".to_string());
    let cpu_cores = sys.cpus().len();
    let total_ram_gb = sys.total_memory() / (1024 * 1024 * 1024);

    let cpu_brand = sys.cpus().first().map(|c| c.brand().to_string()).unwrap_or_else(|| "Unknown CPU".to_string());
    
    let (gpu_model, total_vram_gb) = if cpu_brand.contains("Apple") {
        (format!("{} (Unified Memory)", cpu_brand), total_ram_gb)
    } else {
        (format!("{} (Discrete GPU)", cpu_brand), 0)
    };

    SystemInfo {
        os,
        arch,
        cpu_cores,
        total_ram_gb,
        gpu_model,
        total_vram_gb,
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![get_system_info])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
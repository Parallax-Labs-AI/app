use serde::Serialize;
use std::{env, path::Path, process::Command};
use sysinfo::System;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Hardware {
    pub os: String,
    pub arch: String,
    pub cpu: String,
    pub memory: u64,
    pub available_memory: u64,
    pub gpu: Option<String>,
    pub vram: Option<u64>,
    pub metal: bool,
    pub cuda: bool,
    pub vulkan: bool,
}

pub fn detect() -> Hardware {
    let mut system = System::new_all();
    system.refresh_memory();
    let cpu = system
        .cpus()
        .first()
        .map(|cpu| cpu.brand().trim())
        .filter(|s| !s.is_empty())
        .unwrap_or(env::consts::ARCH)
        .to_owned();
    let mut profile = Hardware {
        os: env::consts::OS.into(),
        arch: env::consts::ARCH.into(),
        cpu,
        memory: system.total_memory(),
        available_memory: system.available_memory(),
        gpu: None,
        vram: None,
        metal: false,
        cuda: false,
        vulkan: false,
    };
    if profile.os == "macos" {
        detect_metal(&mut profile);
    } else {
        if let Some((gpu, memory, driver)) = nvidia() {
            profile.gpu = Some(gpu);
            profile.vram = Some(memory);
            let minimum = if profile.os == "windows" { 551 } else { 570 };
            profile.cuda = driver >= minimum;
        }
        if let Some((gpu, vram)) = vulkan() {
            profile.vulkan = true;
            if profile.gpu.is_none() {
                profile.gpu = Some(gpu);
                profile.vram = vram;
            }
        }
    }
    profile
}

fn detect_metal(profile: &mut Hardware) {
    if profile.arch == "aarch64" {
        profile.metal = true;
        profile.gpu = Some("Apple Silicon GPU · unified memory".into());
        return;
    }
    if let Ok(output) = Command::new("/usr/sbin/system_profiler")
        .args(["-json", "SPDisplaysDataType"])
        .output()
    {
        if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&output.stdout) {
            if let Some(displays) = json["SPDisplaysDataType"].as_array() {
                if let Some(gpu) = displays.iter().find(|gpu| {
                    gpu["spdisplays_metal"]
                        .as_str()
                        .is_some_and(|v| v.starts_with("spdisplays_supported"))
                }) {
                    profile.metal = true;
                    profile.gpu = gpu["sppci_model"].as_str().map(str::to_owned);
                }
            }
        }
    }
}

pub fn background(command: &mut Command) -> &mut Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    command
}

pub fn system_memory() -> (u64, u64) {
    let mut system = System::new();
    system.refresh_memory();
    let total = system.total_memory();
    let mut available = system.available_memory();
    if cfg!(target_os = "macos") {
        // macOS's pressure counter reflects reclaimable memory, including
        // file cache. A count of currently unused pages can be much lower.
        if let Some(percent) = Command::new("/usr/sbin/sysctl")
            .args(["-n", "kern.memorystatus_level"])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .and_then(|output| output.trim().parse::<u64>().ok())
            .filter(|percent| *percent <= 100)
        {
            available = total.saturating_mul(percent) / 100;
        }
    }
    (total, available)
}

pub fn runtime_command(executable: &Path) -> Command {
    let mut command = Command::new(executable);
    background(&mut command);
    for (key, _) in env::vars_os() {
        if ["LLAMA_", "GGML_", "DYLD_", "LD_PRELOAD", "LD_LIBRARY_PATH"]
            .iter()
            .any(|prefix| key.to_string_lossy().starts_with(prefix))
        {
            command.env_remove(key);
        }
    }
    command
}

fn nvidia() -> Option<(String, u64, u32)> {
    let output = background(&mut Command::new("nvidia-smi"))
        .args([
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            let mut fields = line.rsplitn(3, ',').map(str::trim);
            let driver = fields.next()?.split('.').next()?.parse().ok()?;
            let memory = fields
                .next()?
                .parse::<u64>()
                .ok()?
                .checked_mul(1024 * 1024)?;
            Some((fields.next()?.to_owned(), memory, driver))
        })
        .max_by_key(|(_, memory, _)| *memory)
}

fn vulkan() -> Option<(String, Option<u64>)> {
    use ash::vk;
    // All handles belong to this temporary instance and are released before unloading the loader.
    unsafe {
        let entry = ash::Entry::load().ok()?;
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
        let instance = entry
            .create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )
            .ok()?;
        let result = instance
            .enumerate_physical_devices()
            .ok()
            .and_then(|devices| {
                devices
                    .into_iter()
                    .filter_map(|device| {
                        let props = instance.get_physical_device_properties(device);
                        if props.api_version < vk::API_VERSION_1_2
                            || !matches!(
                                props.device_type,
                                vk::PhysicalDeviceType::DISCRETE_GPU
                                    | vk::PhysicalDeviceType::INTEGRATED_GPU
                            )
                        {
                            return None;
                        }
                        let name = props
                            .device_name_as_c_str()
                            .ok()?
                            .to_string_lossy()
                            .into_owned();
                        let memory = instance.get_physical_device_memory_properties(device);
                        let dedicated = (props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU)
                            .then(|| {
                                memory.memory_heaps[..memory.memory_heap_count as usize]
                                    .iter()
                                    .filter(|heap| {
                                        heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
                                    })
                                    .map(|heap| heap.size)
                                    .max()
                                    .unwrap_or(0)
                            })
                            .filter(|bytes| *bytes > 0);
                        Some((name, dedicated))
                    })
                    .max_by_key(|(_, dedicated)| *dedicated)
            });
        instance.destroy_instance(None);
        result
    }
}

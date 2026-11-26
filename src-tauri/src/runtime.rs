use crate::{
    catalog::sha_is_valid,
    hardware::Hardware,
    transfer::{self, Progress},
};
use anyhow::{bail, ensure, Context, Result};
use flate2::read::GzDecoder;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io::{self, Read},
    path::{Component, Path, PathBuf},
    sync::OnceLock,
};
use tokio_util::sync::CancellationToken;

#[derive(Deserialize)]
pub struct Manifest {
    pub release: String,
    pub targets: Vec<Target>,
}

#[derive(Deserialize)]
pub struct Target {
    os: String,
    arch: String,
    pub backend: String,
    assets: Vec<Asset>,
}

#[derive(Deserialize)]
struct Asset {
    name: String,
    size: u64,
    sha256: String,
}

#[derive(Serialize)]
pub struct RuntimeInfo {
    pub release: &'static str,
    pub suggested: Option<String>,
    pub installed: Vec<String>,
    pub previous: Vec<String>,
}

pub fn manifest() -> &'static Manifest {
    static MANIFEST: OnceLock<Manifest> = OnceLock::new();
    MANIFEST.get_or_init(|| {
        serde_json::from_str(include_str!("../data/runtime.json"))
            .expect("valid bundled runtime manifest")
    })
}

pub fn select(
    hardware: &Hardware,
    preference: &str,
) -> Result<Vec<(&'static Target, &'static str)>> {
    let targets = &manifest().targets;
    let find = |backend: &str| {
        targets.iter().find(|target| {
            target.os == hardware.os && target.arch == hardware.arch && target.backend == backend
        })
    };
    let fallback = find("cpu")
        .or_else(|| {
            if hardware.os == "macos" {
                find("metal")
            } else {
                None
            }
        })
        .context("This OS and architecture have no verified runtime in the current release.")?;
    let primary = match preference {
        "cpu" => None,
        "auto" if hardware.metal => find("metal").map(|t| (t, "metal")),
        "auto" if hardware.cuda => find("cuda").map(|t| (t, "cuda")),
        "auto" if hardware.vulkan => find("vulkan").map(|t| (t, "vulkan")),
        "auto" => None,
        "metal" => Some((
            find("metal").context("Metal is unavailable on this platform.")?,
            "metal",
        )),
        "cuda" => Some((
            find("cuda").context("CUDA is unavailable on this platform.")?,
            "cuda",
        )),
        "vulkan" => Some((
            find("vulkan").context("Vulkan is unavailable on this platform.")?,
            "vulkan",
        )),
        _ => bail!("Invalid acceleration mode."),
    };
    let mut choices = Vec::new();
    if let Some(primary) = primary {
        choices.push(primary);
    }
    choices.push((fallback, "cpu"));
    Ok(choices)
}

pub fn info(root: &Path, hardware: &Hardware, preference: &str) -> RuntimeInfo {
    let choices = select(hardware, preference).unwrap_or_default();
    let installed = manifest()
        .targets
        .iter()
        .filter(|target| {
            target.os == hardware.os
                && target.arch == hardware.arch
                && executable(root, target).is_some()
        })
        .map(|target| target.backend.clone())
        .collect();
    let mut previous = fs::read_dir(root)
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let release = entry.file_name().into_string().ok()?;
            if release == manifest().release || !release.starts_with('b') {
                return None;
            }
            let installed = ["metal", "cpu", "cuda", "vulkan"]
                .iter()
                .any(|backend| entry.path().join(backend).join(".verified").is_file());
            installed.then_some(release)
        })
        .collect::<Vec<_>>();
    previous.sort();
    RuntimeInfo {
        release: &manifest().release,
        suggested: choices.first().map(|(_, name)| name.to_string()),
        installed,
        previous,
    }
}

fn directory(root: &Path, target: &Target) -> PathBuf {
    root.join(&manifest().release).join(&target.backend)
}

pub fn executable(root: &Path, target: &Target) -> Option<PathBuf> {
    let dir = directory(root, target);
    let exe = dir.join(if target.os == "windows" {
        "llama-server.exe"
    } else {
        "llama-server"
    });
    if dir.join(".verified").is_file()
        && fs::symlink_metadata(&exe).is_ok_and(|m| m.is_file())
        && profiler(&exe).is_file()
    {
        Some(exe)
    } else {
        None
    }
}

pub fn profiler(server: &Path) -> PathBuf {
    server.with_file_name(if cfg!(windows) {
        "llama-fit-params.exe"
    } else {
        "llama-fit-params"
    })
}

pub async fn install(
    client: &Client,
    root: &Path,
    target: &'static Target,
    repair: bool,
    cancel: &CancellationToken,
    report: impl Fn(Progress) + Send,
) -> Result<PathBuf> {
    let destination = directory(root, target);
    let parent = destination.parent().context("Missing runtime directory")?;
    let backup = destination.with_extension("previous");
    if backup.exists() {
        if destination.exists() {
            fs::remove_dir_all(&backup)?;
        } else {
            fs::rename(&backup, &destination)?;
        }
    }
    if !repair {
        if let Some(exe) = executable(root, target) {
            return Ok(exe);
        }
    }
    let cache = destination.with_extension("downloads");
    fs::create_dir_all(&cache)?;
    let files = cache.join("files");
    if files.exists() {
        fs::remove_dir_all(&files)?;
    }
    fs::create_dir(&files)?;
    let total: u64 = target.assets.iter().map(|asset| asset.size).sum();
    transfer::available(parent, total.saturating_mul(3))?;
    let mut previous = 0;
    for asset in &target.assets {
        validate_asset(asset)?;
        let archive = cache.join(&asset.name);
        let url = format!(
            "https://github.com/ggml-org/llama.cpp/releases/download/{}/{}",
            manifest().release,
            asset.name
        );
        transfer::fetch(
            client,
            &url,
            &archive,
            asset.size,
            &asset.sha256,
            cancel,
            |mut p| {
                p.bytes += previous;
                p.total = total;
                p.eta = p
                    .speed
                    .and_then(|speed| (total - p.bytes).checked_div(speed));
                report(p);
            },
        )
        .await?;
        previous += asset.size;
        report(Progress::phase("extracting", previous, total));
        let output = files.clone();
        let token = cancel.clone();
        tokio::task::spawn_blocking(move || extract(&archive, &output, &token)).await??;
    }
    #[cfg(windows)]
    install_windows_crt(&files)?;
    let exe = files.join(if target.os == "windows" {
        "llama-server.exe"
    } else {
        "llama-server"
    });
    ensure!(
        exe.is_file() && profiler(&exe).is_file(),
        "Verified archive did not contain llama-server and its memory profiler."
    );
    ensure!(!cancel.is_cancelled(), "Runtime setup was stopped.");
    fs::write(files.join(".verified"), &manifest().release)?;
    // Keep a previous installation available if publishing its replacement fails.
    if destination.exists() {
        fs::rename(&destination, &backup)?;
    }
    if let Err(error) = fs::rename(&files, &destination) {
        if backup.exists() {
            let _ = fs::rename(&backup, &destination);
        }
        return Err(error).context("Could not publish the verified runtime.");
    }
    let _ = fs::remove_dir_all(&backup);
    let _ = fs::remove_dir_all(&cache);
    executable(root, target).context("Verified runtime could not be opened.")
}

#[cfg(windows)]
fn install_windows_crt(files: &Path) -> Result<()> {
    let executable = std::env::current_exe()?;
    let resource_dir = executable
        .parent()
        .context("Missing application directory")?
        .join("runtime-crt");
    if cfg!(debug_assertions) && !resource_dir.join("vcruntime140.dll").is_file() {
        return Ok(());
    }
    for name in ["msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"] {
        fs::copy(resource_dir.join(name), files.join(name)).context(
            "Missing packaged Visual C++ libraries. Reinstall the Parallax App Windows bundle.",
        )?;
    }
    Ok(())
}

fn validate_asset(asset: &Asset) -> Result<()> {
    ensure!(
        asset.size > 0
            && sha_is_valid(&asset.sha256)
            && asset
                .name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b".-_".contains(&b))
            && (asset.name.ends_with(".zip") || asset.name.ends_with(".tar.gz")),
        "Invalid bundled runtime asset."
    );
    Ok(())
}

fn safe_path(path: &Path) -> Result<PathBuf> {
    let mut clean = PathBuf::new();
    for part in path.components() {
        match part {
            Component::CurDir => {}
            Component::Normal(piece) => {
                let text = piece
                    .to_str()
                    .context("Runtime archive contains a non-Unicode name.")?;
                ensure!(
                    text.bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
                        && !text.ends_with('.'),
                    "Unsafe component in the runtime archive."
                );
                clean.push(piece);
            }
            _ => bail!("Unsafe path in the runtime archive."),
        }
    }
    Ok(clean)
}

fn needed(name: &str) -> bool {
    matches!(
        name,
        "llama-server" | "llama-fit-params" | "llama-fit-params.exe"
    ) || name == "llama-server.exe"
        || name.starts_with("LICENSE")
        || name.ends_with(".dll")
        || (!name.contains("-impl") && (name.ends_with(".dylib") || name.contains(".so")))
        || name.contains("server-impl")
        || name.contains("fit-params-impl")
}

fn write_entry(
    mut input: impl Read,
    path: &Path,
    size: u64,
    cancel: &CancellationToken,
) -> Result<()> {
    ensure!(size <= 2_000_000_000, "Runtime archive entry is too large.");
    let mut file = File::options().write(true).create_new(true).open(path)?;
    let mut buffer = vec![0; 1024 * 1024];
    let mut remaining = size;
    while remaining > 0 {
        ensure!(!cancel.is_cancelled(), "Runtime setup was stopped.");
        let read = remaining.min(buffer.len() as u64) as usize;
        input.read_exact(&mut buffer[..read])?;
        io::Write::write_all(&mut file, &buffer[..read])?;
        remaining -= read as u64;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(fs::Permissions::from_mode(
            if path
                .file_name()
                .is_some_and(|n| n == "llama-server" || n == "llama-fit-params")
            {
                0o755
            } else {
                0o644
            },
        ))?;
    }
    Ok(())
}

fn extract(archive: &Path, destination: &Path, cancel: &CancellationToken) -> Result<()> {
    let input = File::open(archive)?;
    if archive.extension().is_some_and(|ext| ext == "zip") {
        let mut zip = zip::ZipArchive::new(input)?;
        for i in 0..zip.len() {
            let entry = zip.by_index(i)?;
            let path = safe_path(Path::new(entry.name()))?;
            ensure!(
                entry.is_dir()
                    || entry
                        .unix_mode()
                        .is_none_or(|mode| matches!(mode & 0o170000, 0 | 0o100000)),
                "Unsupported link in runtime ZIP."
            );
            if !entry.is_dir() {
                let name = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .context("Missing runtime file name")?;
                if needed(name) && !(name.starts_with("LICENSE") && destination.join(name).exists())
                {
                    let size = entry.size();
                    write_entry(entry, &destination.join(name), size, cancel)?;
                }
            }
        }
        return Ok(());
    }
    let mut tar = tar::Archive::new(GzDecoder::new(input));
    let mut links = Vec::new();
    for entry in tar.entries()? {
        let entry = entry?;
        let path = safe_path(&entry.path()?)?;
        let kind = entry.header().entry_type();
        if kind.is_dir() {
            continue;
        }
        ensure!(
            kind.is_file() || kind.is_symlink(),
            "Unsupported entry in the runtime archive."
        );
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .context("Missing runtime file name")?;
        if kind.is_symlink() {
            let link = safe_path(&entry.link_name()?.context("Missing runtime library link")?)?;
            ensure!(
                link.components().count() == 1,
                "Unsafe runtime library link target."
            );
            if needed(name) {
                links.push((destination.join(name), link));
            }
        } else if needed(name) && !(name.starts_with("LICENSE") && destination.join(name).exists())
        {
            let size = entry.size();
            write_entry(entry, &destination.join(name), size, cancel)?;
        }
    }
    #[cfg(unix)]
    {
        // Some release archives alias optional components that we intentionally omit.
        // Resolve each retained link through the flat list before creating it.
        for (name, link) in &links {
            let mut target = destination.join(link);
            let mut depth = 0;
            while let Some((_, next)) = links.iter().find(|(alias, _)| alias == &target) {
                depth += 1;
                ensure!(
                    depth <= links.len(),
                    "Runtime archive contains a link cycle."
                );
                target = destination.join(next);
            }
            ensure!(
                target.is_file(),
                "Runtime archive is missing a linked library."
            );
            std::os::unix::fs::symlink(link, name)?;
        }
    }
    #[cfg(not(unix))]
    ensure!(
        links.is_empty(),
        "Symbolic links are unsupported for this runtime."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_releases_and_fallbacks_are_explicit() {
        for target in &manifest().targets {
            for asset in &target.assets {
                validate_asset(asset).unwrap();
            }
        }
        let mut hardware = Hardware {
            os: "linux".into(),
            arch: "x86_64".into(),
            cpu: "test".into(),
            memory: 0,
            available_memory: 0,
            gpu: None,
            vram: None,
            metal: false,
            cuda: true,
            vulkan: true,
        };
        let modes = |hardware: &Hardware, preference| {
            select(hardware, preference)
                .unwrap()
                .into_iter()
                .map(|(_, mode)| mode)
                .collect::<Vec<_>>()
        };
        assert_eq!(modes(&hardware, "auto"), ["cuda", "cpu"]);
        hardware.cuda = false;
        assert_eq!(modes(&hardware, "auto"), ["vulkan", "cpu"]);
        assert_eq!(modes(&hardware, "cpu"), ["cpu"]);
        hardware.os = "macos".into();
        hardware.arch = "aarch64".into();
        hardware.metal = true;
        assert_eq!(modes(&hardware, "auto"), ["metal", "cpu"]);
        hardware.arch = "unsupported".into();
        assert!(select(&hardware, "auto").is_err());
    }

    #[test]
    fn archive_rejects_traversal_and_escaping_links() {
        for path in ["../../escape", "/tmp/escape", "C:/escape", "folder\\escape"] {
            assert!(safe_path(Path::new(path)).is_err(), "{path}");
        }
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("unsafe.tar.gz");
        let encoder = flate2::write::GzEncoder::new(
            File::create(&file).unwrap(),
            flate2::Compression::fast(),
        );
        let mut tar = tar::Builder::new(encoder);
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(tar::EntryType::Symlink);
        header.set_size(0);
        tar.append_link(&mut header, "libfoo.dylib", "../../escape")
            .unwrap();
        tar.into_inner().unwrap().finish().unwrap();
        assert!(extract(&file, tmp.path(), &CancellationToken::new()).is_err());
    }
}

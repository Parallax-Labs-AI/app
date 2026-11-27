use crate::transfer::{self, Progress};
use anyhow::{bail, ensure, Context, Result};
use reqwest::{redirect::Policy, Client, Url};
use semver::Version;
use serde::{Deserialize, Serialize};
use std::{path::Path, time::Duration};
use tokio_util::sync::CancellationToken;

const REPOSITORY: &str = "Parallax-Labs-AI/app";
const CURRENT: &str = env!("CARGO_PKG_VERSION");

#[derive(Deserialize)]
struct Release {
    tag_name: String,
    draft: bool,
    prerelease: bool,
    assets: Vec<Asset>,
}

#[derive(Deserialize)]
struct Asset {
    name: String,
    browser_download_url: String,
    digest: Option<String>,
    size: u64,
    state: String,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum Status {
    Current,
    Available,
    NoInstaller,
}

#[derive(Serialize)]
pub struct Check {
    current: String,
    latest: String,
    status: Status,
    installer: Option<Installer>,
}

#[derive(Serialize)]
struct Installer {
    name: String,
    size: u64,
}

fn client() -> Result<Client> {
    Ok(Client::builder()
        .https_only(true)
        .connect_timeout(Duration::from_secs(12))
        .read_timeout(Duration::from_secs(30))
        .user_agent(concat!("ParallaxApp/", env!("CARGO_PKG_VERSION")))
        .redirect(Policy::custom(|attempt| {
            let url = attempt.url();
            if attempt.previous().len() < 5
                && url.scheme() == "https"
                && matches!(
                    url.host_str(),
                    Some(
                        "api.github.com"
                            | "github.com"
                            | "release-assets.githubusercontent.com"
                            | "objects.githubusercontent.com"
                    )
                )
                && url.username().is_empty()
                && url.password().is_none()
                && url.port().is_none()
            {
                attempt.follow()
            } else {
                attempt.error("Unexpected GitHub release redirect")
            }
        }))
        .build()?)
}

async fn latest(client: &Client, cancel: &CancellationToken) -> Result<Release> {
    let request = client
        .get(format!(
            "https://api.github.com/repos/{REPOSITORY}/releases/latest"
        ))
        .header("Accept", "application/vnd.github+json");
    tokio::select! {
        _ = cancel.cancelled() => bail!("Update check cancelled."),
        result = async {
            let response = request.send().await.context("Couldn't reach GitHub to check for updates.")?;
            let status = response.status();
            ensure!(status.is_success(), "GitHub returned HTTP {status} while checking for updates.");
            response.json().await.context("Couldn't read the latest GitHub release.")
        } => result,
    }
}

impl Asset {
    fn checksum(&self) -> Option<&str> {
        self.digest
            .as_deref()?
            .strip_prefix("sha256:")
            .filter(|hash| hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()))
    }

    fn valid(&self, tag: &str) -> bool {
        self.state == "uploaded"
            && self.size > 0
            && self.checksum().is_some()
            && !self.name.starts_with('.')
            && self
                .name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
            && Url::parse(&self.browser_download_url)
                .ok()
                .is_some_and(|url| {
                    url.as_str()
                        == format!(
                            "https://github.com/{REPOSITORY}/releases/download/{tag}/{}",
                            self.name
                        )
                })
    }
}

fn installer_priority(name: &str, os: &str, arch: &str) -> Option<u8> {
    let name = name.to_ascii_lowercase();
    if !["parallax.app_", "parallax-app-", "parallax_app_"]
        .iter()
        .any(|prefix| name.starts_with(prefix))
    {
        return None;
    }
    let parts = name.split(['.', '_', '-']);
    let native = match arch {
        "aarch64" => parts
            .clone()
            .any(|part| matches!(part, "aarch64" | "arm64")),
        "x86_64" => {
            parts.clone().any(|part| matches!(part, "x64" | "amd64")) || name.contains("x86_64")
        }
        _ => false,
    };
    match os {
        "macos" if name.ends_with(".dmg") && native => Some(0),
        "macos" if name.ends_with(".dmg") && parts.clone().any(|part| part == "universal") => {
            Some(1)
        }
        "windows" if native && name.ends_with("-setup.exe") => Some(0),
        "windows" if native && name.ends_with("_en-us.msi") => Some(1),
        "windows" if native && name.ends_with(".msi") => Some(2),
        "linux" if native && name.ends_with(".appimage") => Some(0),
        _ => None,
    }
}

fn architecture() -> &'static str {
    // Intel builds can be running under Rosetta on an Apple Silicon Mac.
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    if std::process::Command::new("/usr/sbin/sysctl")
        .args(["-in", "hw.optional.arm64"])
        .output()
        .is_ok_and(|output| output.status.success() && output.stdout.starts_with(b"1"))
    {
        return "aarch64";
    }
    std::env::consts::ARCH
}

impl Release {
    fn compare(&self, current: &str) -> Result<std::cmp::Ordering> {
        let tag = self
            .tag_name
            .strip_prefix('v')
            .context("The GitHub release tag must start with v.")?;
        let latest =
            Version::parse(tag).context("The GitHub release tag is not a valid version.")?;
        ensure!(
            !self.draft && !self.prerelease && latest.pre.is_empty(),
            "The GitHub release is not a stable release."
        );
        Ok(latest.cmp_precedence(&Version::parse(current)?))
    }

    fn installer(&self, os: &str, arch: &str) -> Option<&Asset> {
        self.assets
            .iter()
            .filter(|asset| asset.valid(&self.tag_name))
            .filter_map(|asset| Some((installer_priority(&asset.name, os, arch)?, asset)))
            .min_by_key(|(priority, asset)| (*priority, &asset.name))
            .map(|(_, asset)| asset)
    }

    fn check(&self, current: &str, os: &str, arch: &str) -> Result<Check> {
        use std::cmp::Ordering;
        let (status, installer) = match self.compare(current)? {
            Ordering::Less | Ordering::Equal => (Status::Current, None),
            Ordering::Greater => match self.installer(os, arch) {
                Some(asset) => (
                    Status::Available,
                    Some(Installer {
                        name: asset.name.clone(),
                        size: asset.size,
                    }),
                ),
                None => (Status::NoInstaller, None),
            },
        };
        Ok(Check {
            current: format!("v{current}"),
            latest: self.tag_name.clone(),
            status,
            installer,
        })
    }
}

pub async fn check(cancel: &CancellationToken) -> Result<Check> {
    let release = latest(&client()?, cancel).await?;
    release.check(CURRENT, std::env::consts::OS, architecture())
}

pub async fn download(
    tag: &str,
    destination: &Path,
    cancel: &CancellationToken,
    report: impl Fn(Progress),
) -> Result<()> {
    let client = client()?;
    let release = latest(&client, cancel).await?;
    ensure!(
        release.tag_name == tag && release.compare(CURRENT)?.is_gt(),
        "The release changed. Check for updates again."
    );
    let asset = release
        .installer(std::env::consts::OS, architecture())
        .context("This release has no verified installer for your device.")?;
    save_installer(&client, asset, destination, cancel, report).await
}

async fn save_installer(
    client: &Client,
    asset: &Asset,
    destination: &Path,
    cancel: &CancellationToken,
    report: impl Fn(Progress),
) -> Result<()> {
    let extension = Path::new(&asset.name).extension();
    ensure!(
        destination.is_absolute() && destination.extension() == extension,
        "Choose a filename with the installer's original extension."
    );
    ensure!(
        !destination.exists(),
        "A file with that name already exists. Choose a different name."
    );
    let folder = destination.parent().context("Choose a download folder.")?;
    let temporary = tempfile::Builder::new()
        .prefix(".parallax-update-")
        .tempfile_in(folder)?
        .into_temp_path();
    let checksum = asset
        .checksum()
        .context("The installer is missing a SHA-256 checksum.")?
        .to_ascii_lowercase();
    transfer::fetch(
        client,
        &asset.browser_download_url,
        &temporary,
        asset.size,
        &checksum,
        cancel,
        report,
    )
    .await?;
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::PermissionsExt;
        tokio::fs::set_permissions(&temporary, std::fs::Permissions::from_mode(0o755)).await?;
    }
    temporary
        .persist_noclobber(destination)
        .context("Couldn't save the verified installer. Check the destination and retry.")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(tag: &str, name: &str) -> Asset {
        Asset {
            name: name.into(),
            browser_download_url: format!(
                "https://github.com/{REPOSITORY}/releases/download/{tag}/{name}"
            ),
            digest: Some(format!("sha256:{}", "a".repeat(64))),
            size: 1024,
            state: "uploaded".into(),
        }
    }

    fn release(tag: &str, names: &[&str]) -> Release {
        Release {
            tag_name: tag.into(),
            draft: false,
            prerelease: false,
            assets: names.iter().map(|name| asset(tag, name)).collect(),
        }
    }

    #[test]
    fn compares_release_tags_as_versions_and_never_offers_downgrades() {
        let published = release("v0.10.0", &["Parallax.App_0.10.0_x64.dmg"]);
        let check = |current| published.check(current, "macos", "x86_64").unwrap().status;
        assert!(matches!(check("0.9.0"), Status::Available));
        assert!(matches!(check("0.10.0"), Status::Current));
        let newer = published.check("1.0.0", "macos", "x86_64").unwrap();
        assert!(matches!(newer.status, Status::Current));
        assert!(newer.installer.is_none());
        assert!(release("v0.10.0-beta.1", &[]).compare("0.9.0").is_err());
    }

    #[test]
    fn chooses_native_installers_and_ignores_extraction_archives() {
        let mut published = release(
            "v0.5.2",
            &[
                "Parallax.App_aarch64.app.tar.gz",
                "Parallax.App_x64.app.tar.gz",
                "Parallax.App_0.5.2_x64_en-US.msi",
                "Parallax.App_0.5.2_x64-setup.exe",
                "Parallax.App_0.5.2_aarch64.dmg",
                "Parallax.App_0.5.2_x64.dmg",
            ],
        );
        let chosen = |os, arch| published.installer(os, arch).unwrap().name.as_str();
        assert_eq!(chosen("macos", "aarch64"), "Parallax.App_0.5.2_aarch64.dmg");
        assert_eq!(chosen("macos", "x86_64"), "Parallax.App_0.5.2_x64.dmg");
        assert_eq!(
            chosen("windows", "x86_64"),
            "Parallax.App_0.5.2_x64-setup.exe"
        );

        // An external URL or absent checksum cannot be selected for download.
        published.assets[4].browser_download_url = "https://example.org/fake.dmg".into();
        published.assets[5].digest = None;
        assert!(matches!(
            published.check("0.5.1", "macos", "x86_64").unwrap().status,
            Status::NoInstaller
        ));
        assert!(published.installer("macos", "aarch64").is_none());
    }

    #[tokio::test]
    #[ignore = "Downloads the latest public Parallax DMG into a temporary folder"]
    async fn downloads_and_verifies_a_published_github_installer() {
        let client = client().unwrap();
        let cancel = CancellationToken::new();
        let release = latest(&client, &cancel).await.unwrap();
        let result = release.check(CURRENT, "macos", "aarch64").unwrap();
        eprintln!(
            "Installed {}. Latest GitHub release {}.",
            result.current, result.latest
        );
        let asset = release.installer("macos", "aarch64").unwrap();
        assert!(asset.name.ends_with(".dmg"));
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join(&asset.name);
        save_installer(&client, asset, &path, &cancel, |_| {})
            .await
            .unwrap();
        assert_eq!(std::fs::metadata(path).unwrap().len(), asset.size);
    }
}

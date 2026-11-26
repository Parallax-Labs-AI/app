use crate::{
    catalog,
    db::{Database, Installed},
    transfer::{self, Progress},
};
use anyhow::{bail, ensure, Context, Result};
use reqwest::Client;
use std::{
    fs::{self, File},
    io::{BufReader, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[derive(Debug)]
pub struct Gguf {
    pub architecture: String,
    pub context: Option<u32>,
}

struct Header {
    reader: BufReader<File>,
    limit: u64,
}

impl Header {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut bytes = [0; N];
        self.reader.read_exact(&mut bytes)?;
        ensure!(
            self.reader.stream_position()? <= self.limit,
            "GGUF metadata exceeds the supported size."
        );
        Ok(bytes)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn skip(&mut self, length: u64) -> Result<()> {
        let position = self
            .reader
            .stream_position()?
            .checked_add(length)
            .context("GGUF metadata overflow")?;
        ensure!(
            position <= self.limit,
            "Incomplete or oversized GGUF metadata."
        );
        self.reader.seek(SeekFrom::Start(position))?;
        Ok(())
    }
    fn text(&mut self, maximum: u64) -> Result<String> {
        let length = self.u64()?;
        ensure!(length <= maximum, "Unexpectedly long GGUF metadata value.");
        let mut bytes = vec![0; length as usize];
        self.reader.read_exact(&mut bytes)?;
        Ok(String::from_utf8(bytes)?)
    }
    fn skip_value(&mut self, kind: u32) -> Result<()> {
        match kind {
            0 | 1 | 7 => self.skip(1),
            2 | 3 => self.skip(2),
            4..=6 => self.skip(4),
            10..=12 => self.skip(8),
            8 => {
                let length = self.u64()?;
                self.skip(length)
            }
            9 => {
                let element = self.u32()?;
                let count = self.u64()?;
                let width = match element {
                    0 | 1 | 7 => 1,
                    2 | 3 => 2,
                    4..=6 => 4,
                    10..=12 => 8,
                    8 => {
                        ensure!(count <= 2_000_000, "GGUF string array is too large.");
                        for _ in 0..count {
                            self.skip_value(8)?;
                        }
                        return Ok(());
                    }
                    _ => bail!("Unsupported GGUF array type."),
                };
                self.skip(count.checked_mul(width).context("GGUF array overflow")?)
            }
            _ => bail!("Unsupported GGUF metadata type."),
        }
    }
}

pub fn inspect(path: &Path) -> Result<Gguf> {
    let file = File::open(path).context("Could not open the GGUF file.")?;
    let size = file.metadata()?.len();
    let mut header = Header {
        reader: BufReader::new(file),
        limit: size.min(128 * 1024 * 1024),
    };
    ensure!(
        &header.bytes::<4>()? == b"GGUF" && matches!(header.u32()?, 2 | 3),
        "Select a little-endian GGUF v2 or v3 file."
    );
    let tensors = header.u64()?;
    let entries = header.u64()?;
    ensure!(
        (1..=100_000).contains(&tensors) && entries <= 65_536,
        "This file is not a complete chat model."
    );
    let mut architecture = None;
    let mut contexts = Vec::new();
    let mut chat_template = false;
    let mut split = false;
    for _ in 0..entries {
        let key = header.text(1024)?;
        let kind = header.u32()?;
        match key.as_str() {
            "general.architecture" if kind == 8 => architecture = Some(header.text(128)?),
            "tokenizer.chat_template" if kind == 8 => {
                chat_template = !header.text(256 * 1024)?.trim().is_empty()
            }
            "split.count" if kind == 2 => split = u16::from_le_bytes(header.bytes()?) > 1,
            _ if key.ends_with(".context_length") && kind == 4 => {
                contexts.push((key, header.u32()?))
            }
            _ => header.skip_value(kind)?,
        }
    }
    let architecture = architecture.context("This GGUF has no model architecture.")?;
    ensure!(
        supports_architecture(&architecture),
        "This GGUF architecture is outside the app's supported import list."
    );
    ensure!(
        !split,
        "Import a single-file GGUF. Split models require all shards."
    );
    ensure!(
        chat_template,
        "Use an instruction or chat GGUF with an embedded chat template."
    );
    // Read the complete tensor directory and check each data offset belongs to the file.
    // llama.cpp validates quantization types and complete tensor sizes on load.
    let mut last_offset = 0;
    for _ in 0..tensors {
        header.text(1024)?;
        let dims = header.u32()?;
        ensure!((1..=4).contains(&dims), "Invalid GGUF tensor dimensions.");
        header.skip(u64::from(dims) * 8 + 4)?;
        last_offset = last_offset.max(header.u64()?);
    }
    ensure!(
        header.reader.stream_position()?.saturating_add(last_offset) < size,
        "GGUF tensor data is incomplete."
    );
    let context = contexts
        .into_iter()
        .find(|(key, _)| *key == format!("{architecture}.context_length"))
        .map(|(_, n)| n)
        .filter(|n| *n > 0);
    Ok(Gguf {
        architecture,
        context,
    })
}

pub fn supports_architecture(architecture: &str) -> bool {
    matches!(
        architecture,
        "llama"
            | "mistral"
            | "qwen2"
            | "qwen2moe"
            | "qwen3"
            | "qwen3moe"
            | "qwen35"
            | "qwen35moe"
            | "gemma"
            | "gemma2"
            | "gemma3"
            | "phi3"
            | "phi4"
            | "gptoss"
            | "deepseek2"
            | "smollm3"
    )
}

pub fn part_path(directory: &Path, id: &str) -> PathBuf {
    directory.join(format!("{id}.gguf.part"))
}

pub async fn download(
    db: &Database,
    client: &Client,
    directory: &Path,
    id: &str,
    cancel: &CancellationToken,
    report: impl Fn(Progress),
) -> Result<Installed> {
    let (model, variant) = catalog::resolve(id)?;
    fs::create_dir_all(directory)?;
    let part = part_path(directory, id);
    let destination = directory.join(format!("{id}.gguf"));
    let url = format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        model.repo,
        model.revision.as_ref().context("Missing model revision")?,
        variant.file
    );
    let verifying = if destination.exists() {
        // Recover a publish that finished immediately before the app stopped.
        let path = destination.clone();
        let token = cancel.clone();
        report(Progress::phase("verifying", variant.bytes, variant.bytes));
        tokio::task::spawn_blocking(move || -> Result<bool> {
            Ok(fs::symlink_metadata(&path)?.is_file()
                && fs::metadata(&path)?.len() == variant.bytes
                && transfer::digest(&path, &token)? == variant.sha256)
        })
        .await??
    } else {
        false
    };
    if !verifying {
        transfer::fetch(
            client,
            &url,
            &part,
            variant.bytes,
            &variant.sha256,
            cancel,
            &report,
        )
        .await?;
    }
    let candidate = if verifying {
        destination.clone()
    } else {
        part.clone()
    };
    let metadata = tokio::task::spawn_blocking(move || inspect(&candidate)).await??;
    ensure!(
        Some(metadata.architecture.as_str()) == model.architecture.as_deref(),
        "The GGUF architecture differs from the official catalog."
    );
    ensure!(!cancel.is_cancelled(), "Transfer stopped.");
    if !verifying {
        if destination.exists() {
            fs::remove_file(&destination)?;
        }
        fs::rename(part, &destination)?;
    }
    let installed = Installed {
        id: id.into(),
        name: model.name.clone(),
        quantization: Some(variant.quantization.clone()),
        bytes: variant.bytes,
        path: destination,
        architecture: metadata.architecture,
        context: metadata.context.or(model.context),
        imported: false,
        available_on_disk: true,
    };
    db.register(&installed)?;
    Ok(installed)
}

pub fn import(
    db: &Database,
    source: &Path,
    directory: &Path,
    report: impl Fn(Progress),
) -> Result<Installed> {
    ensure!(
        source.is_absolute()
            && source.is_file()
            && source
                .extension()
                .is_some_and(|s| s.eq_ignore_ascii_case("gguf")),
        "Choose a local GGUF file."
    );
    let id = Uuid::new_v4().to_string();
    let bytes = fs::metadata(source)?.len();
    report(Progress::phase("verifying", 0, bytes));
    inspect(source)?;
    fs::create_dir_all(directory)?;
    transfer::available(directory, bytes)?;
    let name = source
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Imported model")
        .chars()
        .take(120)
        .collect();
    let mut stage = tempfile::NamedTempFile::new_in(directory)?;
    report(Progress::phase("importing", 0, bytes));
    let copied = std::io::copy(&mut File::open(source)?, stage.as_file_mut())?;
    ensure!(copied == bytes, "The source file changed during import.");
    stage.as_file().sync_all()?;
    let meta = inspect(stage.path())?;
    let model = Installed {
        id: id.clone(),
        name,
        quantization: None,
        bytes,
        path: directory.join(format!("{id}.gguf")),
        architecture: meta.architecture,
        context: meta.context,
        imported: true,
        available_on_disk: true,
    };
    stage.persist_noclobber(&model.path)?;
    if let Err(error) = db.register(&model) {
        let _ = fs::remove_file(&model.path);
        return Err(error);
    }
    Ok(model)
}

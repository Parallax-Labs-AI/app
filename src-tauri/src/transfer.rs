use anyhow::{bail, ensure, Context, Result};
use futures_util::StreamExt;
use reqwest::{header, Client, StatusCode};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
    time::{Duration, Instant},
};
use tokio::{fs, io::AsyncWriteExt};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Progress {
    pub phase: &'static str,
    pub bytes: u64,
    pub total: u64,
    pub speed: Option<u64>,
    pub eta: Option<u64>,
}

impl Progress {
    pub fn phase(phase: &'static str, bytes: u64, total: u64) -> Self {
        Self {
            phase,
            bytes,
            total,
            speed: None,
            eta: None,
        }
    }
}

pub fn available(path: &Path, needed: u64) -> Result<()> {
    if let Ok(free) = fs2::available_space(path) {
        ensure!(free >= needed.saturating_add(64 * 1024 * 1024),
            "There isn't enough free space in the selected folder. Free up space or choose another model folder.");
    }
    Ok(())
}

pub fn digest(path: &Path, cancel: &CancellationToken) -> Result<String> {
    let mut file = BufReader::with_capacity(1024 * 1024, File::open(path)?);
    let mut buffer = vec![0; 1024 * 1024];
    let mut hash = Sha256::new();
    loop {
        ensure!(!cancel.is_cancelled(), "Transfer stopped.");
        let len = file.read(&mut buffer)?;
        if len == 0 {
            return Ok(format!("{:x}", hash.finalize()));
        }
        hash.update(&buffer[..len]);
    }
}

pub async fn fetch(
    client: &Client,
    url: &str,
    part: &Path,
    total: u64,
    sha256: &str,
    cancel: &CancellationToken,
    report: impl Fn(Progress),
) -> Result<()> {
    ensure!(!cancel.is_cancelled(), "Transfer stopped.");
    let mut have = match fs::symlink_metadata(part).await {
        Ok(info) if info.is_file() => info.len(),
        Ok(_) => bail!("The temporary download path is not a regular file."),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
        Err(error) => return Err(error.into()),
    };
    if have > total {
        fs::remove_file(part).await?;
        have = 0;
    }
    if have < total {
        available(
            part.parent().context("Missing download folder")?,
            total - have,
        )?;
        let mut request = client.get(url).header(header::ACCEPT_ENCODING, "identity");
        if have > 0 {
            request = request.header(header::RANGE, format!("bytes={have}-"));
        }
        let response = tokio::select! {
            _ = cancel.cancelled() => bail!("Transfer stopped."),
            response = request.send() => response.map_err(|_| anyhow::anyhow!("Couldn't reach the download server. Check your connection and retry."))?,
        };
        match response.status() {
            StatusCode::PARTIAL_CONTENT => {
                let expected = format!("bytes {have}-{}/{total}", total - 1);
                ensure!(response.headers().get(header::CONTENT_RANGE).and_then(|value| value.to_str().ok()) == Some(expected.as_str()),
                    "The download server returned an unexpected byte range. Cancel the partial file and retry.");
            }
            StatusCode::OK => {
                if have > 0 { available(part.parent().unwrap(), total.saturating_sub(have))?; }
                have = 0; // Range ignored; replace the partial file instead of appending.
            }
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => bail!("The repository requires access or no longer permits this download. Check the official model page."),
            status => bail!("The download server returned HTTP {status}. Check the source and retry."),
        }
        if let Some(length) = response.content_length() {
            ensure!(
                length == total - have,
                "The download size differs from the pinned metadata."
            );
        }
        let mut file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(have > 0)
            .truncate(have == 0)
            .open(part)
            .await?;
        let initial = have;
        let started = Instant::now();
        let mut last_update = started;
        report(Progress::phase("downloading", have, total));
        let mut stream = response.bytes_stream();
        loop {
            let item = tokio::select! {
                _ = cancel.cancelled() => {
                    file.sync_all().await?;
                    bail!("Transfer stopped.");
                }
                item = stream.next() => item,
            };
            let Some(chunk) = item else {
                break;
            };
            let chunk = chunk.map_err(|_| {
                anyhow::anyhow!("Download interrupted. Resume to keep downloading.")
            })?;
            ensure!(
                have.saturating_add(chunk.len() as u64) <= total,
                "The server sent more data than expected."
            );
            file.write_all(&chunk).await?;
            have += chunk.len() as u64;
            if last_update.elapsed() >= Duration::from_millis(250) {
                let speed = ((have - initial) as f64 / started.elapsed().as_secs_f64()) as u64;
                report(Progress {
                    phase: "downloading",
                    bytes: have,
                    total,
                    speed: Some(speed),
                    eta: (total - have).checked_div(speed),
                });
                last_update = Instant::now();
            }
        }
        file.sync_all().await?;
    }
    ensure!(
        have == total,
        "The download is incomplete. Resume to verify the full file."
    );
    report(Progress::phase("verifying", total, total));
    let path = part.to_owned();
    let token = cancel.clone();
    let actual = tokio::task::spawn_blocking(move || digest(&path, &token)).await??;
    if actual != sha256 {
        fs::remove_file(part).await?;
        bail!("The SHA-256 checksum didn't match. The file was discarded. Retry the download.");
    }
    ensure!(!cancel.is_cancelled(), "Transfer stopped.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{BufRead, Write},
        net::TcpListener,
    };
    fn serve(response: &'static str, expect_range: bool) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(&socket);
            let mut request = String::new();
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" || line.is_empty() {
                    break;
                }
                request.push_str(&line);
            }
            assert_eq!(
                request.to_ascii_lowercase().contains("range: bytes=3-"),
                expect_range
            );
            socket.write_all(response.as_bytes()).unwrap();
        });
        url
    }

    #[tokio::test]
    async fn resumes_and_verifies_before_publication() {
        let tmp = tempfile::tempdir().unwrap();
        let part = tmp.path().join("model.part");
        fs::write(&part, b"abc").await.unwrap();
        let url = serve("HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 3-5/6\r\nContent-Length: 3\r\n\r\ndef", true);
        let sha = format!("{:x}", Sha256::digest(b"abcdef"));
        fetch(
            &Client::new(),
            &url,
            &part,
            6,
            &sha,
            &CancellationToken::new(),
            |_| {},
        )
        .await
        .unwrap();
        assert_eq!(fs::read(&part).await.unwrap(), b"abcdef");
        assert!(fetch(
            &Client::new(),
            &url,
            &part,
            6,
            "wrong",
            &CancellationToken::new(),
            |_| {}
        )
        .await
        .is_err());
        assert!(!part.exists());
    }

    #[tokio::test]
    async fn cancel_stops_an_active_transfer_and_keeps_written_bytes() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            let mut request = BufReader::new(&socket);
            loop {
                let mut line = String::new();
                if request.read_line(&mut line).unwrap() == 0 || line == "\r\n" {
                    break;
                }
            }
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 6\r\n\r\nab")
                .unwrap();
            std::thread::sleep(Duration::from_millis(350));
            socket.write_all(b"c").unwrap();
            std::thread::sleep(Duration::from_millis(800));
        });
        let tmp = tempfile::tempdir().unwrap();
        let part = tmp.path().join("model.part");
        let cancel = CancellationToken::new();
        let result = fetch(&Client::new(), &url, &part, 6, "", &cancel, |progress| {
            if progress.phase == "downloading" && progress.bytes > 0 {
                cancel.cancel();
            }
        })
        .await;
        assert!(result.unwrap_err().to_string().contains("stopped"));
        assert_eq!(fs::read(&part).await.unwrap(), b"abc");
    }

    #[tokio::test]
    async fn cancel_leaves_partial_and_short_response_fails() {
        let tmp = tempfile::tempdir().unwrap();
        let part = tmp.path().join("model.part");
        fs::write(&part, b"abc").await.unwrap();
        let cancel = CancellationToken::new();
        cancel.cancel();
        assert!(
            fetch(&Client::new(), "unused", &part, 6, "", &cancel, |_| {})
                .await
                .is_err()
        );
        assert_eq!(fs::read(&part).await.unwrap(), b"abc");
        let url = serve("HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nxy", true);
        assert!(fetch(
            &Client::new(),
            &url,
            &part,
            6,
            "",
            &CancellationToken::new(),
            |_| {}
        )
        .await
        .is_err());
        assert_eq!(fs::read(&part).await.unwrap(), b"xy");
    }
}

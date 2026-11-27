use crate::{
    db::Installed,
    hardware::{background, runtime_command},
    memory::placement,
};
use anyhow::{bail, ensure, Context, Result};
use reqwest::{redirect::Policy, Client};
use serde::Serialize;
use std::{
    ffi::OsString,
    io::Read,
    net::TcpListener,
    path::Path,
    process::{Child, Command, Stdio},
    sync::mpsc,
    time::{Duration, Instant},
};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerStatus {
    pub phase: &'static str,
    pub model_id: Option<String>,
    pub backend: Option<String>,
    pub detail: Option<String>,
}

impl Default for ServerStatus {
    fn default() -> Self {
        Self {
            phase: "idle",
            model_id: None,
            backend: None,
            detail: None,
        }
    }
}

#[derive(Clone)]
pub struct Endpoint {
    pub base: String,
    pub secret: String,
    pub context: u32,
    pub parallel: u32,
    pub model_id: String,
}

pub struct Server {
    supervisor: Child,
    pub endpoint: Endpoint,
}

impl Drop for Server {
    fn drop(&mut self) {
        drop(self.supervisor.stdin.take());
        let _ = self.supervisor.wait();
    }
}

impl Server {
    pub fn exited(&mut self) -> bool {
        !matches!(self.supervisor.try_wait(), Ok(None))
    }
}

pub fn local_client() -> Result<Client> {
    Ok(Client::builder()
        .no_proxy()
        .redirect(Policy::none())
        .connect_timeout(Duration::from_secs(3))
        .build()?)
}

pub async fn start(
    launcher: &Path,
    executable: &Path,
    model: &Installed,
    backend: &str,
    context: u32,
    parallel: u32,
    gpu_layers: Option<u32>,
    client: &Client,
    closing: &CancellationToken,
) -> Result<Server> {
    let info = std::fs::symlink_metadata(&model.path)
        .context("Model file is missing. Download or import it again.")?;
    ensure!(
        info.is_file() && info.len() == model.bytes,
        "The model file is incomplete. Download or import it again."
    );
    for _ in 0..3 {
        ensure!(!closing.is_cancelled(), "Parallax App is closing.");
        let socket =
            TcpListener::bind(("127.0.0.1", 0)).context("No loopback port could be reserved.")?;
        let port = socket.local_addr()?.port();
        let endpoint = Endpoint {
            base: format!("http://127.0.0.1:{port}"),
            context,
            parallel,
            model_id: model.id.clone(),
            secret: format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple()),
        };
        let mut helper = runtime_command(launcher);
        helper
            .env("LLAMA_API_KEY", &endpoint.secret)
            .current_dir(executable.parent().context("Missing runtime directory")?)
            .arg("--supervise-runtime")
            .arg(executable)
            .args([
                "--host",
                "127.0.0.1",
                "--port",
                &port.to_string(),
                "--alias",
                &model.id,
                "--jinja",
                "--no-ui",
                "--no-agent",
                "--no-ui-mcp-proxy",
                "--cors-origins",
                "http://127.0.0.1",
                "--no-slots",
                "--parallel",
                &parallel.to_string(),
                "--ctx-size",
                &crate::memory::parallel_context(context, parallel).to_string(),
                "--reasoning-format",
                "deepseek",
            ])
            .arg("--model")
            .arg(&model.path)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        if parallel > 1 {
            helper.args(crate::memory::PARALLEL_BATCH_ARGS);
            helper.args([
                "--kv-unified",
                "--kv-unified-per-slot",
                &context.to_string(),
            ]);
        }
        helper.args(placement(backend, gpu_layers));
        drop(socket);
        let supervisor = helper
            .spawn()
            .context("Could not launch the runtime supervisor.")?;
        let mut managed = Server {
            supervisor,
            endpoint,
        };
        let deadline = Instant::now() + Duration::from_secs(240);
        loop {
            ensure!(!closing.is_cancelled(), "Parallax App is closing.");
            match managed.supervisor.try_wait()? {
                Some(status) if status.code() == Some(42) => break, // Bind lost to another local process.
                Some(_) => bail!("llama-server exited while loading the model. The model, memory, or driver may be incompatible."),
                None => {}
            }
            if ready(client, &managed.endpoint, &model.id).await && !managed.exited() {
                if parallel > 1 {
                    tokio::select! {
                        _ = closing.cancelled() => bail!("Parallax App is closing."),
                        result = validate_parallel(client, &managed.endpoint) => result?,
                    }
                }
                return Ok(managed);
            }
            ensure!(
                Instant::now() < deadline,
                "The model did not become ready in four minutes. Try a smaller model or context."
            );
            tokio::select! {
                _ = closing.cancelled() => bail!("Parallax App is closing."),
                _ = tokio::time::sleep(Duration::from_millis(350)) => {}
            }
        }
    }
    bail!("Could not bind the inference server after three attempts. Try loading again.")
}

async fn validate_parallel(client: &Client, endpoint: &Endpoint) -> Result<()> {
    let requests = (0..endpoint.parallel).map(|slot| async move {
        let response: serde_json::Value = client
            .post(format!("{}/completion", endpoint.base))
            .bearer_auth(&endpoint.secret)
            .timeout(Duration::from_secs(30))
            .json(&serde_json::json!({"prompt":format!("Say hello, worker {}.", slot + 1),"n_predict":2,"cache_prompt":false}))
            .send().await?.error_for_status()?.json().await?;
        ensure!(response["tokens_predicted"].as_u64().is_some_and(|tokens| tokens > 0), "Parallel inference did not generate a token.");
        Ok::<(), anyhow::Error>(())
    });
    futures_util::future::try_join_all(requests)
        .await
        .context("Concurrent inference startup check failed.")?;
    Ok(())
}

async fn ready(client: &Client, endpoint: &Endpoint, model_id: &str) -> bool {
    let request = async {
        let health = client
            .get(format!("{}/health", endpoint.base))
            .send()
            .await
            .ok()?;
        if !health.status().is_success() {
            return None;
        }
        // /health is public. The model listing must also reject an unauthenticated request.
        let url = format!("{}/v1/models", endpoint.base);
        let unauthenticated = client.get(&url).send().await.ok()?;
        if unauthenticated.status() != reqwest::StatusCode::UNAUTHORIZED {
            return None;
        }
        let model: serde_json::Value = client
            .get(url)
            .bearer_auth(&endpoint.secret)
            .send()
            .await
            .ok()?
            .error_for_status()
            .ok()?
            .json()
            .await
            .ok()?;
        Some(
            model["data"]
                .as_array()?
                .iter()
                .any(|item| item["id"] == model_id),
        )
    };
    tokio::time::timeout(Duration::from_secs(2), request)
        .await
        .ok()
        .flatten()
        .unwrap_or(false)
}

// This mode has no webview. The supervisor owns llama-server and watches the
// parent app's pipe. Closing or crashing the app closes the pipe in the kernel.
pub fn supervise(mut args: impl Iterator<Item = OsString>) -> i32 {
    let Some(runtime) = args.next() else {
        return 1;
    };
    let mut command = Command::new(runtime);
    background(&mut command);
    let mut child = match command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(_) => return 1,
    };
    let (parent_closed, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = std::io::stdin().lock().read(&mut [0]);
        let _ = parent_closed.send(());
    });
    let stderr = child.stderr.take();
    let bind_failed = std::thread::spawn(move || {
        use std::io::BufRead;
        stderr.is_some_and(|pipe| {
            let mut failed = false;
            for line in std::io::BufReader::new(pipe).lines().map_while(Result::ok) {
                let line = line.to_ascii_lowercase();
                failed |= line.contains("couldn't bind")
                    || line.contains("failed to bind")
                    || line.contains("address already in use");
            }
            failed
        })
    });
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                return if bind_failed.join().unwrap_or(false) {
                    42
                } else if status.success() {
                    0
                } else {
                    1
                }
            }
            Ok(None) => {}
            Err(_) => break,
        }
        if !matches!(
            receiver.recv_timeout(Duration::from_millis(90)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ) {
            break;
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};

    #[tokio::test]
    async fn readiness_checks_health_authentication_and_model_identity() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = Endpoint {
            base: format!("http://{}", listener.local_addr().unwrap()),
            secret: "test".into(),
            context: 1024,
            parallel: 1,
            model_id: "sample".into(),
        };
        std::thread::spawn(move || {
            for response in [
                "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                "HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                "HTTP/1.1 200 OK\r\nContent-Length: 26\r\nConnection: close\r\n\r\n{\"data\":[{\"id\":\"sample\"}]}",
            ] {
                let (mut socket, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(&socket);
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap() == 0 || line == "\r\n" { break; }
                }
                socket.write_all(response.as_bytes()).unwrap();
            }
        });
        let client = local_client().unwrap();
        assert!(!ready(&client, &endpoint, "sample").await);
        assert!(ready(&client, &endpoint, "sample").await);
    }
}

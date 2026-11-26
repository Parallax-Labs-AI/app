use anyhow::{ensure, Context, Result};
use cap_std::{
    ambient_authority,
    fs::{Dir, OpenOptions},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{HashMap, HashSet},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    sync::Mutex,
};
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

const FILE_LIMIT: usize = 16 * 1024;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Project {
    pub name: String,
    pub path: PathBuf,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    Deny,
    Once,
    Kind,
}

#[derive(Clone, Serialize)]
pub struct Approval {
    id: String,
    project: String,
    action: &'static str,
    path: String,
    content: Option<String>,
    remember: bool,
    command: Option<String>,
}

#[cfg(test)]
impl Approval {
    pub(crate) fn id(&self) -> &str {
        &self.id
    }
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
enum Kind {
    List,
    Read,
    Write,
    Create,
    Move,
    Delete,
    Terminal,
}

impl Kind {
    fn remember(self) -> bool {
        matches!(self, Self::List | Self::Read | Self::Terminal)
    }

    fn label(self) -> &'static str {
        match self {
            Self::List => "List project folder",
            Self::Read => "Read text file",
            Self::Write => "Create or replace text file",
            Self::Create => "Create text file",
            Self::Move => "Move or rename file",
            Self::Delete => "Delete file",
            Self::Terminal => "Run workspace command",
        }
    }
}

struct Operation {
    kind: Kind,
    path: PathBuf,
    content: Option<String>,
    command: Option<String>,
    destination: Option<PathBuf>,
}

fn project_path(path: &str, allow_root: bool) -> Result<PathBuf> {
    ensure!(
        !path.is_empty()
            && path.len() <= 1024
            && !path.contains([':', '\\'])
            && !path.chars().any(char::is_control),
        "Use a relative project path."
    );
    ensure!(
        (allow_root && path == ".")
            || Path::new(path).components().all(
                |part| matches!(part, Component::Normal(s) if !s.to_string_lossy().starts_with('.'))
            ),
        "Use a visible path inside the project."
    );
    Ok(path.into())
}

impl Operation {
    fn parse(name: &str, arguments: &str) -> Result<Self> {
        let kind = match name {
            "list_files" => Kind::List,
            "read_file" => Kind::Read,
            "write_file" => Kind::Write,
            "create_file" => Kind::Create,
            "move_file" => Kind::Move,
            "delete_file" => Kind::Delete,
            "workspace_command" => Kind::Terminal,
            _ => anyhow::bail!("Unknown workspace action."),
        };
        let args: Value = serde_json::from_str(arguments)?;
        let command = if kind == Kind::Terminal {
            let command = args["command"]
                .as_str()
                .context("Choose a workspace command.")?;
            ensure!(
                matches!(command, "pwd" | "ls" | "cat" | "wc"),
                "Available workspace commands are pwd, ls, cat, and wc."
            );
            Some(command.to_owned())
        } else {
            None
        };
        let path = project_path(
            args["path"]
                .as_str()
                .context("A relative path is required.")?,
            kind == Kind::List || matches!(command.as_deref(), Some("pwd" | "ls")),
        )?;
        let destination = if kind == Kind::Move {
            Some(project_path(
                args["destination"]
                    .as_str()
                    .context("A destination path is required.")?,
                false,
            )?)
        } else {
            None
        };
        let content = if matches!(kind, Kind::Write | Kind::Create) {
            let text = args["content"]
                .as_str()
                .context("Text content is required.")?;
            ensure!(
                text.len() <= FILE_LIMIT,
                "File content exceeds the workspace tool limit."
            );
            Some(text.into())
        } else {
            None
        };
        ensure!(
            args.as_object().is_some_and(|args| args.len()
                == if content.is_some() || command.is_some() || destination.is_some() {
                    2
                } else {
                    1
                }),
            "Unexpected tool arguments."
        );
        Ok(Self {
            kind,
            path,
            content,
            command,
            destination,
        })
    }

    fn execute(mut self, dir: Dir, root: &Path) -> Result<String> {
        if let Some(command) = self.command.take() {
            match command.as_str() {
                "pwd" => {
                    ensure!(self.path == Path::new("."), "Use . with pwd.");
                    return Ok(root.display().to_string());
                }
                "ls" => self.kind = Kind::List,
                "cat" | "wc" => self.kind = Kind::Read,
                _ => unreachable!(),
            }
            let result = self.execute(dir, root)?;
            return Ok(if command == "wc" {
                format!(
                    "{} lines, {} words, {} bytes",
                    result.lines().count(),
                    result.split_whitespace().count(),
                    result.len()
                )
            } else {
                result
            });
        }
        let path = &self.path;
        let resolved = match dir.canonicalize(path) {
            Ok(resolved) => resolved,
            Err(error)
                if error.kind() == std::io::ErrorKind::NotFound
                    && matches!(self.kind, Kind::Write | Kind::Create) =>
            {
                dir.canonicalize(
                    path.parent()
                        .filter(|p| !p.as_os_str().is_empty())
                        .unwrap_or(Path::new(".")),
                )?
                .join(path.file_name().context("Missing file name.")?)
            }
            Err(error) => return Err(error.into()),
        };
        ensure!(
            resolved
                .components()
                .filter(|part| *part != Component::CurDir)
                .eq(path.components().filter(|part| *part != Component::CurDir)),
            "Symbolic links are unavailable to workspace tools."
        );
        match self.kind {
            Kind::List => {
                let mut names = Vec::new();
                let mut truncated = false;
                for entry in dir.read_dir(path)?.take(257) {
                    let entry = entry?;
                    let name = entry.file_name().to_string_lossy().into_owned();
                    if names.len() == 128 {
                        truncated = true;
                        break;
                    }
                    if !name.starts_with('.')
                        && !name.chars().any(char::is_control)
                        && !entry.file_type()?.is_symlink()
                    {
                        names.push(if entry.file_type()?.is_dir() {
                            format!("{name}/")
                        } else {
                            name
                        });
                    }
                }
                names.sort_unstable();
                Ok(json!({"entries":names,"truncated":truncated}).to_string())
            }
            Kind::Read => {
                ensure!(
                    dir.symlink_metadata(path)?.is_file(),
                    "Select a regular text file."
                );
                let mut bytes = Vec::new();
                dir.open(path)?
                    .take((FILE_LIMIT + 1) as u64)
                    .read_to_end(&mut bytes)?;
                ensure!(
                    bytes.len() <= FILE_LIMIT,
                    "File exceeds the workspace tool limit."
                );
                Ok(String::from_utf8(bytes).context("Select a UTF-8 text file.")?)
            }
            Kind::Write | Kind::Create => {
                if let Ok(metadata) = dir.symlink_metadata(path) {
                    ensure!(metadata.is_file(), "Select a regular text file.");
                }
                let temporary = path.with_file_name(format!(".parallax-{}", uuid::Uuid::new_v4()));
                let mut options = OpenOptions::new();
                options.write(true).create_new(true);
                let mut file = dir.open_with(&temporary, &options)?;
                let result = file
                    .write_all(self.content.as_deref().unwrap_or_default().as_bytes())
                    .and_then(|()| file.sync_all());
                drop(file);
                let publish = result.and_then(|()| {
                    if self.kind == Kind::Create {
                        dir.hard_link(&temporary, &dir, path)
                    } else {
                        dir.rename(&temporary, &dir, path)
                    }
                });
                let _ = dir.remove_file(&temporary);
                publish?;
                Ok(format!("Saved {}", path.display()))
            }
            Kind::Move => {
                let destination = self.destination.context("Missing move destination.")?;
                ensure!(
                    dir.symlink_metadata(path)?.is_file(),
                    "Only regular files can be moved."
                );
                let parent = destination
                    .parent()
                    .filter(|path| !path.as_os_str().is_empty())
                    .unwrap_or(Path::new("."));
                ensure!(
                    dir.canonicalize(parent)?
                        .components()
                        .filter(|part| *part != Component::CurDir)
                        .eq(parent
                            .components()
                            .filter(|part| *part != Component::CurDir)),
                    "Symbolic links are unavailable to workspace tools."
                );
                dir.hard_link(path, &dir, &destination)?;
                if let Err(error) = dir.remove_file(path) {
                    let _ = dir.remove_file(&destination);
                    return Err(error.into());
                }
                Ok(format!(
                    "Moved {} to {}",
                    path.display(),
                    destination.display()
                ))
            }
            Kind::Terminal => unreachable!(),
            Kind::Delete => {
                ensure!(
                    dir.symlink_metadata(path)?.is_file(),
                    "Only regular files can be deleted."
                );
                dir.remove_file(path)?;
                Ok(format!("Deleted {}", path.display()))
            }
        }
    }
}

pub fn open_project(project: &Project) -> Result<(PathBuf, Dir)> {
    ensure!(
        project.path.is_absolute()
            && (1..=100).contains(&project.name.trim().chars().count())
            && !project.name.chars().any(char::is_control),
        "Select a project folder and enter a name."
    );
    let root = project.path.canonicalize()?;
    ensure!(root.parent().is_some(), "Choose a project folder.");
    let dir = Dir::open_ambient_dir(&root, ambient_authority())?;
    Ok((root, dir))
}

pub fn tools() -> Value {
    json!([
        {"type":"function","function":{"name":"workspace_command","description":"Run a built-in workspace command. Use . with pwd, a folder with ls, or a UTF-8 file with cat and wc.","parameters":{"type":"object","properties":{"command":{"type":"string","enum":["pwd","ls","cat","wc"]},"path":{"type":"string"}},"required":["command","path"],"additionalProperties":false}}},
        tool(
            "list_files",
            "List a folder inside this project. Use . for the project root.",
            false
        ),
        tool(
            "read_file",
            "Read one UTF-8 text file in this project.",
            false
        ),
        tool(
            "write_file",
            "Create or replace one project text file. Supply its complete new contents.",
            true
        ),
        tool("create_file", "Create a new project text file without replacing an existing file.", true),
        {"type":"function","function":{"name":"move_file","description":"Move or rename a regular project file to a new path. The destination must not exist.","parameters":{"type":"object","properties":{"path":{"type":"string"},"destination":{"type":"string"}},"required":["path","destination"],"additionalProperties":false}}},
        tool(
            "delete_file",
            "Permanently delete one project file. Only use when the user asks to delete it.",
            false
        ),
    ])
}

fn tool(name: &str, description: &str, write: bool) -> Value {
    let mut properties =
        json!({"path":{"type":"string","description":"Path relative to the project folder"}});
    let mut required = vec!["path"];
    if write {
        properties["content"] = json!({"type":"string"});
        required.push("content");
    }
    json!({"type":"function","function":{"name":name,"description":description,"parameters":{"type":"object","properties":properties,"required":required,"additionalProperties":false}}})
}

#[derive(Default)]
pub struct Access {
    pending: Mutex<HashMap<String, oneshot::Sender<Decision>>>,
    remembered: Mutex<HashSet<(String, PathBuf, Kind, Option<String>)>>,
}

impl Access {
    pub fn decide(&self, id: &str, decision: Decision) -> Result<()> {
        let sender = self
            .pending
            .lock()
            .unwrap()
            .remove(id)
            .context("This approval request has expired.")?;
        sender
            .send(decision)
            .map_err(|_| anyhow::anyhow!("This approval request has expired."))
    }

    pub fn reset(&self) {
        self.remembered.lock().unwrap().clear();
    }

    async fn request(
        &self,
        mut approval: Approval,
        cancel: &CancellationToken,
        emit: &impl Fn(crate::inference::ChatEvent) -> Result<()>,
    ) -> Decision {
        let id = uuid::Uuid::new_v4().to_string();
        approval.id = id.clone();
        let (sender, receiver) = oneshot::channel();
        self.pending.lock().unwrap().insert(id.clone(), sender);
        let decision = if emit(crate::inference::ChatEvent::AgentApproval { request: approval })
            .is_ok()
        {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => Decision::Deny,
                response = tokio::time::timeout(std::time::Duration::from_secs(300), receiver) => {
                    response.ok().and_then(Result::ok).unwrap_or(Decision::Deny)
                }
            }
        } else {
            Decision::Deny
        };
        self.pending.lock().unwrap().remove(&id);
        decision
    }

    pub async fn approve_external(
        &self,
        action: &'static str,
        scope: String,
        path: String,
        command: Option<String>,
        detail: Option<String>,
        cancel: &CancellationToken,
        emit: &impl Fn(crate::inference::ChatEvent) -> Result<()>,
    ) -> Result<()> {
        let request = Approval {
            id: String::new(),
            project: scope,
            action,
            path,
            command,
            content: detail,
            remember: false,
        };
        ensure!(
            matches!(self.request(request, cancel, emit).await, Decision::Once)
                && !cancel.is_cancelled(),
            "The user denied this action."
        );
        Ok(())
    }

    pub async fn run(
        &self,
        chat: &str,
        project: &Project,
        root: &Path,
        dir: &Dir,
        cancel: &CancellationToken,
        call: &crate::inference::ToolCall,
        policy: &crate::settings::Settings,
        emit: &impl Fn(crate::inference::ChatEvent) -> Result<()>,
    ) -> Result<String> {
        let operation = Operation::parse(&call.name, &call.arguments)?;
        let key = (
            chat.to_owned(),
            root.to_owned(),
            operation.kind,
            operation.command.clone(),
        );
        let allowed = self.remembered.lock().unwrap().contains(&key);
        if !(allowed
            || policy.agent_auto_read && matches!(operation.kind, Kind::List | Kind::Read)
            || policy.agent_auto_create && operation.kind == Kind::Create)
        {
            let mut path = root.join(&operation.path).display().to_string();
            if let Some(destination) = &operation.destination {
                path.push_str(&format!(" → {}", root.join(destination).display()));
            }
            let approval = Approval {
                id: String::new(),
                project: project.name.clone(),
                action: operation.kind.label(),
                path,
                content: operation.content.clone(),
                remember: operation.kind.remember(),
                command: operation.command.clone(),
            };
            match self.request(approval, cancel, emit).await {
                Decision::Once => {}
                Decision::Kind if operation.kind.remember() => {
                    self.remembered.lock().unwrap().insert(key);
                }
                _ => anyhow::bail!("The user denied this workspace action."),
            }
        }
        ensure!(!cancel.is_cancelled(), "Workspace action cancelled.");
        let dir = dir.try_clone()?;
        let root = root.to_owned();
        let cancel = cancel.clone();
        tokio::task::spawn_blocking(move || {
            ensure!(!cancel.is_cancelled(), "Workspace action cancelled.");
            operation.execute(dir, &root)
        })
        .await?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{ChatEvent, ToolCall};

    #[tokio::test]
    async fn create_can_be_auto_approved_but_move_asks_and_never_overwrites() {
        let temp = tempfile::tempdir().unwrap();
        let project = Project {
            name: "Test".into(),
            path: temp.path().to_owned(),
        };
        let (root, dir) = open_project(&project).unwrap();
        let access = Access::default();
        let cancel = CancellationToken::new();
        let policy = crate::settings::Settings {
            agent_auto_create: true,
            ..Default::default()
        };
        let create = ToolCall {
            id: "test".into(),
            name: "create_file".into(),
            arguments: json!({"path":"new.txt","content":"original"}).to_string(),
        };
        let no_prompt = |_| panic!("Non-overwriting creation is auto-approved.");
        access
            .run(
                "test", &project, &root, &dir, &cancel, &create, &policy, &no_prompt,
            )
            .await
            .unwrap();
        assert!(access
            .run("test", &project, &root, &dir, &cancel, &create, &policy, &no_prompt)
            .await
            .is_err());
        dir.write("occupied.txt", "keep").unwrap();
        let move_to = |destination| ToolCall {
            id: "test".into(),
            name: "move_file".into(),
            arguments: json!({"path":"new.txt","destination":destination}).to_string(),
        };
        let approve = |event| {
            let ChatEvent::AgentApproval { request } = event else {
                panic!("Missing approval.");
            };
            assert!(!request.remember);
            assert!(request.path.contains(" → "));
            access.decide(&request.id, Decision::Once)
        };
        assert!(access
            .run(
                "test",
                &project,
                &root,
                &dir,
                &cancel,
                &move_to("occupied.txt"),
                &policy,
                &approve
            )
            .await
            .is_err());
        assert_eq!(dir.read_to_string("occupied.txt").unwrap(), "keep");
        access
            .run(
                "test",
                &project,
                &root,
                &dir,
                &cancel,
                &move_to("moved.txt"),
                &policy,
                &approve,
            )
            .await
            .unwrap();
        assert!(!dir.exists("new.txt"));
        assert_eq!(dir.read_to_string("moved.txt").unwrap(), "original");
    }

    #[tokio::test]
    async fn approvals_are_scoped_and_mutations_require_allow_once() {
        let temp = tempfile::tempdir().unwrap();
        let project = Project {
            name: "Test".into(),
            path: temp.path().to_owned(),
        };
        let (root, dir) = open_project(&project).unwrap();
        let cancel = CancellationToken::new();
        let access = Access::default();
        let policy = crate::settings::Settings::default();
        let call = |name: &str, arguments: Value| ToolCall {
            id: "test".into(),
            name: name.into(),
            arguments: arguments.to_string(),
        };
        let write = call(
            "write_file",
            json!({"path":"notes.txt","content":"hello project\n"}),
        );
        let access_ref = &access;
        let accept = |decision| {
            move |event| {
                let ChatEvent::AgentApproval { request } = event else {
                    panic!("Unexpected event")
                };
                access_ref.decide(&request.id, decision)
            }
        };
        assert!(access
            .run(
                "chat",
                &project,
                &root,
                &dir,
                &cancel,
                &write,
                &policy,
                &accept(Decision::Kind)
            )
            .await
            .is_err());
        assert!(!dir.exists("notes.txt"));
        access
            .run(
                "chat",
                &project,
                &root,
                &dir,
                &cancel,
                &write,
                &policy,
                &accept(Decision::Once),
            )
            .await
            .unwrap();

        let read = call("read_file", json!({"path":"notes.txt"}));
        assert_eq!(
            access
                .run(
                    "chat",
                    &project,
                    &root,
                    &dir,
                    &cancel,
                    &read,
                    &policy,
                    &accept(Decision::Kind)
                )
                .await
                .unwrap(),
            "hello project\n"
        );
        let no_request = |_| panic!("Remembered approval should be reused");
        access
            .run(
                "chat",
                &project,
                &root,
                &dir,
                &cancel,
                &read,
                &policy,
                &no_request,
            )
            .await
            .unwrap();
        assert!(access
            .run(
                "other-chat",
                &project,
                &root,
                &dir,
                &cancel,
                &read,
                &policy,
                &accept(Decision::Deny)
            )
            .await
            .is_err());
        access.reset();
        assert!(access
            .run(
                "chat",
                &project,
                &root,
                &dir,
                &cancel,
                &read,
                &policy,
                &accept(Decision::Deny)
            )
            .await
            .is_err());

        let command = call(
            "workspace_command",
            json!({"command":"wc","path":"notes.txt"}),
        );
        assert_eq!(
            access
                .run(
                    "chat",
                    &project,
                    &root,
                    &dir,
                    &cancel,
                    &command,
                    &policy,
                    &accept(Decision::Once)
                )
                .await
                .unwrap(),
            "1 lines, 2 words, 14 bytes"
        );
        assert!(Operation::parse("workspace_command", r#"{"command":"sh","path":"."}"#).is_err());
        assert!(Operation::parse("read_file", r#"{"path":"../outside"}"#).is_err());
        let list = call("list_files", json!({"path":"."}));
        assert!(access
            .run(
                "chat",
                &project,
                &root,
                &dir,
                &cancel,
                &list,
                &policy,
                &accept(Decision::Once)
            )
            .await
            .unwrap()
            .contains("notes.txt"));

        let delete = call("delete_file", json!({"path":"notes.txt"}));
        assert!(access
            .run(
                "chat",
                &project,
                &root,
                &dir,
                &cancel,
                &delete,
                &policy,
                &accept(Decision::Kind)
            )
            .await
            .is_err());
        assert!(dir.exists("notes.txt"));
        access
            .run(
                "chat",
                &project,
                &root,
                &dir,
                &cancel,
                &delete,
                &policy,
                &accept(Decision::Once),
            )
            .await
            .unwrap();
        assert!(!dir.exists("notes.txt"));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_links_outside_the_project() {
        let project = tempfile::tempdir().unwrap();
        let external = tempfile::tempdir().unwrap();
        std::fs::write(external.path().join("outside.txt"), "outside").unwrap();
        std::os::unix::fs::symlink(external.path(), project.path().join("external")).unwrap();
        let (root, dir) = open_project(&Project {
            name: "Test".into(),
            path: project.path().to_owned(),
        })
        .unwrap();
        let operation = Operation::parse(
            "write_file",
            r#"{"path":"external/outside.txt","content":"changed"}"#,
        )
        .unwrap();
        assert!(operation.execute(dir, &root).is_err());
        assert_eq!(
            std::fs::read_to_string(external.path().join("outside.txt")).unwrap(),
            "outside"
        );
    }
}

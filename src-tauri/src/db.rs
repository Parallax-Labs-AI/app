use crate::settings::{ModelPreferences, Settings};
use anyhow::{ensure, Context, Result};
use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Mutex, MutexGuard},
};
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Installed {
    pub id: String,
    pub name: String,
    pub quantization: Option<String>,
    pub bytes: u64,
    pub path: PathBuf,
    pub architecture: String,
    pub context: Option<u32>,
    pub imported: bool,
    #[serde(default)]
    pub available_on_disk: bool,
}

impl Installed {
    pub fn present(&self) -> bool {
        std::fs::symlink_metadata(&self.path)
            .is_ok_and(|metadata| metadata.is_file() && metadata.len() == self.bytes)
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Conversation {
    pub id: String,
    pub title: String,
    pub system_prompt: String,
    pub updated: i64,
}

#[derive(Serialize)]
pub struct ProjectConversation {
    pub project: crate::agent::Project,
    pub conversation: Conversation,
}

#[derive(Debug, Serialize)]
pub struct Message {
    pub id: i64,
    pub role: String,
    pub content: String,
    pub reasoning: String,
    pub search_failed: bool,
    pub squadron: Vec<crate::squadron::Run>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Thread {
    pub conversation: Conversation,
    pub messages: Vec<Message>,
    pub project: Option<crate::agent::Project>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Turn {
    Send { content: String },
    Edit { message: i64, content: String },
    Regenerate,
}

pub struct Database(Mutex<Connection>);

fn read_project(db: &Connection, id: &str) -> Result<Option<crate::agent::Project>> {
    let project: Option<String> =
        db.query_row("SELECT project FROM chats WHERE id=?", [id], |row| {
            row.get(0)
        })?;
    project
        .map(|record| serde_json::from_str(&record).map_err(Into::into))
        .transpose()
}

fn read_conversation(row: &Row<'_>) -> rusqlite::Result<Conversation> {
    Ok(Conversation {
        id: row.get(0)?,
        title: row.get(1)?,
        system_prompt: row.get(2)?,
        updated: row.get(3)?,
    })
}

fn read_messages(db: &Connection, id: &str) -> Result<Vec<Message>> {
    let mut query = db.prepare(
        "SELECT id,role,content,reasoning,search_failed,squadron FROM messages WHERE chat=? ORDER BY id",
    )?;
    let rows = query.query_map([id], |r| {
        Ok(Message {
            id: r.get(0)?,
            role: r.get(1)?,
            content: r.get(2)?,
            reasoning: r.get(3)?,
            search_failed: r.get(4)?,
            squadron: serde_json::from_str(&r.get::<_, String>(5)?).map_err(|error| {
                rusqlite::Error::FromSqlConversionFailure(
                    5,
                    rusqlite::types::Type::Text,
                    Box::new(error),
                )
            })?,
        })
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

impl Database {
    pub fn open(path: &Path) -> Result<Self> {
        let db = Connection::open(path)?;
        db.execute_batch("PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000; PRAGMA secure_delete=ON;")?;
        let version: u32 = db.pragma_query_value(None, "user_version", |row| row.get(0))?;
        if version == 0 {
            db.execute_batch("BEGIN;
                CREATE TABLE chats(id TEXT PRIMARY KEY, title TEXT NOT NULL, system_prompt TEXT NOT NULL DEFAULT '', updated INTEGER NOT NULL);
                CREATE TABLE messages(id INTEGER PRIMARY KEY AUTOINCREMENT, chat TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE, role TEXT NOT NULL CHECK(role IN ('user','assistant')), content TEXT NOT NULL);
                CREATE INDEX messages_chat ON messages(chat,id);
                CREATE TABLE installed(id TEXT PRIMARY KEY, record TEXT NOT NULL);
                CREATE TABLE preferences(id INTEGER PRIMARY KEY CHECK(id=1), record TEXT NOT NULL);
                CREATE TABLE pending_downloads(id TEXT PRIMARY KEY, directory TEXT NOT NULL);
                PRAGMA user_version=1; COMMIT;")?;
        }
        ensure!(version <= 5, "This database requires a newer Parallax App.");
        if version < 2 {
            db.execute_batch(
                "BEGIN;
                ALTER TABLE messages ADD COLUMN reasoning TEXT NOT NULL DEFAULT '';
                CREATE TABLE model_preferences(id TEXT PRIMARY KEY, record TEXT NOT NULL);
                PRAGMA user_version=2; COMMIT;",
            )?;
        }
        if version < 3 {
            db.execute_batch(
                "BEGIN;
                ALTER TABLE messages ADD COLUMN search_failed INTEGER NOT NULL DEFAULT 0;
                PRAGMA user_version=3; COMMIT;",
            )?;
        }
        if version < 4 {
            db.execute_batch(
                "BEGIN; ALTER TABLE chats ADD COLUMN project TEXT; PRAGMA user_version=4; COMMIT;",
            )?;
        }
        if version < 5 {
            db.execute_batch("BEGIN; ALTER TABLE messages ADD COLUMN squadron TEXT NOT NULL DEFAULT '[]'; PRAGMA user_version=5; COMMIT;")?;
        }
        Ok(Self(Mutex::new(db)))
    }

    fn db(&self) -> Result<MutexGuard<'_, Connection>> {
        self.0
            .lock()
            .map_err(|_| anyhow::anyhow!("Local database is busy. Restart Parallax App."))
    }

    pub fn settings(&self) -> Result<Settings> {
        let record: Option<String> = self
            .db()?
            .query_row("SELECT record FROM preferences WHERE id=1", [], |r| {
                r.get(0)
            })
            .optional()?;
        record
            .map(|s| serde_json::from_str(&s).map_err(Into::into))
            .unwrap_or_else(|| Ok(Settings::default()))
    }

    pub fn save_settings(&self, settings: &Settings) -> Result<()> {
        settings.validate()?;
        self.db()?.execute("INSERT INTO preferences VALUES(1,?) ON CONFLICT(id) DO UPDATE SET record=excluded.record", [serde_json::to_string(settings)?])?;
        Ok(())
    }

    pub fn model_preferences(&self) -> Result<HashMap<String, ModelPreferences>> {
        let db = self.db()?;
        let mut query = db.prepare("SELECT id,record FROM model_preferences")?;
        let rows = query.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        rows.map(|row| {
            let (id, record) = row?;
            Ok((id, serde_json::from_str(&record)?))
        })
        .collect()
    }

    pub fn preferences_for(&self, id: &str) -> Result<ModelPreferences> {
        let record: Option<String> = self
            .db()?
            .query_row(
                "SELECT record FROM model_preferences WHERE id=?",
                [id],
                |row| row.get(0),
            )
            .optional()?;
        match record {
            Some(record) => Ok(serde_json::from_str(&record)?),
            None => Ok(ModelPreferences::legacy(self.settings()?)),
        }
    }

    pub fn save_model_preferences(&self, id: &str, settings: &ModelPreferences) -> Result<()> {
        self.db()?.execute(
            "INSERT INTO model_preferences VALUES(?,?) ON CONFLICT(id) DO UPDATE SET record=excluded.record",
            params![id, serde_json::to_string(settings)?],
        )?;
        Ok(())
    }

    pub fn installed(&self) -> Result<Vec<Installed>> {
        let db = self.db()?;
        let mut query = db.prepare("SELECT record FROM installed ORDER BY rowid")?;
        let rows = query.query_map([], |r| r.get::<_, String>(0))?;
        rows.map(|row| {
            let mut model: Installed = serde_json::from_str(&row?)?;
            model.available_on_disk = model.present();
            Ok(model)
        })
        .collect()
    }

    pub fn register(&self, model: &Installed) -> Result<()> {
        let mut db = self.db()?;
        let txn = db.transaction()?;
        txn.execute("INSERT INTO installed(id,record) VALUES(?,?) ON CONFLICT(id) DO UPDATE SET record=excluded.record",
            params![model.id, serde_json::to_string(model)?])?;
        txn.execute("DELETE FROM pending_downloads WHERE id=?", [&model.id])?;
        txn.commit()?;
        Ok(())
    }

    pub fn remove_model(&self, id: &str) -> Result<()> {
        self.db()?
            .execute("DELETE FROM installed WHERE id=?", [id])?;
        Ok(())
    }

    pub fn pending(&self) -> Result<Vec<(String, PathBuf)>> {
        let db = self.db()?;
        let mut query = db.prepare("SELECT id,directory FROM pending_downloads")?;
        let rows = query.query_map([], |row| {
            Ok((row.get(0)?, PathBuf::from(row.get::<_, String>(1)?)))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn track_pending(&self, id: &str, dir: &Path) -> Result<()> {
        self.db()?.execute(
            "INSERT INTO pending_downloads VALUES(?,?) ON CONFLICT(id) DO NOTHING",
            params![
                id,
                dir.to_str()
                    .context("The selected folder path is not valid Unicode.")?
            ],
        )?;
        Ok(())
    }

    pub fn remove_pending(&self, id: &str) -> Result<()> {
        self.db()?
            .execute("DELETE FROM pending_downloads WHERE id=?", [id])?;
        Ok(())
    }

    pub fn create_chat(&self) -> Result<Conversation> {
        let id = Uuid::new_v4().to_string();
        let prompt = self.settings()?.default_system_prompt;
        let db = self.db()?;
        db.execute(
            "INSERT INTO chats(id,title,system_prompt,updated) VALUES(?,'New chat',?,unixepoch())",
            params![id, prompt],
        )?;
        Ok(db.query_row(
            "SELECT id,title,system_prompt,updated FROM chats WHERE id=?",
            [&id],
            read_conversation,
        )?)
    }

    pub fn list(&self, search: &str) -> Result<Vec<Conversation>> {
        ensure!(search.len() <= 512, "Search is too long.");
        let db = self.db()?;
        let mut query = db.prepare(
            "SELECT id,title,system_prompt,updated FROM chats
            WHERE project IS NULL AND (?1='' OR instr(lower(title),lower(?1))>0 OR EXISTS(
                SELECT 1 FROM messages WHERE chat=chats.id AND (instr(lower(content),lower(?1))>0 OR instr(lower(reasoning),lower(?1))>0)))
            ORDER BY updated DESC,rowid DESC",
        )?;
        let rows = query.query_map([search], read_conversation)?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn project_conversations(&self) -> Result<Vec<ProjectConversation>> {
        let db = self.db()?;
        let mut statement = db.prepare("SELECT id,title,system_prompt,updated,project FROM chats WHERE project IS NOT NULL ORDER BY updated DESC,rowid DESC")?;
        let rows = statement.query_map([], |row| {
            Ok((read_conversation(row)?, row.get::<_, String>(4)?))
        })?;
        rows.map(|row| {
            let (conversation, record) = row?;
            Ok(ProjectConversation {
                conversation,
                project: serde_json::from_str(&record)?,
            })
        })
        .collect()
    }

    pub fn open_project_conversation(&self, project: &crate::agent::Project) -> Result<Thread> {
        let prompt = self.settings()?.default_system_prompt;
        let record = serde_json::to_string(project)?;
        let id = {
            let db = self.db()?;
            let mut statement = db.prepare("SELECT id,project FROM chats WHERE project IS NOT NULL ORDER BY updated DESC,rowid DESC")?;
            let rows = statement.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            let mut existing = None;
            for row in rows {
                let (id, record) = row?;
                let saved: crate::agent::Project = serde_json::from_str(&record)?;
                if saved.path == project.path
                    || saved
                        .path
                        .canonicalize()
                        .is_ok_and(|path| path == project.path)
                {
                    existing = Some(id);
                    break;
                }
            }
            if let Some(id) = existing {
                id
            } else {
                let id = Uuid::new_v4().to_string();
                db.execute("INSERT INTO chats(id,title,system_prompt,updated,project) VALUES(?,'New chat',?,unixepoch(),?)", params![id, prompt, record])?;
                id
            }
        };
        self.thread(&id)
    }

    pub fn thread(&self, id: &str) -> Result<Thread> {
        let db = self.db()?;
        let conversation = db
            .query_row(
                "SELECT id,title,system_prompt,updated FROM chats WHERE id=?",
                [id],
                read_conversation,
            )
            .optional()?
            .context("This conversation no longer exists.")?;
        Ok(Thread {
            conversation,
            messages: read_messages(&db, id)?,
            project: read_project(&db, id)?,
        })
    }

    pub fn project(&self, id: &str) -> Result<Option<crate::agent::Project>> {
        let db = self.db()?;
        read_project(&db, id)
    }

    pub fn bind_project(&self, id: &str, project: &crate::agent::Project) -> Result<()> {
        let record = serde_json::to_string(project)?;
        ensure!(self.db()?.execute(
            "UPDATE chats SET project=?1 WHERE id=?2 AND project IS NULL AND NOT EXISTS (SELECT 1 FROM messages WHERE chat=?2)",
            params![record, id],
        )? == 1, "Start a new Agent chat to choose a project.");
        Ok(())
    }

    pub fn save_generated_title(&self, id: &str, title: &str) -> Result<()> {
        self.db()?.execute(
            "UPDATE chats SET title=? WHERE id=? AND title='New chat'",
            params![title, id],
        )?;
        Ok(())
    }

    pub fn update_chat(&self, id: &str, title: &str, prompt: &str) -> Result<()> {
        ensure!(
            !title.trim().is_empty() && title.chars().count() <= 120 && prompt.len() <= 16_000,
            "Enter a title up to 120 characters and a shorter system prompt."
        );
        ensure!(
            self.db()?.execute(
                "UPDATE chats SET title=?,system_prompt=?,updated=unixepoch() WHERE id=?",
                params![title.trim(), prompt, id]
            )? == 1,
            "This conversation no longer exists."
        );
        Ok(())
    }

    pub fn delete_chat(&self, id: Option<&str>) -> Result<()> {
        let db = self.db()?;
        match id {
            Some(id) => {
                db.execute("DELETE FROM chats WHERE id=?", [id])?;
            }
            None => {
                db.execute("DELETE FROM chats", [])?;
            }
        }
        db.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
        Ok(())
    }

    pub fn export(&self) -> Result<Vec<Thread>> {
        let db = self.db()?;
        let mut query =
            db.prepare("SELECT id,title,system_prompt,updated FROM chats ORDER BY updated DESC")?;
        let rows = query.query_map([], read_conversation)?;
        rows.map(|row| {
            let conversation = row?;
            let messages = read_messages(&db, &conversation.id)?;
            let project = read_project(&db, &conversation.id)?;
            Ok(Thread {
                project,
                conversation,
                messages,
            })
        })
        .collect()
    }

    pub fn begin_turn(&self, id: &str, turn: Turn) -> Result<(i64, Vec<serde_json::Value>)> {
        let mut db = self.db()?;
        let txn = db.transaction()?;
        let conversation = txn
            .query_row(
                "SELECT id,title,system_prompt,updated FROM chats WHERE id=?",
                [id],
                read_conversation,
            )
            .optional()?
            .context("This conversation no longer exists.")?;
        match turn {
            Turn::Send { content } => {
                let content = prompt_content(&content)?;
                // Merge a new prompt after a stopped/failed turn that yielded no answer.
                let last: Option<(i64, String, String)> = txn.query_row(
                    "SELECT id,role,content FROM messages WHERE chat=? ORDER BY id DESC LIMIT 1",
                    [id], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional()?;
                match last {
                    Some((message, role, previous)) if role == "user" => {
                        let combined = format!("{previous}\n\n{content}");
                        prompt_content(&combined)?;
                        txn.execute(
                            "UPDATE messages SET content=? WHERE id=?",
                            params![combined, message],
                        )?;
                    }
                    _ => {
                        txn.execute(
                            "INSERT INTO messages(chat,role,content) VALUES(?,'user',?)",
                            params![id, content],
                        )?;
                    }
                }
            }
            Turn::Edit { message, content } => {
                let content = prompt_content(&content)?;
                ensure!(
                    txn.execute(
                        "UPDATE messages SET content=? WHERE id=? AND chat=? AND role='user'",
                        params![content, message, id]
                    )? == 1,
                    "Only a user message in this conversation can be edited."
                );
                txn.execute(
                    "DELETE FROM messages WHERE chat=? AND id>?",
                    params![id, message],
                )?;
            }
            Turn::Regenerate => {
                let last_user: Option<i64> = txn.query_row(
                    "SELECT MAX(id) FROM messages WHERE chat=? AND role='user'",
                    [id],
                    |r| r.get(0),
                )?;
                let last_user = last_user.context("Send a message before regenerating.")?;
                txn.execute(
                    "DELETE FROM messages WHERE chat=? AND id>?",
                    params![id, last_user],
                )?;
            }
        }
        let mut history = vec![];
        if !conversation.system_prompt.trim().is_empty() {
            history.push(serde_json::json!({"role":"system","content":conversation.system_prompt}));
        }
        for message in read_messages(&txn, id)? {
            if !message.content.is_empty() {
                history.push(serde_json::json!({"role":message.role,"content":message.content}));
            }
        }
        txn.execute("UPDATE chats SET updated=unixepoch() WHERE id=?", [id])?;
        txn.execute(
            "INSERT INTO messages(chat,role,content) VALUES(?,'assistant','')",
            [id],
        )?;
        let reply = txn.last_insert_rowid();
        txn.commit()?;
        Ok((reply, history))
    }

    pub fn write_reply(
        &self,
        message: i64,
        text: &str,
        reasoning: &str,
        final_write: bool,
    ) -> Result<()> {
        let db = self.db()?;
        if final_write && text.is_empty() && reasoning.is_empty() {
            db.execute(
                "DELETE FROM messages WHERE id=? AND role='assistant' AND search_failed=0 AND squadron='[]'",
                [message],
            )?;
        } else {
            db.execute(
                "UPDATE messages SET content=?,reasoning=? WHERE id=? AND role='assistant'",
                params![text, reasoning, message],
            )?;
        }
        Ok(())
    }

    pub fn save_squadron(&self, message: i64, runs: &[crate::squadron::Run]) -> Result<()> {
        self.db()?.execute(
            "UPDATE messages SET squadron=? WHERE id=? AND role='assistant'",
            params![serde_json::to_string(runs)?, message],
        )?;
        Ok(())
    }

    pub fn set_search_failed(&self, message: i64, failed: bool) -> Result<()> {
        self.db()?.execute(
            "UPDATE messages SET search_failed=? WHERE id=? AND role='assistant'",
            params![failed, message],
        )?;
        Ok(())
    }

    pub fn recover_interrupted(&self) -> Result<()> {
        self.db()?.execute(
            "DELETE FROM messages WHERE role='assistant' AND content='' AND reasoning='' AND search_failed=0 AND squadron='[]'",
            [],
        )?;
        Ok(())
    }
}

fn prompt_content(content: &str) -> Result<&str> {
    let trimmed = content.trim();
    ensure!(
        !trimmed.is_empty() && trimmed.len() <= 128_000,
        "Enter a message up to 128 KB."
    );
    Ok(trimmed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_titles_leave_manual_names_intact() {
        let db = Database::open(Path::new(":memory:")).unwrap();
        let chat = db.create_chat().unwrap();
        db.begin_turn(
            &chat.id,
            Turn::Send {
                content: "Can you explain how Mars moves?".into(),
            },
        )
        .unwrap();
        assert_eq!(db.thread(&chat.id).unwrap().conversation.title, "New chat");
        db.save_generated_title(&chat.id, "The Orbit of Mars")
            .unwrap();
        assert_eq!(
            db.thread(&chat.id).unwrap().conversation.title,
            "The Orbit of Mars"
        );
        db.update_chat(&chat.id, "Astronomy", "").unwrap();
        db.save_generated_title(&chat.id, "Mars in Motion").unwrap();
        assert_eq!(db.thread(&chat.id).unwrap().conversation.title, "Astronomy");
    }

    #[test]
    fn reopening_a_project_restores_its_chat_and_squadron_activity() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("app.sqlite");
        let db = Database::open(&path).unwrap();
        let ordinary = db.create_chat().unwrap();
        let project = crate::agent::Project {
            name: "Design".into(),
            path: temp.path().canonicalize().unwrap(),
        };
        let id = db
            .open_project_conversation(&project)
            .unwrap()
            .conversation
            .id;
        let (message, _) = db
            .begin_turn(
                &id,
                Turn::Send {
                    content: "Review the project".into(),
                },
            )
            .unwrap();
        db.save_squadron(
            message,
            &[vec![crate::squadron::Worker {
                assignment: "Review the documentation".into(),
                query: None,
                outcome: Some(crate::squadron::Outcome {
                    status: crate::squadron::Status::Complete,
                    report: "Documentation reviewed".into(),
                }),
            }]],
        )
        .unwrap();
        db.write_reply(message, "", "", true).unwrap();
        drop(db);

        let db = Database::open(&path).unwrap();
        db.recover_interrupted().unwrap();
        assert_eq!(db.list("").unwrap()[0].id, ordinary.id);
        assert!(db.list("Review the project").unwrap().is_empty());
        let thread = db.open_project_conversation(&project).unwrap();
        assert_eq!(thread.conversation.id, id);
        assert_eq!(thread.project, Some(project));
        assert_eq!(db.project_conversations().unwrap().len(), 1);
        let worker = &thread.messages[1].squadron[0][0];
        assert_eq!(
            worker.outcome.as_ref().unwrap().report,
            "Documentation reviewed"
        );
    }

    #[test]
    fn failed_search_survives_an_empty_reply_and_database_recovery() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("app.sqlite");
        let db = Database::open(&path).unwrap();
        let chat = db.create_chat().unwrap();
        let (message, _) = db
            .begin_turn(
                &chat.id,
                Turn::Send {
                    content: "Search the web for Apollo 11.".into(),
                },
            )
            .unwrap();
        db.set_search_failed(message, true).unwrap();
        db.write_reply(message, "", "", true).unwrap();
        drop(db);
        let reopened = Database::open(&path).unwrap();
        reopened.recover_interrupted().unwrap();
        assert!(reopened.thread(&chat.id).unwrap().messages[1].search_failed);
    }

    #[test]
    fn default_system_prompt_only_applies_to_new_chats() {
        let db = Database::open(Path::new(":memory:")).unwrap();
        let before = db.create_chat().unwrap();
        let mut settings = db.settings().unwrap();
        settings.default_system_prompt = "Answer briefly.".into();
        db.save_settings(&settings).unwrap();
        let after = db.create_chat().unwrap();
        assert!(db
            .thread(&before.id)
            .unwrap()
            .conversation
            .system_prompt
            .is_empty());
        assert_eq!(after.system_prompt, "Answer briefly.");
        db.update_chat(&after.id, &after.title, "Answer in Spanish.")
            .unwrap();
        let (_, messages) = db
            .begin_turn(
                &after.id,
                Turn::Send {
                    content: "Hello".into(),
                },
            )
            .unwrap();
        assert_eq!(
            messages[0],
            serde_json::json!({"role":"system","content":"Answer in Spanish."})
        );
    }

    #[test]
    fn version_one_history_and_global_preferences_migrate() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("app.sqlite");
        let old = Connection::open(&path).unwrap();
        old.execute_batch(
            "CREATE TABLE chats(id TEXT PRIMARY KEY,title TEXT,system_prompt TEXT,updated INTEGER);
            CREATE TABLE messages(id INTEGER PRIMARY KEY,chat TEXT,role TEXT,content TEXT);
            CREATE TABLE preferences(id INTEGER PRIMARY KEY,record TEXT);
            INSERT INTO chats VALUES('old','Original','','0');
            INSERT INTO messages VALUES(1,'old','assistant','Older reply');
            INSERT INTO preferences VALUES(1,'{\"context\":8192,\"thinking\":true}');
            PRAGMA user_version=1;",
        )
        .unwrap();
        drop(old);
        let upgraded = Database::open(&path).unwrap();
        let saved = upgraded.thread("old").unwrap();
        assert_eq!(saved.messages[0].content, "Older reply");
        assert!(saved.messages[0].reasoning.is_empty());
        let options = upgraded.preferences_for("old-model").unwrap();
        assert_eq!(options.custom_tokens, 8192);
        assert_eq!(options.reasoning, crate::settings::Reasoning::Max);
    }

    #[test]
    fn chat_persists_and_edit_discards_the_following_branch() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("app.sqlite");
        let db = Database::open(&path).unwrap();
        let chat = db.create_chat().unwrap();
        let (reply, _) = db
            .begin_turn(
                &chat.id,
                Turn::Send {
                    content: "First prompt".into(),
                },
            )
            .unwrap();
        db.write_reply(reply, "Saved response", "Saved reasoning", true)
            .unwrap();
        drop(db);
        let db = Database::open(&path).unwrap();
        assert_eq!(db.list("saved response").unwrap().len(), 1);
        let thread = db.thread(&chat.id).unwrap();
        assert_eq!(thread.messages[1].content, "Saved response");
        assert_eq!(thread.messages[1].reasoning, "Saved reasoning");
        let (reply, history) = db
            .begin_turn(
                &chat.id,
                Turn::Edit {
                    message: thread.messages[0].id,
                    content: "Revised".into(),
                },
            )
            .unwrap();
        db.write_reply(reply, "", "", true).unwrap();
        assert_eq!(
            history,
            vec![serde_json::json!({"role":"user","content":"Revised"})]
        );
        assert_eq!(db.thread(&chat.id).unwrap().messages.len(), 1);
    }
}

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Settings {
    pub onboarded: bool,
    pub web_search: bool,
    pub custom_mcp: Vec<crate::extensions::Program>,
    pub agent_commands: Vec<crate::extensions::Program>,
    pub agent_auto_read: bool,
    pub squadron_enabled: bool,
    pub squadron_prefer_workers: bool,
    pub squadron_max_slots: u32,
    pub agent_auto_create: bool,
    pub default_system_prompt: String,
    pub theme: String,
    pub reduced_motion: bool,
    pub chat_text_size: String,
    pub accent: String,
    pub custom_accent: String,
    pub default_model: Option<String>,
    pub autoload: bool,
    pub model_directory: Option<PathBuf>,
    pub acceleration: String,
    pub context: u32,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub gpu_layers: Option<u32>,
    #[serde(default)]
    pub thinking: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            onboarded: false,
            web_search: true,
            custom_mcp: Vec::new(),
            agent_commands: Vec::new(),
            agent_auto_read: false,
            squadron_enabled: true,
            squadron_prefer_workers: false,
            squadron_max_slots: 8,
            agent_auto_create: false,
            default_system_prompt: String::new(),
            theme: "dark".into(),
            reduced_motion: false,
            chat_text_size: "medium".into(),
            accent: "blue".into(),
            custom_accent: "#607CFF".into(),
            default_model: None,
            autoload: false,
            model_directory: None,
            acceleration: "auto".into(),
            context: 4096,
            max_tokens: 1024,
            temperature: 0.7,
            top_p: 0.95,
            top_k: 40,
            gpu_layers: None,
            thinking: false,
        }
    }
}

impl Settings {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (2..=8).contains(&self.squadron_max_slots),
            "Choose between 2 and 8 Squadron workers."
        );
        crate::extensions::validate(&self.custom_mcp, true)?;
        crate::extensions::validate(&self.agent_commands, false)?;
        ensure!(
            self.default_system_prompt.len() <= 16_000,
            "The default system prompt is too long."
        );
        ensure!(
            matches!(self.theme.as_str(), "dark" | "light" | "system"),
            "Choose a supported theme."
        );
        ensure!(
            matches!(self.chat_text_size.as_str(), "small" | "medium" | "large"),
            "Choose a supported chat text size."
        );
        ensure!(
            matches!(self.accent.as_str(), "blue" | "violet" | "teal" | "custom"),
            "Choose a supported accent color."
        );
        ensure!(
            self.custom_accent.strip_prefix('#').is_some_and(
                |hex| hex.len() == 6 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
            ),
            "Use a six-digit hex accent color."
        );
        ensure!(
            matches!(
                self.acceleration.as_str(),
                "auto" | "cpu" | "metal" | "cuda" | "vulkan"
            ),
            "Choose a supported acceleration mode."
        );
        ensure!(
            [1024, 2048, 4096, 8192, 16384, 32768].contains(&self.context),
            "Choose a supported context size."
        );
        ensure!(
            (16..=65_536).contains(&self.max_tokens),
            "Choose between 16 and 65,536 output tokens."
        );
        ensure!(
            (0.0..=2.0).contains(&self.temperature)
                && (0.01..=1.0).contains(&self.top_p)
                && self.top_k <= 200
                && self.gpu_layers.is_none_or(|n| n <= 256),
            "Sampling or offload setting is out of range."
        );
        if let Some(path) = &self.model_directory {
            ensure!(
                path.is_absolute() && path.is_dir(),
                "Choose an existing model storage folder."
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextMode {
    #[default]
    Recommended,
    Most,
    Custom,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Reasoning {
    #[default]
    Disabled,
    Low,
    Max,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ModelPreferences {
    pub context_mode: ContextMode,
    pub custom_tokens: u32,
    pub allow_over_budget: bool,
    pub reasoning: Reasoning,
}

impl Default for ModelPreferences {
    fn default() -> Self {
        Self {
            context_mode: ContextMode::Recommended,
            custom_tokens: 4096,
            allow_over_budget: false,
            reasoning: Reasoning::Disabled,
        }
    }
}

impl ModelPreferences {
    pub fn legacy(settings: Settings) -> Self {
        let defaults = Settings::default();
        Self {
            context_mode: if settings.context == defaults.context {
                ContextMode::Recommended
            } else {
                ContextMode::Custom
            },
            custom_tokens: settings.context,
            reasoning: if settings.thinking {
                Reasoning::Max
            } else {
                Reasoning::Disabled
            },
            ..Self::default()
        }
    }

    pub fn validate(&self, model_limit: u32) -> Result<()> {
        if self.context_mode == ContextMode::Custom {
            ensure!(
                (crate::memory::MIN_CONTEXT..=model_limit).contains(&self.custom_tokens),
                "Choose a context between {} and {model_limit} tokens for this model.",
                crate::memory::MIN_CONTEXT
            );
        }
        Ok(())
    }
}

use crate::{
    db::Installed,
    hardware,
    settings::{ContextMode, ModelPreferences},
};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path, process::Stdio, time::Duration};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

const MIB: u64 = 1024 * 1024;
pub const MIN_CONTEXT: u32 = 512;
pub const APP_CONTEXT_LIMIT: u32 = 1_048_576;
const STEP: u32 = 256;
const RECOMMENDED_LIMIT: u32 = 32_768;
const STANDARD_CONTEXT: u32 = 4096;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryRow {
    pub kind: String,
    pub name: String,
    pub description: Option<String>,
    pub total: Option<u64>,
    pub free: Option<u64>,
    #[serde(rename = "self")]
    pub used: u64,
    pub model: u64,
    pub context: u64,
    pub compute: u64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Measurement {
    pub tokens: u32,
    pub rows: Vec<MemoryRow>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Profile {
    pub backend: String,
    pub unified: bool,
    pub system_total: u64,
    pub system_available: u64,
    pub model_limit: u32,
    pub recommended: Option<Measurement>,
    pub maximum: Option<Measurement>,
    pub estimated_max_before_startup: Option<u32>,
    pub minimum: Option<Measurement>,
    pub custom: Option<Measurement>,
}

pub fn limit(model: &Installed) -> u32 {
    model.context.unwrap_or(32_768).min(APP_CONTEXT_LIMIT)
}

fn headroom(free: u64, suggested: bool, system: bool) -> u64 {
    if suggested {
        1024.max(free / 5)
    } else if system {
        512
    } else {
        256
    }
}

impl Profile {
    pub fn validate_custom(&self, preferences: &ModelPreferences) -> Result<()> {
        if preferences.context_mode != ContextMode::Custom {
            return Ok(());
        }
        ensure!(
            preferences.custom_tokens <= self.model_limit,
            "The custom context exceeds this model’s {}-token limit.",
            self.model_limit
        );
        if preferences.allow_over_budget {
            return Ok(());
        }
        let maximum = self.maximum.as_ref().context("No context fits the measured memory budget. Free up memory, or explicitly allow a larger custom context.")?;
        ensure!(preferences.custom_tokens <= maximum.tokens, "This device’s measured limit is {} tokens. Choose a smaller context or enable the custom override.", maximum.tokens);
        Ok(())
    }

    pub fn context(&self, preferences: &ModelPreferences) -> Result<u32> {
        self.validate_custom(preferences)?;
        match preferences.context_mode {
            ContextMode::Recommended => self.recommended.as_ref().map(|m| m.tokens)
                .context("No recommended context fits the available memory. Try a smaller quantization or free up memory."),
            ContextMode::Most => self.maximum.as_ref().map(|m| m.tokens)
                .context("No tested context fits the available memory. Try a smaller quantization or free up memory."),
            ContextMode::Custom => {
                if !preferences.allow_over_budget {
                    let sample = self.custom.as_ref().context("The custom context has not been measured.")?;
                    ensure!(sample.tokens == preferences.custom_tokens && self.fits(sample, false),
                        "This custom context does not fit the current memory budget.");
                }
                Ok(preferences.custom_tokens)
            }
        }
    }

    pub fn max_retry_sizes(&self) -> Vec<u32> {
        let Some(maximum) = &self.maximum else {
            return Vec::new();
        };
        let mut sizes = [
            maximum.tokens * 7 / 8 / STEP * STEP,
            maximum.tokens * 3 / 4 / STEP * STEP,
            self.recommended
                .as_ref()
                .map_or(MIN_CONTEXT, |sample| sample.tokens),
            MIN_CONTEXT,
        ];
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        let mut candidates = Vec::new();
        for size in sizes {
            if (MIN_CONTEXT..maximum.tokens).contains(&size) && candidates.last() != Some(&size) {
                candidates.push(size);
            }
        }
        candidates
    }

    pub fn record_startup_limit(&mut self, sample: Measurement) {
        let Some(previous) = &self.maximum else {
            return;
        };
        if sample.tokens >= previous.tokens {
            return;
        }
        self.estimated_max_before_startup = Some(previous.tokens);
        if self
            .recommended
            .as_ref()
            .is_some_and(|recommended| recommended.tokens > sample.tokens)
        {
            self.recommended = Some(sample.clone());
        }
        self.maximum = Some(sample);
    }

    pub async fn remeasure(
        &self,
        executable: &Path,
        model: &Installed,
        gpu_layers: Option<u32>,
        tokens: u32,
        cancel: &CancellationToken,
    ) -> Result<Measurement> {
        if let Some(known) = [&self.minimum, &self.recommended, &self.maximum]
            .into_iter()
            .flatten()
            .find(|sample| sample.tokens == tokens)
        {
            return Ok(known.clone());
        }
        let mut samples = BTreeMap::new();
        let measurement = sample(
            &mut samples,
            executable,
            model,
            &self.backend,
            gpu_layers,
            tokens,
            cancel,
        )
        .await?;
        ensure!(
            self.fits(measurement, false),
            "The smaller context does not fit the measured memory budget."
        );
        Ok(measurement.clone())
    }

    fn fits(&self, sample: &Measurement, suggested: bool) -> bool {
        let mut shared = 0u64;
        for row in &sample.rows {
            if row.kind != "device" || self.unified {
                shared = shared.saturating_add(row.used);
            }
            if row.kind == "device" && row.used > 0 {
                let Some(free) = row
                    .free
                    .filter(|_| row.total.is_some_and(|total| total > 0))
                else {
                    return false;
                };
                if row.used > free.saturating_sub(headroom(free, suggested, false)) {
                    return false;
                }
            }
        }
        shared == 0
            || shared
                <= self.system_available.saturating_sub(headroom(
                    self.system_available,
                    suggested,
                    true,
                ))
    }
}

pub async fn profile(
    profiler: &Path,
    model: &Installed,
    backend: &str,
    gpu_layers: Option<u32>,
    custom: Option<u32>,
    cancel: &CancellationToken,
) -> Result<Profile> {
    let (total, available) = hardware::system_memory();
    let mut result = Profile {
        backend: backend.into(),
        unified: backend == "metal" && cfg!(target_arch = "aarch64"),
        system_total: total / MIB,
        system_available: available / MIB,
        model_limit: limit(model),
        recommended: None,
        maximum: None,
        estimated_max_before_startup: None,
        minimum: None,
        custom: None,
    };
    ensure!(
        result.model_limit >= MIN_CONTEXT,
        "This model does not support a context of at least {MIN_CONTEXT} tokens."
    );
    let mut samples = BTreeMap::new();
    for suggested in [false, true] {
        let cap = if suggested {
            result.model_limit.min(RECOMMENDED_LIMIT)
        } else {
            result.model_limit
        };
        let floor = sample(
            &mut samples,
            profiler,
            model,
            backend,
            gpu_layers,
            MIN_CONTEXT,
            cancel,
        )
        .await?;
        if !result.fits(floor, suggested) {
            continue;
        }
        let high = sample(
            &mut samples,
            profiler,
            model,
            backend,
            gpu_layers,
            cap,
            cancel,
        )
        .await?;
        let best = if result.fits(high, suggested) {
            high.clone()
        } else {
            let mut low = MIN_CONTEXT / STEP;
            let mut upper = cap.div_ceil(STEP);
            while low + 1 < upper {
                let middle = (low + upper) / 2;
                let candidate = sample(
                    &mut samples,
                    profiler,
                    model,
                    backend,
                    gpu_layers,
                    middle * STEP,
                    cancel,
                )
                .await?;
                if result.fits(candidate, suggested) {
                    low = middle;
                } else {
                    upper = middle;
                }
            }
            samples[&(low * STEP)].clone()
        };
        if suggested {
            result.recommended = Some(best);
        } else {
            result.maximum = Some(best);
        }
    }
    result.minimum = samples.get(&MIN_CONTEXT).cloned();
    if result.recommended.is_none() {
        if let Some(maximum) = &result.maximum {
            let tokens = maximum.tokens.min(STANDARD_CONTEXT);
            let fallback = sample(
                &mut samples,
                profiler,
                model,
                backend,
                gpu_layers,
                tokens,
                cancel,
            )
            .await?;
            result.recommended = if result.fits(fallback, false) {
                Some(fallback.clone())
            } else {
                result.minimum.clone()
            };
        }
    }
    if let Some(tokens) =
        custom.filter(|tokens| (MIN_CONTEXT..=result.model_limit).contains(tokens))
    {
        result.custom = Some(
            sample(
                &mut samples,
                profiler,
                model,
                backend,
                gpu_layers,
                tokens,
                cancel,
            )
            .await?
            .clone(),
        );
    }
    Ok(result)
}

async fn sample<'a>(
    samples: &'a mut BTreeMap<u32, Measurement>,
    profiler: &Path,
    model: &Installed,
    backend: &str,
    gpu_layers: Option<u32>,
    tokens: u32,
    cancel: &CancellationToken,
) -> Result<&'a Measurement> {
    match samples.entry(tokens) {
        std::collections::btree_map::Entry::Occupied(entry) => Ok(entry.into_mut()),
        std::collections::btree_map::Entry::Vacant(entry) => {
            let mut command = hardware::runtime_command(profiler);
            command
                .current_dir(profiler.parent().context("Missing profiler folder")?)
                .args([
                    "--fit-print",
                    "on",
                    "--log-jsonl",
                    "--parallel",
                    "1",
                    "--ctx-size",
                    &tokens.to_string(),
                ])
                .args(placement(backend, gpu_layers))
                .arg("--model")
                .arg(&model.path)
                .stdin(Stdio::null())
                .stderr(Stdio::piped());
            let query = Command::from(command).kill_on_drop(true).output();
            let output = tokio::select! {
                _ = cancel.cancelled() => bail!("Memory check stopped."),
                result = tokio::time::timeout(Duration::from_secs(30), query) => result.context("llama.cpp's memory check timed out.")??,
            };
            ensure!(
                output.status.success(),
                "llama.cpp could not measure memory for this model and backend."
            );
            let rows = parse(&output.stdout)?;
            ensure!(
                backend == "cpu" || rows.iter().any(|row| row.kind == "device" && row.used > 0),
                "llama.cpp did not report memory for an active accelerator."
            );
            Ok(entry.insert(Measurement { tokens, rows }))
        }
    }
}

pub fn placement(backend: &str, gpu_layers: Option<u32>) -> Vec<String> {
    let layers = if backend == "cpu" {
        "0".into()
    } else {
        gpu_layers
            .map(|n| n.to_string())
            .unwrap_or_else(|| "all".into())
    };
    let mut args = vec!["--fit".into(), "off".into(), "--gpu-layers".into(), layers];
    if backend == "cpu" {
        args.extend(["--device".into(), "none".into()]);
    }
    args
}

fn parse(output: &[u8]) -> Result<Vec<MemoryRow>> {
    for line in output.rsplit(|&byte| byte == b'\n') {
        let Ok(event) = serde_json::from_slice::<serde_json::Value>(line) else {
            continue;
        };
        if event["type"] != "fit_memory_breakdown" || event["data"]["unit"] != "MiB" {
            continue;
        }
        let rows: Vec<MemoryRow> = serde_json::from_value(event["data"]["rows"].clone())?;
        ensure!(
            rows.iter().any(|row| row.kind == "host") && rows.iter().any(|row| row.used > 0),
            "llama.cpp did not report usable memory estimates."
        );
        return Ok(rows);
    }
    bail!("llama.cpp did not return a memory report.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startup_verified_max_updates_both_presets_on_the_same_backend() {
        let sample = |tokens| Measurement {
            tokens,
            rows: Vec::new(),
        };
        let mut profile = Profile {
            backend: "metal".into(),
            unified: true,
            system_total: 8192,
            system_available: 4700,
            model_limit: 32768,
            recommended: Some(sample(23040)),
            maximum: Some(sample(24576)),
            estimated_max_before_startup: None,
            minimum: Some(sample(MIN_CONTEXT)),
            custom: None,
        };
        assert_eq!(profile.max_retry_sizes(), [23040, 21504, 18432, 512]);
        profile.record_startup_limit(sample(21504));
        assert_eq!(profile.backend, "metal");
        assert_eq!(profile.estimated_max_before_startup, Some(24576));
        let recommended = ModelPreferences::default();
        let max = ModelPreferences {
            context_mode: ContextMode::Most,
            ..recommended.clone()
        };
        assert_eq!(profile.context(&recommended).unwrap(), 21504);
        assert_eq!(profile.context(&max).unwrap(), 21504);
        let custom = ModelPreferences {
            context_mode: ContextMode::Custom,
            custom_tokens: 24576,
            ..recommended
        };
        assert!(profile.validate_custom(&custom).is_err());
    }

    #[test]
    fn pinned_report_counts_device_and_unassigned_buffer_memory() {
        let output = br#"ggml Metal ready
{"type":"fit_memory_breakdown","data":{"unit":"MiB","rows":[{"kind":"device","name":"MTL0","description":"Apple M1","total":5461,"free":5460,"self":3758,"model":2146,"context":1074,"compute":537},{"kind":"host","name":"Host","self":601,"model":497,"context":0,"compute":104},{"kind":"buffer_type","name":"Mapped","self":497,"model":497,"context":0,"compute":0}]}}
Host 497 0 104"#;
        let sample = Measurement {
            tokens: 32768,
            rows: parse(output).unwrap(),
        };
        let mut profile = Profile {
            backend: "metal".into(),
            unified: true,
            system_total: 8192,
            system_available: 5400,
            model_limit: 262144,
            recommended: None,
            maximum: Some(sample.clone()),
            estimated_max_before_startup: None,
            minimum: None,
            custom: None,
        };
        assert!(profile.fits(&sample, false));
        assert!(!profile.fits(&sample, true));
        profile.system_available = 4400;
        assert!(!profile.fits(&sample, false));
        profile.unified = false;
        assert!(profile.fits(&sample, false));
        let mut prefs = ModelPreferences {
            context_mode: ContextMode::Custom,
            custom_tokens: 40_000,
            ..Default::default()
        };
        assert!(profile.validate_custom(&prefs).is_err());
        prefs.allow_over_budget = true;
        assert_eq!(profile.context(&prefs).unwrap(), 40_000);
    }
}

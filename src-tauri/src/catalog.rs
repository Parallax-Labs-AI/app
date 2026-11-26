use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

#[derive(Debug, Serialize, Deserialize)]
pub struct CatalogModel {
    pub id: String,
    pub name: String,
    pub description: String,
    pub repo: String,
    pub revision: Option<String>,
    pub architecture: Option<String>,
    pub context: Option<u32>,
    pub license: Option<String>,
    pub status: Availability,
    #[serde(default)]
    pub reason: String,
    #[serde(default)]
    pub variants: Vec<Variant>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Availability {
    Available,
    ComingSoon,
    Incompatible,
    Unavailable,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Variant {
    pub file: String,
    pub quantization: String,
    pub bytes: u64,
    pub sha256: String,
}

pub fn catalog() -> &'static [CatalogModel] {
    static MODELS: OnceLock<Vec<CatalogModel>> = OnceLock::new();
    MODELS.get_or_init(|| {
        serde_json::from_str(include_str!("../data/models.json")).expect("valid bundled catalog")
    })
}

pub fn sha_is_valid(hash: &str) -> bool {
    hash.len() == 64
        && hash
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub fn variant_id(model: &CatalogModel, variant: &Variant) -> String {
    format!(
        "{}-{}-{}",
        model.id,
        variant.quantization.to_ascii_lowercase(),
        variant.sha256.get(..12).unwrap_or(&variant.sha256)
    )
}

pub fn validate(model: &CatalogModel, variant: &Variant) -> Result<()> {
    ensure!(
        model.status == Availability::Available
            && model
                .architecture
                .as_deref()
                .is_some_and(crate::models::supports_architecture),
        "No compatible GGUF is published for this model."
    );
    let revision = model
        .revision
        .as_deref()
        .context("Missing model revision")?;
    ensure!(
        revision.len() == 40 && revision.bytes().all(|b| b.is_ascii_hexdigit()),
        "Invalid model revision"
    );
    ensure!(
        model.repo.strip_prefix("parallaxlabs/") == Some(model.id.as_str())
            && model
                .id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-'),
        "Invalid official repository"
    );
    ensure!(
        variant.file.ends_with(".gguf")
            && variant
                .file
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b)),
        "Invalid GGUF filename"
    );
    ensure!(
        variant.bytes > 0 && sha_is_valid(&variant.sha256),
        "Missing verified GGUF size or checksum"
    );
    Ok(())
}

pub fn resolve(id: &str) -> Result<(&'static CatalogModel, &'static Variant)> {
    let (model, variant) = catalog()
        .iter()
        .find_map(|model| {
            model
                .variants
                .iter()
                .find(|variant| variant_id(model, variant) == id)
                .map(|variant| (model, variant))
        })
        .context("This model file is not in the official catalog.")?;
    validate(model, variant)?;
    Ok((model, variant))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn catalog_entries_are_hashed_pinpointed_ggufs() {
        for model in catalog() {
            for variant in &model.variants {
                validate(model, variant).unwrap();
            }
        }
        assert!(resolve("unreleased").is_err());
        let mut model: CatalogModel = serde_json::from_str(r#"{"id":"future","name":"Future","description":"","repo":"parallaxlabs/future","status":"coming_soon"}"#).unwrap();
        model.architecture = Some("qwen35".into());
        assert!(validate(&model, &catalog()[0].variants[0]).is_err());
    }
}

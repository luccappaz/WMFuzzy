use thiserror::Error;

/// Errors produced by the `WMModel` pipeline.
#[derive(Error, Debug)]
pub enum WMModelError {
    #[error("No membership values computed, generate rules first.")]
    NoMFValues,
    #[error("No rules loaded! Generate rules or load them from a file.")]
    NoRulesLoaded,
    #[error(
        "Granularity missing or unconfigured for feature: {0}. Need to call set_granularity first"
    )]
    MissingGranularity(String),
    #[error("Limits count ({0}) for feature '{1}' does not match granularity count ({2}).")]
    LimitsMismatch(usize, String, usize),
    #[error("Column '{0}' not found in DataFrame.")]
    ColumnNotFound(String),
    #[error("Expected Float64 column for '{0}'.")]
    ExpectedFloat64(String),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Polars error: {0}")]
    Polars(#[from] polars::error::PolarsError),
    #[error("Dataset error: {0}")]
    DatasetError(String),
    #[error("Rule not found or insufficient activation")]
    InferenceError,
    #[error("Invalid strategy")]
    InvalidStrategy(String),
    #[error("Memberships not configured yet. Need to build")]
    UnconfiguredError,
}

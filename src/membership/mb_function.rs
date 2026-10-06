use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::str::FromStr;

use crate::membership::triangular::TriangularMF;
use crate::{error::WMModelError, fuzzy::granularity::Granularity};

/// Internal representation of supported fuzzy partition strategies.
///
/// Kept `pub(crate)` to enforce domain invariant validations through type constructors
/// (`FuzzyStrategy::linear`, `FuzzyStrategy::custom`) and Serde conversion hooks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub(crate) enum InnerStrategy {
    /// Evenly spaced linear grid over `[min, max]`.
    Linear { min: f64, max: f64 },
    /// Custom grid defined by explicit knot coordinates.
    Custom { knots: Vec<f64> },
}

/// Opaque wrapper guaranteeing mathematical domain invariants for fuzzy partitions.
///
/// Ensures domain invariants upon construction and deserialization:
/// - **Linear**: Requires $min < max$.
/// - **Custom**: Requires at least 2 knots sorted in strictly ascending order without duplicates.
///
/// Applies identically to input features ($X$) and target variables ($Y$) in Case 1 model architectures.
///
/// # Examples
///
/// ```rust
/// use wm_fuzzy::prelude::FuzzyStrategy;
///
/// // Create a linear strategy spanning [0.0, 100.0]
/// let linear_strat = FuzzyStrategy::linear(0.0, 100.0).unwrap();
///
/// // Create a custom strategy with explicit grid knots
/// let custom_strat = FuzzyStrategy::custom(vec![0.0, 25.0, 50.0, 100.0]).unwrap();
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "InnerStrategy", into = "InnerStrategy")]
pub struct FuzzyStrategy {
    inner: InnerStrategy,
}

impl TryFrom<InnerStrategy> for FuzzyStrategy {
    type Error = WMModelError;

    /// Validates and converts an unvalidated [`InnerStrategy`] into a [`FuzzyStrategy`].
    ///
    /// # Errors
    /// Returns [`WMModelError::InvalidStrategy`] if bounds are invalid ($min \ge max$)
    /// or if custom knots are unsorted/insufficient.
    fn try_from(inner: InnerStrategy) -> Result<Self, Self::Error> {
        match &inner {
            InnerStrategy::Linear { min, max } if min >= max => Err(WMModelError::InvalidStrategy(
                "The 'min' bound must be strictly less than 'max'.".into(),
            )),
            InnerStrategy::Custom { knots } => {
                Self::validate_knots(knots)?;
                Ok(Self { inner })
            }
            _ => Ok(Self { inner }),
        }
    }
}

impl From<FuzzyStrategy> for InnerStrategy {
    fn from(strategy: FuzzyStrategy) -> Self {
        strategy.inner
    }
}

impl FuzzyStrategy {
    /// Creates a validated linear partition strategy spanning `[min, max]`.
    ///
    /// Grid knots are evaluated dynamically based on the requested [`Granularity`].
    ///
    /// # Errors
    /// Returns [`WMModelError::InvalidStrategy`] if `min >= max`.
    ///
    /// # Examples
    /// ```rust
    /// use wm_fuzzy::prelude::FuzzyStrategy;
    ///
    /// let strategy = FuzzyStrategy::linear(0.0, 10.0).unwrap();
    /// ```
    pub fn linear(min: f64, max: f64) -> Result<Self, WMModelError> {
        let inner = InnerStrategy::Linear { min, max };
        inner.try_into()
    }

    /// Creates a validated custom partition strategy from explicit grid knot coordinates.
    ///
    /// # Errors
    /// Returns [`WMModelError::InvalidStrategy`] if:
    /// - Fewer than 2 knots are provided.
    /// - Knots are not strictly sorted in ascending order ($x_0 < x_1 < \dots < x_n$).
    ///
    /// # Examples
    /// ```rust
    /// use wm_fuzzy::prelude::FuzzyStrategy;
    ///
    /// let strategy = FuzzyStrategy::custom(vec![0.0, 2.5, 5.0, 10.0]).unwrap();
    /// ```
    pub fn custom<T: Into<Vec<f64>>>(knots: T) -> Result<Self, WMModelError> {
        let inner = InnerStrategy::Custom {
            knots: knots.into(),
        };
        inner.try_into()
    }

    /// Expands the strategy into concrete grid knot coordinates using the provided [`Granularity`].
    ///
    /// For **Linear** strategies, computes $N$ uniformly spaced points from $min$ to $max$.
    /// For **Custom** strategies, returns a cloned slice of the pre-validated knots.
    ///
    /// # Arguments
    /// * `granularity` - Number of partition sets ($N \ge 2$).
    pub fn resolve_knots(&self, granularity: Granularity) -> Vec<f64> {
        match &self.inner {
            InnerStrategy::Linear { min, max } => {
                let n = granularity.count();
                let step = (max - min) / (n - 1) as f64;
                (0..n).map(|i| min + i as f64 * step).collect()
            }
            InnerStrategy::Custom { knots } => knots.clone(),
        }
    }

    /// Deserializes a JSON string mapping feature/target names to [`FuzzyStrategy`] instances.
    ///
    /// # Errors
    /// Returns [`WMModelError::InvalidStrategy`] if the JSON string is malformed or violates domain invariants.
    pub fn map_from_json(json_str: &str) -> Result<HashMap<String, FuzzyStrategy>, WMModelError> {
        serde_json::from_str(json_str).map_err(|e| {
            WMModelError::InvalidStrategy(format!("Failed to parse JSON strategy map: {}", e))
        })
    }

    /// Reads and deserializes a JSON file into a strategy map.
    ///
    /// # Errors
    /// Returns [`WMModelError::InvalidStrategy`] if file I/O fails or if contents violate strategy invariants.
    pub fn map_from_json_file<P: AsRef<Path>>(
        path: P,
    ) -> Result<HashMap<String, FuzzyStrategy>, WMModelError> {
        let file = File::open(path).map_err(|e| {
            WMModelError::InvalidStrategy(format!("Failed to open strategy JSON file: {}", e))
        })?;
        let reader = BufReader::new(file);

        serde_json::from_reader(reader).map_err(|e| {
            WMModelError::InvalidStrategy(format!("Failed to parse strategy JSON file: {}", e))
        })
    }

    /// Serializes and writes a map of [`FuzzyStrategy`] configurations to a JSON file.
    ///
    /// # Errors
    /// Returns [`WMModelError::InvalidStrategy`] if creating or writing to the destination file fails.
    pub fn map_to_json_file<P: AsRef<Path>>(
        map: &HashMap<String, FuzzyStrategy>,
        path: P,
    ) -> Result<(), WMModelError> {
        let file = File::create(path).map_err(|e| {
            WMModelError::InvalidStrategy(format!("Failed to create strategy JSON file: {}", e))
        })?;
        let writer = BufWriter::new(file);

        serde_json::to_writer_pretty(writer, map).map_err(|e| {
            WMModelError::InvalidStrategy(format!("Failed to write strategy JSON file: {}", e))
        })
    }

    /// Validates custom knot sequence ordering and minimum length.
    fn validate_knots(knots: &[f64]) -> Result<(), WMModelError> {
        if knots.len() < 2 {
            return Err(WMModelError::InvalidStrategy(format!(
                "Custom grid knots require at least 2 points, got {}.",
                knots.len()
            )));
        }
        if !knots.windows(2).all(|w| w[0] < w[1]) {
            return Err(WMModelError::InvalidStrategy(
                "Grid knots must be strictly sorted in ascending order.".into(),
            ));
        }
        Ok(())
    }

    /// Exposes internal strategy variant for crate-internal model compilation.
    pub(crate) fn inner(&self) -> &InnerStrategy {
        &self.inner
    }
}

/// Enumeration of supported membership function geometries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MFKind {
    /// Triangular membership function geometry ($a, b, c$).
    Triangular,
    // Gaussian,
    // Trapezoidal,
}

impl MFKind {
    /// Infers the partition [`Granularity`] from explicit grid knot count based on the geometry type.
    ///
    /// For **Triangular** MFs, each grid knot corresponds exactly to 1 modal peak/center.
    ///
    /// # Errors
    /// Returns [`WMModelError`] if `knot_count < 2`.
    pub fn infer_granularity(&self, knot_count: usize) -> Result<Granularity, WMModelError> {
        match self {
            MFKind::Triangular => Granularity::try_from(knot_count),
        }
    }
}

impl FromStr for MFKind {
    type Err = WMModelError;

    /// Parses a string slice into an [`MFKind`] variant (case-insensitive).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "triangular" => Ok(MFKind::Triangular),
            _ => Err(WMModelError::InvalidStrategy(format!(
                "Invalid membership function kind: '{}'",
                s
            ))),
        }
    }
}

impl TryFrom<&str> for MFKind {
    type Error = WMModelError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        s.parse()
    }
}

/// Polymorphic container wrapping all concrete fuzzy membership function types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MembershipFunction {
    /// Triangular membership function variant.
    Triangular(TriangularMF),
}

/// Fundamental operational interface for evaluating fuzzy membership functions.
pub trait MembershipOp: Send + Sync {
    /// Evaluates the membership degree $\mu(x) \in [0.0, 1.0]$ for a crisp input $x$.
    fn eval(&self, x: f64) -> f64;

    /// Returns the modal center/peak coordinate $c$ where $\mu(c) = 1.0$.
    fn center(&self) -> f64;

    /// Returns the support interval $(x_{\min}, x_{\max})$ where $\mu(x) > 0.0$.
    fn support(&self) -> (f64, f64);
}

/// Interface for generating partitioned fuzzy membership function collections across a universe of discourse.
pub trait Partitionable: Sized {
    /// Generates a complete sequence of membership functions covering the partition domain.
    ///
    /// # Arguments
    /// * `kind` - Geometry type ([`MFKind`]).
    /// * `strategy` - Partitioning bounds or custom grid knots ([`FuzzyStrategy`]).
    /// * `granularity` - Number of fuzzy sets to instantiate ([`Granularity`]).
    fn create_partition(
        kind: MFKind,
        strategy: FuzzyStrategy,
        granularity: Granularity,
    ) -> Vec<Self>;
}

impl MembershipOp for MembershipFunction {
    fn eval(&self, x: f64) -> f64 {
        match self {
            Self::Triangular(mf) => mf.eval(x),
        }
    }

    fn center(&self) -> f64 {
        match self {
            Self::Triangular(mf) => mf.center(),
        }
    }

    fn support(&self) -> (f64, f64) {
        match self {
            Self::Triangular(mf) => mf.support(),
        }
    }
}

impl Partitionable for MembershipFunction {
    fn create_partition(
        kind: MFKind,
        strategy: FuzzyStrategy,
        granularity: Granularity,
    ) -> Vec<Self> {
        match kind {
            MFKind::Triangular => TriangularMF::create_partition(kind, strategy, granularity)
                .into_iter()
                .map(MembershipFunction::Triangular)
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_strategy_json_str_and_file() {
        let json_data = r#"{
            "x1": { "type": "linear", "min": 0.0, "max": 10.0 },
            "x2": { "type": "custom", "knots": [0.0, 2.5, 5.0, 10.0] }
        }"#;

        // Test loading from string
        let map = FuzzyStrategy::map_from_json(json_data).unwrap();
        assert!(
            matches!(map.get("x1").unwrap().inner(), InnerStrategy::Linear { min, max } if *min == 0.0 && *max == 10.0)
        );

        // Test file roundtrip
        let temp_file = NamedTempFile::new().unwrap();
        FuzzyStrategy::map_to_json_file(&map, temp_file.path()).unwrap();
        let loaded_file_map = FuzzyStrategy::map_from_json_file(temp_file.path()).unwrap();

        assert_eq!(map, loaded_file_map);
    }
}

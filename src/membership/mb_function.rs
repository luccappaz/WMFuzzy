use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::str::FromStr;

use crate::membership::triangular::TriangularMF;
use crate::{error::WMModelError, fuzzy::granularity::Granularity};

/// Internal representation of supported partition strategies.
/// Kept private to enforce validation through constructors and Serde deserialization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub(crate) enum InnerStrategy {
    Linear { min: f64, max: f64 },
    Custom { knots: Vec<f64> },
}

/// Public opaque wrapper for partition strategies.
/// Guarantees domain invariants (e.g., valid bounds, ordered points) upon construction and deserialization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "InnerStrategy", into = "InnerStrategy")]
pub struct PartitionStrategy {
    inner: InnerStrategy,
}

impl TryFrom<InnerStrategy> for PartitionStrategy {
    type Error = WMModelError;

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

impl From<PartitionStrategy> for InnerStrategy {
    fn from(strategy: PartitionStrategy) -> Self {
        strategy.inner
    }
}

impl PartitionStrategy {
    /// Creates a linear partition strategy over [min, max].
    pub fn linear(min: f64, max: f64) -> Result<Self, WMModelError> {
        let inner = InnerStrategy::Linear { min, max };
        inner.try_into()
    }

    /// Creates a custom partition strategy from explicit grid knots.
    pub fn custom<T: Into<Vec<f64>>>(knots: T) -> Result<Self, WMModelError> {
        let inner = InnerStrategy::Custom {
            knots: knots.into(),
        };
        inner.try_into()
    }

    /// Expands the strategy into grid knots using the provided feature granularity.
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

    /// Deserializes a JSON string containing a feature strategy map directly into a `HashMap<String, PartitionStrategy>`.
    pub fn map_from_json(
        json_str: &str,
    ) -> Result<HashMap<String, PartitionStrategy>, WMModelError> {
        serde_json::from_str(json_str).map_err(|e| {
            WMModelError::InvalidStrategy(format!("Failed to parse JSON strategy map: {}", e))
        })
    }

    /// Reads and deserializes a JSON file directly into a `HashMap<String, PartitionStrategy>`.
    pub fn map_from_json_file<P: AsRef<Path>>(
        path: P,
    ) -> Result<HashMap<String, PartitionStrategy>, WMModelError> {
        let file = File::open(path).map_err(|e| {
            WMModelError::InvalidStrategy(format!("Failed to open strategy JSON file: {}", e))
        })?;
        let reader = BufReader::new(file);

        serde_json::from_reader(reader).map_err(|e| {
            WMModelError::InvalidStrategy(format!("Failed to parse strategy JSON file: {}", e))
        })
    }

    /// Serializes and saves a map of partition strategies to a JSON file.
    pub fn map_to_json_file<P: AsRef<Path>>(
        map: &HashMap<String, PartitionStrategy>,
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

    pub(crate) fn inner(&self) -> &InnerStrategy {
        &self.inner
    }
}

/// Enum specifying the supported fuzzy membership function kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MFKind {
    Triangular,
    // Gaussian,
    // Trapezoidal,
}

impl MFKind {
    /// Infers the partition granularity from explicit grid knot count based on the MF geometry.
    pub fn infer_granularity(&self, knot_count: usize) -> Result<Granularity, WMModelError> {
        match self {
            // For Triangular MFs, each grid knot corresponds to 1 MF center/peak.
            MFKind::Triangular => Granularity::try_from(knot_count),
            // Future MF kinds can define their own knot-to-granularity mapping rules:
            // MFKind::Gaussian => Granularity::try_from(knot_count),
            // MFKind::Trapezoidal => Granularity::try_from((knot_count + 1) / 2),
        }
    }
}

impl FromStr for MFKind {
    type Err = WMModelError;

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

/// Enum wrapping the supported fuzzy membership function types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MembershipFunction {
    Triangular(TriangularMF),
}

/// Core interface for evaluating and querying fuzzy membership functions.
pub trait MembershipOp: Send + Sync {
    /// Evaluates the membership degree for a given value `x` (returns a value in [0.0, 1.0]).
    fn eval(&self, x: f64) -> f64;

    /// Returns the peak center where the membership degree equals 1.0.
    fn center(&self) -> f64;

    /// Returns the support interval `(min, max)` where the membership degree is strictly greater than 0.0.
    fn support(&self) -> (f64, f64);
}

/// Interface for generating fuzzy partitions across the universe of discourse.
pub trait Partitionable: Sized {
    /// Generates a collection of membership functions according to the given strategy.
    fn create_partition(
        kind: MFKind,
        strategy: PartitionStrategy,
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
        strategy: PartitionStrategy,
        granularity: Granularity,
    ) -> Vec<Self> {
        match kind {
            // Delegates partition generation to TriangularMF and wraps the result in the enum.
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
        let map = PartitionStrategy::map_from_json(json_data).unwrap();
        assert!(
            matches!(map.get("x1").unwrap().inner(), InnerStrategy::Linear { min, max } if *min == 0.0 && *max == 10.0)
        );

        // Test file roundtrip
        let temp_file = NamedTempFile::new().unwrap();
        PartitionStrategy::map_to_json_file(&map, temp_file.path()).unwrap();
        let loaded_file_map = PartitionStrategy::map_from_json_file(temp_file.path()).unwrap();

        assert_eq!(map, loaded_file_map);
    }
}

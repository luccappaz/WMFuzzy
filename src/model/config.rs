use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use crate::error::WMModelError;
use crate::fuzzy::granularity::Granularity;
use crate::membership::mb_function::{MFKind, PartitionStrategy};

/// Configuration parameters for an individual feature/variable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Optional partition granularity (can be inferred from custom knots if omitted).
    pub granularity: Option<Granularity>,
    /// Partition strategy (linear domain bounds or explicit custom grid knots).
    pub strategy: PartitionStrategy,
    /// Optional custom linguistic labels for the membership functions.
    #[serde(default)]
    pub labels: Vec<String>,
}

/// Consolidated model configuration mapped by feature name to prevent JSON key redundancy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub kind: MFKind,
    pub features: HashMap<String, FeatureConfig>,
}

impl ModelConfig {
    /// Deserializes a `ModelConfig` directly from a JSON string.
    pub fn from_json(json_str: &str) -> Result<Self, WMModelError> {
        serde_json::from_str(json_str).map_err(|e| {
            WMModelError::InvalidStrategy(format!("Failed to parse configuration JSON: {}", e))
        })
    }

    /// Reads and deserializes a `ModelConfig` from a JSON file path.
    pub fn from_json_file<P: AsRef<Path>>(path: P) -> Result<Self, WMModelError> {
        let file = File::open(path).map_err(|e| {
            WMModelError::InvalidStrategy(format!("Failed to open configuration file: {}", e))
        })?;
        let reader = BufReader::new(file);

        serde_json::from_reader(reader).map_err(|e| {
            WMModelError::InvalidStrategy(format!("Failed to parse configuration file: {}", e))
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::model::wm::WMModel;

    #[test]
    fn test_load_model_from_compact_json() {
        let json_spec = r#"{
            "kind": "triangular",
            "features": {
                "x1": {
                    "granularity": 3,
                    "strategy": {
                        "type": "linear",
                        "min": 0.0,
                        "max": 10.0
                    },
                    "labels": ["low", "medium", "high"]
                },
                "x2": {
                    "strategy": {
                        "type": "custom",
                        "knots": [0.0, 5.0, 10.0]
                    }
                }
            }
        }"#;

        // Build and validate model directly from the JSON string
        let model = WMModel::from_json(json_spec).unwrap();

        // Verify x1 (Linear strategy with custom labels)
        let x1_mfs = model.mf_configs().unwrap().get("x1").unwrap();
        assert!(x1_mfs.contains_key("low"));
        assert!(x1_mfs.contains_key("medium"));
        assert!(x1_mfs.contains_key("high"));

        // Verify x2 (Custom strategy with inferred granularity = 3)
        let x2_mfs = model.mf_configs().unwrap().get("x2").unwrap();
        assert_eq!(x2_mfs.len(), 3);
    }
}

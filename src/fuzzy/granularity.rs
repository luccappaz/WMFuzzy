use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs::File, io::BufReader, path::Path, str::FromStr};

use crate::error::WMModelError;

/// Partition granularity level for fuzzy set attributes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "usize", into = "usize")]
pub enum Granularity {
    Three = 3,
    Five = 5,
}

impl Granularity {
    /// Returns the number of partitions for this granularity level.
    #[inline]
    pub fn count(&self) -> usize {
        *self as usize
    }

    /// Generates the default linguistic labels according to the granularity level.
    pub fn default_labels(&self) -> Vec<String> {
        match self {
            Granularity::Three => vec!["low", "med", "high"],
            Granularity::Five => vec!["very_low", "low", "med", "high", "very_high"],
        }
        .into_iter()
        .map(String::from)
        .collect()
    }

    /// Deserializes a JSON string containing a feature map (e.g., `{"x_1": 3, "x_2": 5}`)
    /// directly into a `HashMap<String, Granularity>`.
    pub fn map_from_json(
        json_str: &str,
    ) -> Result<std::collections::HashMap<String, Granularity>, WMModelError> {
        serde_json::from_str(json_str).map_err(|e| {
            WMModelError::InvalidStrategy(format!("Failed to parse JSON granularity map: {}", e))
        })
    }

    /// Reads and deserializes a JSON file directly into a `HashMap<String, Granularity>`.
    pub fn map_from_json_file<P: AsRef<Path>>(
        path: P,
    ) -> Result<HashMap<String, Granularity>, WMModelError> {
        let file = File::open(path).map_err(|e| {
            WMModelError::InvalidStrategy(format!("Failed to open JSON file: {}", e))
        })?;

        let reader = BufReader::new(file);

        let map = serde_json::from_reader(reader).map_err(|e| {
            WMModelError::InvalidStrategy(format!("Failed to parse JSON content: {}", e))
        })?;

        Ok(map)
    }
}

impl From<Granularity> for usize {
    fn from(g: Granularity) -> Self {
        match g {
            Granularity::Three => 3,
            Granularity::Five => 5,
        }
    }
}

impl TryFrom<usize> for Granularity {
    type Error = WMModelError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            3 => Ok(Granularity::Three),
            5 => Ok(Granularity::Five),
            n => Err(WMModelError::InvalidStrategy(format!(
                "Unsupported granularity level: {}. Expected 3, 5, etc.",
                n
            ))),
        }
    }
}

impl TryFrom<&str> for Granularity {
    type Error = WMModelError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let trimmed = value.trim();
        match trimmed.to_lowercase().as_str() {
            "3" | "three" => Ok(Granularity::Three),
            "5" | "five" => Ok(Granularity::Five),
            _ => {
                // Fallback to numeric string parsing
                if let Ok(n) = trimmed.parse::<usize>() {
                    Granularity::try_from(n)
                } else {
                    Err(WMModelError::InvalidStrategy(format!(
                        "Invalid granularity string: '{}'. Expected '3', '5', 'three', or 'five'.",
                        value
                    )))
                }
            }
        }
    }
}

impl FromStr for Granularity {
    type Err = WMModelError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Granularity::try_from(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_granularity_from_json_string() {
        // Direct parsing of a JSON map with numeric values
        let json_data = r#"{"x_1": 3, "x_2": 5}"#;
        let map: HashMap<String, Granularity> = Granularity::map_from_json(json_data).unwrap();

        assert_eq!(map.get("x_1"), Some(&Granularity::Three));
        assert_eq!(map.get("x_2"), Some(&Granularity::Five));

        // Individual string conversions via TryFrom and FromStr
        let g1 = Granularity::try_from("3").unwrap();
        let g2: Granularity = "five".parse().unwrap();

        assert_eq!(g1, Granularity::Three);
        assert_eq!(g2, Granularity::Five);
    }
}

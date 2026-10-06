use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs::File, io::BufReader, path::Path, str::FromStr};

use crate::error::WMModelError;

/// Partition granularity level for fuzzy set attributes ($N \ge 2$).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "usize", into = "usize")]
pub struct Granularity(usize);

impl Granularity {
    pub const TWO: Self = Self(2);
    pub const THREE: Self = Self(3);
    pub const FOUR: Self = Self(4);
    pub const FIVE: Self = Self(5);
    pub const SIX: Self = Self(6);
    pub const SEVEN: Self = Self(7);
    pub const EIGHT: Self = Self(8);

    /// Constructs a new `Granularity` instance ensuring $N \ge 2$.
    pub fn new(count: usize) -> Result<Self, WMModelError> {
        Self::try_from(count)
    }

    /// Returns the number of partitions for this granularity level.
    #[inline]
    pub fn count(&self) -> usize {
        self.0
    }

    /// Generates default linguistic labels for standard levels (2 to 8).
    ///
    /// Returns an error if $N > 8$, indicating that custom labels must be provided
    /// explicitly via `model.add_label`.
    pub fn default_labels(&self) -> Result<Vec<String>, WMModelError> {
        let labels = match self.0 {
            2 => vec!["low", "high"],
            3 => vec!["low", "med", "high"],
            4 => vec!["very_low", "low", "high", "very_high"],
            5 => vec!["very_low", "low", "med", "high", "very_high"],
            6 => vec![
                "very_low",
                "low",
                "med_low",
                "med_high",
                "high",
                "very_high",
            ],
            7 => vec![
                "extremely_low",
                "very_low",
                "low",
                "med",
                "high",
                "very_high",
                "extremely_high",
            ],
            8 => vec![
                "extremely_low",
                "very_low",
                "low",
                "med_low",
                "med_high",
                "high",
                "very_high",
                "extremely_high",
            ],
            n => {
                return Err(WMModelError::InvalidStrategy(format!(
                    "Granularity level {} has no default linguistic terms. You must explicitly provide labels via `model.add_label`.",
                    n
                )));
            }
        };

        Ok(labels.into_iter().map(String::from).collect())
    }

    /// Deserializes a JSON string containing a feature map (e.g., `{"x_1": 3, "x_2": 10}`)
    /// directly into a `HashMap<String, Granularity>`.
    pub fn map_from_json(json_str: &str) -> Result<HashMap<String, Granularity>, WMModelError> {
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
        g.0
    }
}

impl TryFrom<usize> for Granularity {
    type Error = WMModelError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        if value < 2 {
            return Err(WMModelError::InvalidStrategy(format!(
                "Invalid granularity level: {}. Granularity must be at least 2.",
                value
            )));
        }
        Ok(Granularity(value))
    }
}

impl TryFrom<&str> for Granularity {
    type Error = WMModelError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let trimmed = value.trim();
        match trimmed.to_lowercase().as_str() {
            "2" | "two" => Ok(Granularity::TWO),
            "3" | "three" => Ok(Granularity::THREE),
            "4" | "four" => Ok(Granularity::FOUR),
            "5" | "five" => Ok(Granularity::FIVE),
            "6" | "six" => Ok(Granularity::SIX),
            "7" | "seven" => Ok(Granularity::SEVEN),
            "8" | "eight" => Ok(Granularity::EIGHT),
            _ => {
                if let Ok(n) = trimmed.parse::<usize>() {
                    Granularity::try_from(n)
                } else {
                    Err(WMModelError::InvalidStrategy(format!(
                        "Invalid granularity string: '{}'. Expected a number >= 2 or words ('two' through 'eight').",
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
    fn test_granularity_bounds_and_labels() {
        assert!(Granularity::try_from(1).is_err());

        let g2 = Granularity::try_from(2).unwrap();
        assert_eq!(g2.default_labels().unwrap(), vec!["low", "high"]);

        let g8 = Granularity::try_from(8).unwrap();
        assert_eq!(g8.default_labels().unwrap().len(), 8);

        let g10 = Granularity::try_from(10).unwrap();
        assert!(g10.default_labels().is_err());
    }

    #[test]
    fn test_granularity_from_json_string() {
        let json_data = r#"{"x_1": 3, "x_2": 5, "x_3": 12}"#;
        let map: HashMap<String, Granularity> = Granularity::map_from_json(json_data).unwrap();

        assert_eq!(map.get("x_1"), Some(&Granularity::THREE));
        assert_eq!(map.get("x_2"), Some(&Granularity::FIVE));
        assert_eq!(map.get("x_3"), Some(&Granularity(12)));

        let g1 = Granularity::try_from("2").unwrap();
        let g2: Granularity = "eight".parse().unwrap();

        assert_eq!(g1, Granularity::TWO);
        assert_eq!(g2, Granularity::EIGHT);
    }
}

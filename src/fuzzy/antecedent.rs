use std::{collections::HashMap, fmt};

use serde::{Deserialize, Serialize};

use crate::{fuzzy::TNorm, types::FuzzyValues};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Antecedents {
    pub data: HashMap<String, String>,
}

impl Antecedents {
    pub fn new(data: HashMap<String, String>) -> Self {
        Self { data }
    }

    pub fn get_value(&self, feature: &str) -> Option<&String> {
        self.data.get(feature)
    }

    /// Safely computes the input center map.
    /// Returns `None` if any feature or group is missing from `centers`.
    pub fn get_input_center(
        &self,
        centers: &HashMap<String, FuzzyValues>,
    ) -> Option<HashMap<String, f64>> {
        let mut input_center = HashMap::with_capacity(self.data.len());
        for (feature, group) in &self.data {
            let center = centers.get(feature).and_then(|m| m.get(group))?;
            input_center.insert(feature.clone(), *center);
        }
        Some(input_center)
    }

    /// Computes the Euclidean distance between two antecedents.
    /// Returns `f64::INFINITY` if antecedents are empty, feature sets mismatch, or centers are missing.
    pub fn distance(&self, other: &Antecedents, centers: &HashMap<String, FuzzyValues>) -> f64 {
        if self.data.is_empty() || other.data.is_empty() || self.data.len() != other.data.len() {
            return f64::INFINITY;
        }

        let mut sum_sq: f64 = 0.0;
        for (feature, group) in &self.data {
            let other_group = match other.data.get(feature) {
                Some(g) => g,
                None => return f64::INFINITY,
            };

            let c1 = match centers.get(feature).and_then(|m| m.get(group)) {
                Some(val) => val,
                None => return f64::INFINITY,
            };

            let c2 = match centers.get(feature).and_then(|m| m.get(other_group)) {
                Some(val) => val,
                None => return f64::INFINITY,
            };

            sum_sq += (c1 - c2).powi(2);
        }

        sum_sq.sqrt()
    }
}

impl fmt::Display for Antecedents {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut conditions: Vec<_> = self
            .data
            .iter()
            .map(|(feature, group)| format!("{}={}", feature, group))
            .collect();
        conditions.sort();
        write!(f, "{}", conditions.join(" AND "))
    }
}

/// Calculates rule firing strength based on feature memberships.
pub fn calculate_firing_strength(
    antecedents: &Antecedents,
    memberships: &HashMap<String, HashMap<String, f64>>,
    t_norm: TNorm,
) -> f64 {
    if antecedents.data.is_empty() {
        return 0.0;
    }

    let get_mu = |feature: &str, group: &str| {
        memberships
            .get(feature)
            .and_then(|groups| groups.get(group))
            .copied()
            .unwrap_or(0.0)
    };

    match t_norm {
        TNorm::Product => {
            let mut strength = 1.0;
            for (feature, group) in &antecedents.data {
                strength *= get_mu(feature, group);
                if strength == 0.0 {
                    break;
                }
            }
            strength
        }
        TNorm::Min => {
            let mut min_mu = 1.0;
            for (feature, group) in &antecedents.data {
                let mu = get_mu(feature, group);
                if mu < min_mu {
                    min_mu = mu;
                }
                if min_mu == 0.0 {
                    break;
                }
            }
            min_mu
        }
    }
}

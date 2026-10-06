use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fmt;

use crate::fuzzy::TNorm;

/// Represents the antecedent condition of a fuzzy rule (e.g., `x1 IS low AND x2 IS high`).
///
/// Uses `BTreeMap` internally so feature names are stored in canonical lexicographical order,
/// enabling `Hash`, `Eq`, `Ord`, and `PartialOrd` to be derived automatically.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Antecedents {
    pub data: BTreeMap<String, String>,
}

impl Antecedents {
    /// Creates a new `Antecedents` instance from an iterator of feature-label pairs.
    pub fn new(data: impl IntoIterator<Item = (String, String)>) -> Self {
        Self {
            data: data.into_iter().collect(),
        }
    }

    /// Retrieves the center values ($c$) for each feature's assigned membership function label.
    ///
    /// Used during minimum Euclidean distance calculations when no rules fire.
    pub fn get_input_center(
        &self,
        mf_centers: &HashMap<String, HashMap<String, f64>>,
    ) -> Option<HashMap<String, f64>> {
        let mut input_centers = HashMap::with_capacity(self.data.len());

        for (feature, label) in &self.data {
            let center = mf_centers.get(feature)?.get(label)?;
            input_centers.insert(feature.clone(), *center);
        }

        Some(input_centers)
    }
}

impl fmt::Display for Antecedents {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let terms: Vec<String> = self
            .data
            .iter()
            .map(|(feature, label)| format!("{} IS {}", feature, label))
            .collect();

        write!(f, "{}", terms.join(" AND "))
    }
}

/// Computes the overall firing strength ($\mu_A(x)$) of an antecedent condition
/// using the specified T-Norm operator (`Product` or `Minimum`).
pub fn calculate_firing_strength(
    antecedents: &Antecedents,
    row_memberships: &HashMap<String, HashMap<String, f64>>,
    t_norm: TNorm,
) -> f64 {
    if antecedents.data.is_empty() {
        return 0.0;
    }

    let mut firing_strength = 1.0;

    for (feature, label) in &antecedents.data {
        let mu = row_memberships
            .get(feature)
            .and_then(|fuzzy_sets| fuzzy_sets.get(label))
            .copied()
            .unwrap_or(0.0);

        match t_norm {
            TNorm::Product => {
                firing_strength *= mu;
            }
            TNorm::Minimum => {
                firing_strength = firing_strength.min(mu);
            }
        }

        // Short-circuit execution if firing strength reaches zero
        if firing_strength <= 0.0 {
            return 0.0;
        }
    }

    firing_strength
}

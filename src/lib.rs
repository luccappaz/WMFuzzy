use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

pub mod wm_model;

/// Linguistic partitioning granularity for a feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Granularity {
    Three = 3,
    Five = 5,
}

impl Granularity {
    #[inline]
    pub fn count(&self) -> usize {
        *self as usize
    }

    pub fn default_labels(&self) -> Vec<String> {
        match self {
            Granularity::Three => vec!["low", "med", "high"],
            Granularity::Five => vec!["very_low", "low", "med", "high", "very_high"],
        }
        .into_iter()
        .map(String::from)
        .collect()
    }
}

/// T-norm operator used for rule antecedent aggregation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TNorm {
    Product,
    Min,
}

/// Represents the antecedent conjunction (IF feature_1 IS group_1 AND ...).
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

    /// Compute the input center given fuzzy groups max values (center)
    pub fn get_input_center<'a>(
        &'a self,
        centers: &'a HashMap<String, HashMap<String, f64>>,
    ) -> HashMap<String, f64> {
        let mut input_center = HashMap::new();
        for (feature, group) in &self.data {
            let c1 = centers.get(feature).and_then(|m| m.get(group)).unwrap();
            input_center.insert(feature.into(), *c1);
        }
        input_center
    }

    /// Computes the distance between each antecedents based on theirs input center
    pub fn distance(
        &self,
        other: &Antecedents,
        centers: &HashMap<String, HashMap<String, f64>>,
    ) -> f64 {
        if self.data.is_empty() || other.data.is_empty() {
            return f64::INFINITY;
        }
        let input_center = self.get_input_center(centers);
        let other_input_center = other.get_input_center(centers);

        let mut sum_sq = 0.0;
        for feature in self.data.keys() {
            let c1 = input_center.get(feature);
            let c2 = other_input_center.get(feature);

            match (c1, c2) {
                (Some(&c1), Some(&c2)) => sum_sq += (c1 - c2).powi(2),
                _ => return f64::INFINITY,
            }
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

/// Represents an individual fuzzy rule.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FuzzyRule {
    pub antecedents: Antecedents,
    pub weighted_average: f64,
    pub weighted_variance: f64,
    pub doc: f64,
    pub support: f64,
}

impl fmt::Display for FuzzyRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "IF {} THEN y is centered at {:.4} with variance of {:.4} and doc of {:.4}",
            self.antecedents, self.weighted_average, self.weighted_variance, self.doc
        )
    }
}

/// Helper struct for activated rules during inference.
#[derive(Debug)]
pub struct ActivatedRule<'a> {
    pub rule: &'a FuzzyRule,
    pub firing_strength: f64,
}

/// Binary classification evaluation metrics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvaluationMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub confusion_matrix: [[usize; 2]; 2],
}

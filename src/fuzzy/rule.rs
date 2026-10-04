use std::fmt;

use serde::{Deserialize, Serialize};

use crate::fuzzy::Antecedents;

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

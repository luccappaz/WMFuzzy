use serde::{Deserialize, Serialize};
use std::fmt;

use crate::fuzzy::Antecedents;

/// Representa o tipo de consequente da regra fuzzy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Consequent {
    /// Case 1: The target attribute y has fuzzy groups (ex: "y IS high")
    Linguistic { label: String },
    /// Case 2: The target attribute y does not have fuzzy partitions
    Numeric { weighted_variance: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FuzzyRule {
    pub antecedents: Antecedents,
    pub consequent: Consequent,
    /// In both cases, the numeric value for defuzzification is kept:
    /// - Case 1: The center of the membership value for the group B
    /// - Case 2: The weighted average value \bar{y}
    pub weighted_average: f64,
    pub doc: f64,
    pub support: f64,
}

impl FuzzyRule {
    /// Builder for the case 1 rules
    pub fn linguistic(
        antecedents: Antecedents,
        label: impl Into<String>,
        center: f64,
        doc: f64,
        support: f64,
    ) -> Self {
        Self {
            antecedents,
            consequent: Consequent::Linguistic {
                label: label.into(),
            },
            weighted_average: center,
            doc,
            support,
        }
    }

    /// Builder for the case 2 rules, with the numeric consequent
    pub fn numeric(
        antecedents: Antecedents,
        weighted_average: f64,
        weighted_variance: f64,
        doc: f64,
        support: f64,
    ) -> Self {
        Self {
            antecedents,
            consequent: Consequent::Numeric { weighted_variance },
            weighted_average,
            doc,
            support,
        }
    }

    /// Return the semantic label for the Case 1
    pub fn label(&self) -> Option<&str> {
        match &self.consequent {
            Consequent::Linguistic { label } => Some(label.as_str()),
            Consequent::Numeric { .. } => None,
        }
    }
}

/// Display implementation according to each Case
impl fmt::Display for FuzzyRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.consequent {
            Consequent::Linguistic { label } => {
                write!(
                    f,
                    "IF {} THEN y IS {} (doc: {:.4}, support: {:.4})",
                    self.antecedents, label, self.doc, self.support
                )
            }
            Consequent::Numeric { weighted_variance } => {
                write!(
                    f,
                    "IF {} THEN y is centered at {:.4} with variance of {:.4} and doc of {:.4}",
                    self.antecedents, self.weighted_average, weighted_variance, self.doc
                )
            }
        }
    }
}

/// Helper struct for activated rules during inference.
#[derive(Debug)]
pub struct ActivatedRule<'a> {
    pub rule: &'a FuzzyRule,
    pub firing_strength: f64,
}

use serde::{Deserialize, Serialize};

/// Binary classification evaluation metrics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvaluationMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub confusion_matrix: [[usize; 2]; 2],
}

use serde::{Deserialize, Serialize};

/// Metrics for binary classification evaluation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClassificationMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub confusion_matrix: [[u32; 2]; 2],
}

/// Metrics for continuous regression evaluation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegressionMetrics {
    pub mse: f64,
    pub rmse: f64,
    pub mae: f64,
    pub r2: f64,
    pub mape: f64,
}

/// Consolidated evaluation metrics supporting both classification and regression tasks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum EvaluationMetrics {
    Classification(ClassificationMetrics),
    Regression(RegressionMetrics),
}

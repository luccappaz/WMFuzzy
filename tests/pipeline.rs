use crate::common::setup_test_model;
use std::collections::HashMap;

use wm_fuzzy::{
    Granularity, TNorm,
    wm_model::{WMModel, WMModelError},
};

mod common;

#[test]
fn test_build_model_limits_mismatch() {
    let mut model = WMModel::default();
    let mut gran = HashMap::new();
    let mut limits = HashMap::new();

    // Granularity::Three expects 3 limit thresholds, but only 2 are provided
    gran.insert("feature1".to_string(), Granularity::Three);
    limits.insert("feature1".to_string(), vec![0.0, 10.0]); // Invalid count

    let result = model.build(gran, limits);
    assert!(matches!(result, Err(WMModelError::LimitsMismatch(2, _, 3))));
}

#[test]
fn test_build_model_missing_granularity() {
    let mut model = WMModel::default();
    let gran = HashMap::new();
    let mut limits = HashMap::new();

    limits.insert("feature1".to_string(), vec![0.0, 5.0, 10.0]);

    let result = model.build(gran, limits);
    assert!(matches!(result, Err(WMModelError::MissingGranularity(_))));
}

#[test]
fn test_generate_rules() {
    let (mut model, df_x, s_y) = setup_test_model();

    let rules = model
        .generate_rules(
            &df_x,
            &s_y,
            None,
            TNorm::Product,
            0.1, // min_support
            0.0, // min_confidence
        )
        .unwrap();

    assert!(!rules.is_empty(), "Rules should have been generated");
    assert_eq!(model.rule_count(), rules.len());
}

#[test]
fn test_predict_and_fallback() {
    let (mut model, df_x, s_y) = setup_test_model();

    model
        .generate_rules(&df_x, &s_y, None, TNorm::Product, 0.1, 0.0)
        .unwrap();

    let (cont_preds, bin_preds) = model
        .predict(
            &df_x,
            None,
            Some(50.0), // Threshold
            0.0,        // Fallback value
            TNorm::Product,
        )
        .unwrap();

    assert_eq!(cont_preds.len(), 3);
    assert_eq!(bin_preds.len(), 3);
}

#[test]
fn test_min_distance_rule() {
    let (mut model, df_x, s_y) = setup_test_model();
    model
        .generate_rules(&df_x, &s_y, None, TNorm::Product, 0.1, 0.0)
        .unwrap();

    let mut crisp_row = HashMap::new();
    crisp_row.insert("x1".to_string(), 2.5); // Midway between 0.0 and 5.0

    let (rule, dist) = model.min_distance_rule(&crisp_row);
    assert!(rule.is_some());
    assert!(dist.unwrap() >= 0.0);
}

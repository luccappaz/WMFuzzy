use polars::prelude::*;
use std::collections::HashMap;
use tempfile::NamedTempFile;
use wm_fuzzy::{fuzzy::TNorm, model::WMModel};

/// Integration Test 1: Full pipeline using JSON configuration,
/// Polars DataFrame rule extraction, and batch prediction.
#[test]
fn test_e2e_json_config_and_prediction_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Compact JSON model definition
    let json_config = r#"{
        "kind": "triangular",
        "features": {
            "temperature": {
                "granularity": 3,
                "strategy": {
                    "type": "linear",
                    "min": 0.0,
                    "max": 100.0
                },
                "labels": ["low", "medium", "high"]
            },
            "humidity": {
                "strategy": {
                    "type": "custom",
                    "knots": [0.0, 50.0, 100.0]
                }
            }
        }
    }"#;

    // 2. Initialize and build model directly from JSON
    let mut model = WMModel::from_json(json_config)?;
    assert!(model.is_configured());

    // 3. Prepare training dataset using Polars
    let x_train = df![
        "temperature" => &[10.0, 20.0, 50.0, 80.0, 90.0],
        "humidity" => &[10.0, 30.0, 50.0, 70.0, 90.0],
    ]?;
    let y_train = Series::new("target".into(), &[0.1, 0.2, 0.5, 0.8, 0.9]);

    // 4. Generate fuzzy rules using Wang-Mendel algorithm
    let rules = model.generate_rules(&x_train, &y_train, None, TNorm::Product, 0.01, 0.1)?;
    assert!(!rules.is_empty());
    assert_eq!(model.rule_count(), rules.len());

    // 5. Batch predict on test dataset
    let x_test = df![
        "temperature" => &[15.0, 85.0],
        "humidity" => &[20.0, 80.0],
    ]?;

    let (preds, binary_preds) = model.predict(&x_test, None, Some(0.5), 0.0, TNorm::Product)?;
    assert_eq!(preds.len(), 2);
    assert_eq!(binary_preds.len(), 2);

    // Verify predictions remain within valid range [0.0, 1.0]
    for &p in &preds {
        assert!((0.0..=1.0).contains(&p));
    }

    Ok(())
}

/// Integration Test 2: Fluent Builder API, active rule firing,
/// minimum distance fallback for out-of-bounds input, and disk persistence.
#[test]
fn test_e2e_builder_inference_fallback_and_persistence() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Programmatic Builder Setup
    let mut model = WMModel::new();
    model
        .set_kind("triangular")?
        .add_custom_strategy("feature_a", vec![0.0, 5.0, 10.0])?
        .add_custom_strategy("feature_b", vec![0.0, 2.5, 5.0, 7.5, 10.0])?;

    model.build()?;
    assert!(model.is_configured());

    // 2. Train model on sample data
    let x_train = df![
        "feature_a" => &[1.0, 5.0, 9.0],
        "feature_b" => &[2.0, 5.0, 8.0],
    ]?;
    let y_train = Series::new("target".into(), &[0.2, 0.5, 0.9]);

    model.generate_rules(&x_train, &y_train, None, TNorm::Min, 0.0, 0.0)?;
    assert!(model.rule_count() > 0);

    // 3. Single-Sample Inference with Active Rules
    let mut in_bounds_sample = HashMap::new();
    in_bounds_sample.insert("feature_a".to_string(), 5.0);
    in_bounds_sample.insert("feature_b".to_string(), 5.0);

    let (y_pred, active_rules) = model.infer(&in_bounds_sample, TNorm::Min);
    assert!(y_pred.is_some());
    assert!(!active_rules.is_empty());

    // 4. Out-of-bounds Sample testing Minimum Distance Fallback
    let mut extreme_sample = HashMap::new();
    extreme_sample.insert("feature_a".to_string(), 999.0);
    extreme_sample.insert("feature_b".to_string(), 999.0);

    let (_, active_fallback) = model.infer(&extreme_sample, TNorm::Min);
    assert!(
        active_fallback.is_empty(),
        "Extreme values should fire 0 active rules"
    );

    let (nearest_rule, dist) = model.min_distance_rule(&extreme_sample);
    assert!(nearest_rule.is_some());
    assert!(dist.is_some());

    // 5. Rule File Persistence
    let temp_file = NamedTempFile::new()?;
    let path_str = temp_file.path().to_str().unwrap();

    assert!(model.save_rules(path_str).is_ok());
    assert!(temp_file.path().metadata()?.len() > 0);

    Ok(())
}

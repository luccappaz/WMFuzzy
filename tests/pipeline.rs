use polars::prelude::*;
use std::collections::HashMap;
use tempfile::NamedTempFile;
use wm_fuzzy::{fuzzy::TNorm, metrics::EvaluationMetrics, model::WMModel};

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

    model.generate_rules(&x_train, &y_train, None, TNorm::Minimum, 0.0, 0.0)?;
    assert!(model.rule_count() > 0);

    // 3. Single-Sample Inference with Active Rules
    let mut in_bounds_sample = HashMap::new();
    in_bounds_sample.insert("feature_a".to_string(), 5.0);
    in_bounds_sample.insert("feature_b".to_string(), 5.0);

    let (y_pred, active_rules) = model.infer(&in_bounds_sample, TNorm::Minimum);
    assert!(y_pred.is_some());
    assert!(!active_rules.is_empty());

    // 4. Out-of-bounds Sample testing Minimum Distance Fallback
    let mut extreme_sample = HashMap::new();
    extreme_sample.insert("feature_a".to_string(), 999.0);
    extreme_sample.insert("feature_b".to_string(), 999.0);

    let (_, active_fallback) = model.infer(&extreme_sample, TNorm::Minimum);
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

    // 6. Verify reloading rules from disk
    let mut reloaded_model = WMModel::new();
    reloaded_model.load_rules(path_str)?;
    assert_eq!(reloaded_model.rule_count(), model.rule_count());

    Ok(())
}

/// Integration Test 3: Case 1 E2E Pipeline with Target Variable Partitioning
/// (Linguistic Consequent, JSON Config, and Classification Metrics Evaluation)
#[test]
fn test_e2e_linguistic_target_classification() -> Result<(), Box<dyn std::error::Error>> {
    // 1. JSON Specification including Target fuzzy partitioning
    let json_config = r#"{
        "kind": "triangular",
        "features": {
            "pressure": {
                "granularity": 3,
                "strategy": {
                    "type": "linear",
                    "min": 0.0,
                    "max": 100.0
                },
                "labels": ["low", "normal", "high"]
            }
        },
        "target": {
            "granularity": 3,
            "strategy": {
                "type": "linear",
                "min": 0.0,
                "max": 1.0
            },
            "labels": ["safe", "warning", "danger"]
        }
    }"#;

    let mut model = WMModel::from_json(json_config)?;
    let target_mfs = model.target_mfs().unwrap();
    assert_eq!(target_mfs.len(), 3);

    // 2. Create training dataset
    let x_train = df!["pressure" => &[10.0, 50.0, 90.0]]?;
    let y_train = Series::new("target".into(), &[0.1, 0.5, 0.9]);

    model.generate_rules(&x_train, &y_train, None, TNorm::Product, 0.0, 0.0)?;
    assert!(model.rule_count() > 0);

    // 3. Evaluate Binary Classification Performance & Export JSON Metrics
    let temp_metrics_file = NamedTempFile::new()?;
    let metrics_eval = model.evaluate(
        &x_train,
        &y_train,
        Some(0.5),
        Some(0.0),
        Some(temp_metrics_file.path()),
    )?;

    if let EvaluationMetrics::Classification(clf) = metrics_eval {
        assert!(clf.accuracy >= 0.0 && clf.accuracy <= 1.0);
        assert!(clf.f1_score >= 0.0);
    } else {
        panic!("Expected EvaluationMetrics::Classification variant");
    }

    assert!(temp_metrics_file.path().metadata()?.len() > 0);

    Ok(())
}

/// Integration Test 4: Case 2 E2E Pipeline with Continuous Numeric Target
/// (Continuous Regression Metrics Evaluation & JSON Export)
#[test]
fn test_e2e_numeric_target_regression() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Builder API without target fuzzy partitions
    let mut model = WMModel::new();
    model
        .set_kind("triangular")?
        .add_linear_strategy("sensor_input", 0.0, 10.0)?
        .add_granularity("sensor_input", 3)?;

    model.build()?;

    // 2. Prepare continuous regression dataset
    let x_data = df!["sensor_input" => &[1.0, 2.5, 5.0, 7.5, 9.0]]?;
    let y_data = Series::new("target".into(), &[10.5, 25.0, 50.2, 74.8, 91.0]);

    model.generate_rules(&x_data, &y_data, None, TNorm::Product, 0.0, 0.0)?;

    // Check that target_mfs() returns Err when no target fuzzy sets are configured
    assert!(model.target_mfs().is_err());

    // 3. Evaluate Continuous Regression Performance (threshold = None)
    let temp_metrics_file = NamedTempFile::new()?;
    let metrics_eval = model.evaluate(
        &x_data,
        &y_data,
        None,
        Some(0.0),
        Some(temp_metrics_file.path()),
    )?;

    if let EvaluationMetrics::Regression(reg) = metrics_eval {
        assert!(reg.mse >= 0.0);
        assert!(reg.rmse >= 0.0);
        assert!(reg.mae >= 0.0);
        assert!(reg.r2 <= 1.0);
        assert!(reg.mape >= 0.0);
    } else {
        panic!("Expected EvaluationMetrics::Regression variant");
    }

    assert!(temp_metrics_file.path().metadata()?.len() > 0);

    Ok(())
}

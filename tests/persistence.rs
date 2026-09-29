use wm_fuzzy::{TNorm, wm_model::WMModel};

use crate::common::setup_test_model;
use std::fs;

mod common;
#[test]
fn test_save_and_load_rules() {
    let (mut model, df_x, s_y) = setup_test_model();
    model
        .generate_rules(&df_x, &s_y, None, TNorm::Product, 0.1, 0.0)
        .unwrap();

    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_rules.json");

    // Export rules to JSON
    model.save_rules(&file_path).unwrap();
    assert!(file_path.exists());

    // Import rules into a clean instance
    let mut new_model = WMModel::default();
    new_model.load_rules(&file_path).unwrap();

    assert_eq!(new_model.rule_count(), model.rule_count());

    // Clean up temporary artifact
    let _ = fs::remove_file(file_path);
}

use std::collections::HashMap;
use std::fs;

use wm_fuzzy::wm_model::plot_all_membership_functions;

#[test]
fn test_plot_all_membership_functions() {
    let mut limits = HashMap::new();
    limits.insert("temperature".to_string(), vec![0.0, 15.0, 30.0]);

    let temp_dir = std::env::temp_dir().join("wm_fuzzy_plots");
    let result = plot_all_membership_functions(&limits, temp_dir.to_str().unwrap());

    assert!(result.is_ok());
    let expected_file = temp_dir.join("mf_temperature.png");
    assert!(expected_file.exists());

    // Clean up directory
    let _ = fs::remove_dir_all(temp_dir);
}

use std::collections::HashMap;

use polars::prelude::*;
use wm_fuzzy::{Granularity, wm_model::WMModel};

pub fn setup_test_model() -> (WMModel, DataFrame, Series) {
    let mut gran = HashMap::new();
    gran.insert("x1".to_string(), Granularity::Three); // [Low, Medium, High]

    let mut limits = HashMap::new();
    limits.insert("x1".to_string(), vec![0.0, 5.0, 10.0]);

    let mut model = WMModel::default();
    model.build(gran, limits).unwrap();

    let df_x = df!(
        "x1" => &[0.0, 5.0, 10.0]
    )
    .unwrap();

    let s_y = Series::new(PlSmallStr::from_str("y"), &[0.0, 50.0, 100.0]);

    (model, df_x, s_y)
}

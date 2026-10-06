use polars::prelude::*;
use std::error::Error;
use std::path::PathBuf;

use wm_fuzzy::fuzzy::TNorm;
use wm_fuzzy::metrics::RegressionMetrics;
use wm_fuzzy::model::wm::WMModel;

pub fn run_wine_pipeline(
    file_path: PathBuf,
    target_column_name: &str,
    ignored_columns: &[&str],
    test_size_ratio: f64,
) -> Result<RegressionMetrics, Box<dyn Error>> {
    // 1. Read external CSV via Polars
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .with_parse_options(CsvParseOptions::default().with_separator(b','))
        .try_into_reader_with_file_path(Some(file_path.clone()))?
        .finish()?;

    println!(
        "📂 Dataset loaded: {:?} (Rows: {}, Columns: {})",
        file_path,
        df.height(),
        df.width()
    );

    // 2. Perform Train / Test Split FIRST (e.g., 80% train, 20% test with deterministic shuffle)
    let df_shuffled = df.sample_n_literal(df.height(), false, Some(true), Some(42))?;
    let train_size = ((df.height() as f64) * (1.0 - test_size_ratio)) as usize;
    let test_size = df.height() - train_size;

    let df_train = df_shuffled.slice(0, train_size);
    let df_test = df_shuffled.slice(train_size as i64, test_size);

    let target_train = df_train
        .column(target_column_name)?
        .as_materialized_series()
        .clone();
    let target_test = df_test
        .column(target_column_name)?
        .as_materialized_series()
        .clone();

    println!(
        "✂️ Dataset Split: Train = {} rows | Test = {} rows",
        df_train.height(),
        df_test.height()
    );

    // 3. Identify feature column names
    let feature_names: Vec<&str> = df
        .get_column_names()
        .into_iter()
        .map(|col| col.as_str())
        .filter(|&col| col != target_column_name && !ignored_columns.contains(&col))
        .collect();

    // 4. Configure WMModel domain boundaries strictly using Training Data (prevents data leakage)
    let mut model = WMModel::new();
    model.set_kind("triangular")?;

    for &feature in &feature_names {
        let series = df_train.column(feature)?.cast(&DataType::Float64)?;
        let ca = series.f64()?;
        let min = ca.min().ok_or("Failed to compute train feature min")?;
        let max = ca.max().ok_or("Failed to compute train feature max")?;

        model.add_granularity(feature, 3)?;
        model.add_linear_strategy(feature, min, max)?;
    }

    let target_train_f64 = target_train.cast(&DataType::Float64)?;
    let target_ca = target_train_f64.f64()?;
    let target_min = target_ca
        .min()
        .ok_or("Failed to compute train target min")?;
    let target_max = target_ca
        .max()
        .ok_or("Failed to compute train target max")?;

    model.add_target_granularity(7)?;
    model.add_target_linear_strategy(target_min, target_max)?;

    // 5. Generate fuzzy rules using ONLY training set
    model.generate_rules(&df_train, &target_train, None, TNorm::Product, 0.0, 0.0)?;
    println!(
        "✅ Model rules generated from train set: {}",
        model.rules()?.len()
    );

    model.save_rules("results/wine_rules.json")?;

    // 6. Evaluate regression using ONLY test set
    let metrics = model.evaluate_regression(&df_test, &target_test, Some(0.0), None)?;

    println!("\n================ TEST SET EVALUATION ================");
    println!("{:#?}", metrics);

    Ok(metrics)
}

fn main() -> Result<(), Box<dyn Error>> {
    let file_path: PathBuf = "data/WineQT.csv".into();
    let _metrics = run_wine_pipeline(file_path, "quality", &["Id"], 0.2)?;
    Ok(())
}

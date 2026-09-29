use polars::prelude::*;
use std::collections::HashMap;
use wm_fuzzy::{
    Granularity, TNorm,
    wm_model::{WMModel, plot_all_membership_functions},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\x1b[1;36m=== 🔮 wm_fuzzy: Interactive Wang-Mendel Demo ===\x1b[0m\n");

    // 1. Model instantiation
    println!("\x1b[1;33m[1/6] Initializing WMModel...\x1b[0m");
    let mut model = WMModel::default();

    // 2. Configure Granularity and Limits for each variable
    let mut granularity = HashMap::new();
    granularity.insert("temperature".to_string(), Granularity::Three); // Low, Medium, High
    granularity.insert("humidity".to_string(), Granularity::Three);

    let mut limits = HashMap::new();
    limits.insert("temperature".to_string(), vec![0.0, 25.0, 50.0]);
    limits.insert("humidity".to_string(), vec![0.0, 50.0, 100.0]);

    model.build(granularity, limits.clone())?;
    println!("  └─ Granularity & Limits configured for 'temperature' and 'humidity'.");

    // 3. Prepare training data (Polars DataFrame and Series)
    println!("\n\x1b[1;33m[2/6] Loading Polars Training Dataset...\x1b[0m");
    let x_train = df![
        "temperature" => &[10.0, 20.0, 30.0, 40.0, 15.0, 35.0],
        "humidity" => &[20.0, 40.0, 60.0, 80.0, 30.0, 75.0],
    ]?;

    let y_train = Series::new("output".into(), &[0.1, 0.4, 0.7, 0.9, 0.2, 0.8]);
    println!("  └─ Train shape: {} rows", x_train.height());

    // 4. Fuzzy Rule Generation / Training
    println!("\n\x1b[1;33m[3/6] Generating Fuzzy Rules (Wang-Mendel Algorithm)...\x1b[0m");
    let min_support = 0.05;
    let min_confidence = 0.1;

    let rules = model.generate_rules(
        &x_train,
        &y_train,
        None,
        TNorm::Product,
        min_support,
        min_confidence,
    )?;

    println!(
        "  └─ \x1b[1;32m✅ Generated {} fuzzy rules.\x1b[0m",
        rules.len()
    );

    // 5. Interactive Inference Prompt via model.prompt_infer
    println!("\n\x1b[1;33m[4/6] Launching Interactive Inference Prompt...\x1b[0m");
    let activated_rules = model.prompt_infer(TNorm::Product)?;
    println!(
        "\n  └─ \x1b[1;32m🔥 Total activated rules: {}\x1b[0m",
        activated_rules.len()
    );

    // 6. Batch prediction on a test DataFrame
    println!("\n\x1b[1;33m[5/6] Executing Batch Prediction...\x1b[0m");
    let x_test = df![
        "temperature" => &[12.0, 38.0],
        "humidity" => &[25.0, 70.0],
    ]?;

    let (continuous_preds, binary_preds) =
        model.predict(&x_test, None, Some(0.5), 0.0, TNorm::Product)?;

    println!("  ├─ 🔮 Continuous predictions: {:?}", continuous_preds);
    println!("  └─ 🔮 Binary predictions:     {:?}", binary_preds);

    // 7. Save generated rules to disk and plot membership functions
    println!("\n\x1b[1;33m[6/6] Exporting Artifacts...\x1b[0m");
    model.save_rules("fuzzy_rules.json")?;
    plot_all_membership_functions(&limits, "./plots")?;
    println!("  ├─ Saved rules to: \x1b[36mfuzzy_rules.json\x1b[0m");
    println!("  └─ Plotted MFs to:  \x1b[36m./plots/\x1b[0m");

    println!("\n\x1b[1;32m✨ Demo completed successfully!\x1b[0m");

    Ok(())
}

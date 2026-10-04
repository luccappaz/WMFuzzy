use polars::prelude::*;
use std::collections::HashMap;
use std::io::{Write, stdin, stdout};
use std::path::PathBuf;
use wm_fuzzy::fuzzy::TNorm;
use wm_fuzzy::model::wm::WMModel;

fn read_user_input(prompt: &str) -> String {
    print!("{}", prompt);
    stdout().flush().expect("Failed to flush output buffer");
    let mut buffer = String::new();
    stdin()
        .read_line(&mut buffer)
        .expect("Failed to read user input");
    buffer.trim().to_string()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let csv_path = manifest_dir.join("data").join("data.csv");

    let sex_input = read_user_input("Enter Sex for filtering (0 or 1): ");
    let target_sex: f64 = sex_input
        .parse()
        .map_err(|_| "Invalid input for Sex field. Enter 0 or 1.")?;

    let selected_features = [
        "age",
        "waist_thigh_ratio",
        "waist_hip_ratio",
        "sleep_hours_per_night",
        "physical_activity_categorized",
    ];

    // Explicit name of the output column
    const TARGET: &str = "metabolic_syndrome";

    // Prepare the list of all required columns (Features + Target)
    let mut all_cols: Vec<String> = selected_features.iter().map(|&s| s.to_string()).collect();
    all_cols.push(TARGET.to_string());

    // Polars Pipeline: Read -> Filter -> Select Columns
    let df = LazyCsvReader::new(PlRefPath::new(csv_path.to_string_lossy()))
        .finish()?
        .filter(col("sex").eq(lit(target_sex)))
        .select(
            all_cols
                .iter()
                .map(|name| col(PlSmallStr::from_str(name)))
                .collect::<Vec<_>>(),
        )
        .collect()?;

    if df.height() == 0 {
        return Err(format!("No samples found for sex == {}", target_sex).into());
    }

    // Map target partition counts for each feature
    let feature_granularities: HashMap<String, usize> = [
        ("age".to_string(), 3),
        ("waist_thigh_ratio".to_string(), 5),
        ("waist_hip_ratio".to_string(), 3),
        ("sleep_hours_per_night".to_string(), 3),
        ("physical_activity_categorized".to_string(), 3),
    ]
    .into_iter()
    .collect();

    // Compute quantile limits (custom grid knots) using Polars
    let limits = compute_limits(&df, &feature_granularities)?;

    // Initialize model and configure strategies from computed quantile knots
    let mut model = WMModel::new();
    model.set_kind("triangular")?;

    for (feature, knots) in &limits {
        model.add_custom_strategy(feature, knots.clone())?;
    }

    model.build()?;

    // Generate and save Membership Function (MF) plots
    model.plot_membership_functions("./plots")?;

    println!(
        "✅ Model configured: Configured MFs = {}",
        model.is_configured()
    );

    let y = df.column(TARGET).unwrap().as_materialized_series().clone();
    let x = df.drop(TARGET).unwrap();

    // Generate fuzzy rules directly from DataFrame
    let rules = model.generate_rules(&x, &y, None, TNorm::Product, 0.01, 0.2)?;

    println!("✅ Rules generated successfully: {}", rules.len());
    println!(
        "Filtered samples (sex == {}): {}\n",
        target_sex,
        df.height()
    );

    // =========================================================================
    // Interactive Inference
    // =========================================================================
    println!("--- Input Values for Fuzzy Inference ---");
    let mut sample_input: HashMap<String, f64> = HashMap::new();

    for feature in &selected_features {
        let raw_val = read_user_input(&format!("Enter value for '{}': ", feature));
        let val: f64 = raw_val
            .parse()
            .map_err(|_| format!("Invalid numeric value entered for feature '{}'", feature))?;
        sample_input.insert((*feature).to_string(), val);
    }

    println!("\n--- Inference Result ---");

    // Evaluate firing strength of each rule for the input vector
    let (mut y_pred, active_rules) = model.infer(&sample_input, TNorm::Product);

    // Variable to store the final estimation of the model
    if !active_rules.is_empty() || y_pred.is_some() {
        println!("🔥 Active Rules ({}):", active_rules.len());

        for (idx, active_rule) in active_rules.into_iter().enumerate() {
            println!(
                "  [{}] Strength = {:.4} | {}",
                idx + 1,
                active_rule.firing_strength,
                active_rule.rule
            );
        }

        println!("\n👉 Weighted Average (ȳ): {:.4}", y_pred.unwrap());
    } else {
        println!(
            "⚠️ No rule was directly activated (μ = 0). Searching for the rule with minimum distance..."
        );
        // Find the rule with smallest normalized Euclidean distance
        let nearest_rule = model.min_distance_rule(&sample_input);
        if let (Some(rule), Some(dist)) = nearest_rule {
            println!("🎯 Selected Rule (Minimum Distance = {:.4}):", dist);
            println!(
                "  • IF {} THEN y = {:.4} (DoC = {:.4})",
                rule.antecedents, rule.weighted_average, rule.doc
            );
            println!("\n👉 Assumed Average (ȳ): {:.4}", rule.weighted_average);
            y_pred = Some(rule.weighted_average);
        } else {
            println!("❌ Could not find any nearby rule. Check the configured limits.");
        }
    }

    // =========================================================================
    // Threshold Check
    // =========================================================================
    const THRESHOLD: f64 = 0.48;

    if let Some(estimate) = y_pred
        && estimate > THRESHOLD
    {
        println!(
            "\n🚨 WARNING: Estimate ({:.4}) above threshold of {:.2} -> seek medical advice!",
            estimate, THRESHOLD
        );
    }

    println!("\n📊 Model Summary:");
    println!("   • Total rules: {}", model.rule_count());
    println!("   • Valid configuration: {}", model.is_configured());

    Ok(())
}

fn compute_limits(
    df: &DataFrame,
    feature_granularities: &HashMap<String, usize>,
) -> Result<HashMap<String, Vec<f64>>, Box<dyn std::error::Error>> {
    let mut limits = HashMap::new();

    for (feature, &count) in feature_granularities {
        let feature_ps = PlSmallStr::from_str(feature.as_str());

        if !df.get_column_names().contains(&&feature_ps) {
            continue;
        }

        let feature_limits = match count {
            3 => {
                let result = df
                    .clone()
                    .lazy()
                    .select([
                        col(feature_ps.clone())
                            .quantile(lit(0.0), QuantileMethod::Nearest)
                            .alias("q0"),
                        col(feature_ps.clone())
                            .quantile(lit(0.50), QuantileMethod::Nearest)
                            .alias("q50"),
                        col(feature_ps.clone())
                            .quantile(lit(1.0), QuantileMethod::Nearest)
                            .alias("q100"),
                    ])
                    .collect()?;

                vec![
                    result.column("q0")?.f64()?.get(0).unwrap_or(0.0),
                    result.column("q50")?.f64()?.get(0).unwrap_or(0.0),
                    result.column("q100")?.f64()?.get(0).unwrap_or(0.0),
                ]
            }
            5 => {
                let result = df
                    .clone()
                    .lazy()
                    .select([
                        col(feature_ps.clone())
                            .quantile(lit(0.0), QuantileMethod::Nearest)
                            .alias("q0"),
                        col(feature_ps.clone())
                            .quantile(lit(0.25), QuantileMethod::Nearest)
                            .alias("q25"),
                        col(feature_ps.clone())
                            .quantile(lit(0.50), QuantileMethod::Nearest)
                            .alias("q50"),
                        col(feature_ps.clone())
                            .quantile(lit(0.75), QuantileMethod::Nearest)
                            .alias("q75"),
                        col(feature_ps.clone())
                            .quantile(lit(1.0), QuantileMethod::Nearest)
                            .alias("q100"),
                    ])
                    .collect()?;

                vec![
                    result.column("q0")?.f64()?.get(0).unwrap_or(0.0),
                    result.column("q25")?.f64()?.get(0).unwrap_or(0.0),
                    result.column("q50")?.f64()?.get(0).unwrap_or(0.0),
                    result.column("q75")?.f64()?.get(0).unwrap_or(0.0),
                    result.column("q100")?.f64()?.get(0).unwrap_or(0.0),
                ]
            }
            _ => return Err(format!("Unsupported granularity count: {}", count).into()),
        };

        limits.insert(feature.clone(), feature_limits);
    }

    Ok(limits)
}

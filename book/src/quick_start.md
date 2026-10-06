# Quick Start

Get **WMFuzzy** up and running in your Rust project in under 5 minutes.

---

## 📦 1. Add WMFuzzy to Your Dependencies

Add `wm_fuzzy` to your `Cargo.toml`:

```toml
[dependencies]
wm_fuzzy = "0.3"
polars = "0.46" # Required when working with Polars DataFrames
```

> **Note**: `polars` support is enabled by default. If you only need core fuzzy logic without DataFrame integration, disable default features:
> ```toml
> wm_fuzzy = { version = "0.3", default-features = false }
> ```

---

## ⚡ 2. Choose Your Workflow

WMFuzzy offers two primary APIs for model initialization: **Declarative JSON** (recommended for production configurations) and the **Fluent Builder API** (recommended for dynamic or programmatic setups).

### Option A: Declarative JSON Configuration

Define your membership functions, partition granularities, and linguistic labels in a single JSON schema.

```rust
use polars::prelude::*;
use wm_fuzzy::{wm_model::WMModel, TNorm};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Define feature partitions and linguistic terms in JSON
    let json_config = r#"{
        "kind": "triangular",
        "features": {
            "temperature": {
                "granularity": 3,
                "strategy": { "type": "linear", "min": 0.0, "max": 100.0 },
                "labels": ["cold", "warm", "hot"]
            },
            "humidity": {
                "strategy": { "type": "custom", "knots": [0.0, 50.0, 100.0] }
            }
        }
    }"#;

    // 2. Parse configuration and build membership functions
    let mut model = WMModel::from_json(json_config)?;

    // 3. Create training dataset using Polars
    let x_train = df![
        "temperature" => &[15.0, 50.0, 85.0],
        "humidity" => &[20.0, 50.0, 80.0],
    ]?;
    let y_train = Series::new("target".into(), &[0.1, 0.5, 0.9]);

    // 4. Extract fuzzy rules with the Wang-Mendel algorithm
    let rules = model.generate_rules(
        &x_train,
        &y_train,
        None,           // Optional rule weights
        TNorm::Product, // Antecedent aggregation operator
        0.01,           // Minimum Degree of Consistency (DoC) threshold
        0.1,            // Minimum rule activation threshold
    )?;

    println!("Extracted {} fuzzy rules.", rules.len());

    // 5. Perform batch prediction on new data
    let x_test = df![
        "temperature" => &[20.0, 80.0],
        "humidity" => &[30.0, 75.0],
    ]?;

    let (predictions, binary_predictions) = model.predict(
        &x_test,
        None,           // Optional custom target column name
        Some(0.5),      // Binary classification threshold
        0.0,            // Fallback default value
        TNorm::Product,
    )?;

    println!("Continuous predictions (ȳ): {:?}", predictions);
    println!("Binary classification outputs: {:?}", binary_predictions);

    Ok(())
}
```

---

### Option B: Programmatic Fluent Builder API

Construct models step-by-step when partition boundaries need to be calculated at runtime (e.g., using dataset quantiles).

```rust
use std::collections::HashMap;
use wm_fuzzy::{wm_model::WMModel, MFKind, TNorm};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = WMModel::new();

    // Configure model invariants fluently
    model
        .set_kind(MFKind::Triangular)?
        .add_linear_strategy("temperature", 0.0, 100.0, 3)?
        .add_labels("temperaure", vec!["cold", "warm", "hot"])
        .add_custom_strategy("humidity", vec![0.0, 25.0, 50.0, 75.0, 100.0])?;

    // Build internal membership function maps
    model.build()?;

    // Perform single-sample inference
    let mut x_train = HashMap::new();
    x_train.insert("temperature".to_string(), 45.0);
    x_train.insert("humidity".to_string(), 60.0);

    let y_train = Series::new("target".into(), &[0.1, 0.5, 0.9]);

    // 4. Extract fuzzy rules with the Wang-Mendel algorithm
    let rules = model.generate_rules(
        &x_train,
        &y_train,
        None,           // Optional rule weights
        TNorm::Product, // Antecedent aggregation operator
        0.01,           // Minimum Degree of Consistency (DoC) threshold
        0.1,            // Minimum rule activation threshold
    )?;

    let (y_pred, active_rules) = model.infer(&sample, TNorm::Minimum);

    if let Some(value) = y_pred {
        println!("Defuzzified Output (ȳ): {:.4}", value);
        println!("Fired {} active rules.", active_rules.len());
    } else {
        // Fallback to nearest rule when input falls outside active boundaries
        let (nearest_rule, distance) = model.min_distance_rule(&sample);
        println!("No active rules fired. Nearest rule distance: {:?}", distance);
        println!("Fallback Rule: {:?}", nearest_rule);
    }

    Ok(())
}
```

---

## 📖 Next Steps

Now that you have WMFuzzy up and running, explore the following guides to deepen your understanding:

- **[The Wang-Mendel Algorithm](../concepts/wang_mendel.md)**: Mathematical foundations of rule generation and Degree of Consistency ($DoC$).
- **[Polars Integration](../guides/polars_integration.md)**: Using empirical quantiles to construct knot grids dynamically.
- **[Inference & Fallback Logic](../guides/inference.md)**: Detailed breakdown of T-Norm operators and Euclidean distance fallback.

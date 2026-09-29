# `wm_fuzzy`

[![Crates.io](https://img.shields.io/crates/v/wm_fuzzy.svg)](https://crates.io/crates/wm_fuzzy)
[![Docs.rs](https://img.shields.io/docsrs/wm_fuzzy)](https://docs.rs/wm_fuzzy)
[![License](https://img.shields.io/badge/license-MIT)](#license)
[![CI](https://github.com/luccappaz/wm_model_rust/actions/workflows/ci.yml/badge.svg)](https://github.com/luccappaz/wm_model_rust/actions/workflows/ci.yml)

A high-performance Rust implementation of the **Wang-Mendel (WM) algorithm** for automated fuzzy rule generation, inference, and binary/continuous classification from numerical data.

---

## Features

* **Automated Rule Extraction:** Fast generation of fuzzy IF-THEN rules from numerical datasets via Cartesian partitioning.
* **Algebraic T-Norms:** Configurable firing strength aggregation using **Product** or **Minimum** ($t$-norms).
* **Robust Fallback Engine:** Out-of-distribution handling via continuous Euclidean distance mapping in the membership center space.
* **Native Serialization:** Full JSON support for saving and loading learned rule bases and evaluation metrics (`serde`).
* **Interactive CLI Prompt:** Terminal interface for inspecting fuzzy partitions, $[a, b, c]$ vertices, and querying rule activations.
* **Evaluation & Diagnostics:** Computes Confusion Matrix, Accuracy, Precision, Recall, and F1-Score directly.

---

## Installation

Add `wm_fuzzy` to your project's `Cargo.toml`:

```toml
[dependencies]
wm_fuzzy = "0.2.0"
plotters = "0.3.7"
polars = {version = "0.55.2", features = ["lazy"]}
serde = { version = "1.0.229", features = ["derive"] }
serde_json = "1.0.151"
thiserror = "2.0.20"
```

---

## Quickstart

```rust
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
```

---

## Mathematical Formulation

### 1. Triangular Membership Functions
For parameters $[a, b, c]$:

$$\mu(x) = \begin{cases} 
0 & x \le a \text{ or } x \ge c \\
\frac{x - a}{b - a} & a < x \le b \\
\frac{c - x}{c - b} & b < x < c 
\end{cases}$$

### 2. Rule Firing Strength ($t$-norm)
$$\alpha_k(x) = \prod_{i=1}^{n} \mu_{A_i^k}(x_i) \quad \text{or} \quad \min_{i=1 \dots n} \mu_{A_i^k}(x_i)$$

### 3. Defuzzification (Weighted Average)
$$\hat{y} = \frac{\sum_{k=1}^{R} \alpha_k \cdot \bar{y}_k}{\sum_{k=1}^{R} \alpha_k}$$

---

## CLI & Interactive Exploration

To launch the built-in interactive inference prompt:

```rust
let activated_rules = model.prompt_infer(TNorm::Product)?;
```

```text
==================================================
                INPUT INFERENCE                   
==================================================

👉 Enter the value for the attribute "temperature": 22.5

👉 Enter the value for the attribute "humidity": 45.0
```

---

## Development & Testing

```bash
# Run unit and integration tests
cargo test --all-targets --all-features

# Run linter checks
cargo clippy --all-targets --all-features -- -D warnings

# Build documentation locally
cargo doc --no-deps --open

# Run the demo example
cargo run --example demo
```

---

## Demo

![Metabolic Syndrome CLI Demo](assets/demo.gif)

## License

* MIT license [LICENSE]

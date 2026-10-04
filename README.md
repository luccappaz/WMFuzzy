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
wm_fuzzy = "0.3.0"
polars = "0.55.2" # Optional feature
```

---

## Quickstart

```rust
use polars::prelude::*;
use wm_fuzzy::{wm_model::WMModel, TNorm};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Define model architecture using compact JSON
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

    // 2. Parse configuration and build membership functions in one call
    let mut model = WMModel::from_json(json_config)?;

    // 3. Load dataset using Polars
    let x_train = df![
        "temperature" => &[15.0, 50.0, 85.0],
        "humidity" => &[20.0, 50.0, 80.0],
    ]?;
    let y_train = Series::new("target".into(), &[0.1, 0.5, 0.9]);

    // 4. Train model with the Wang-Mendel rule generation algorithm
    let rules = model.generate_rules(&x_train, &y_train, None, TNorm::Product, 0.01, 0.1)?;
    println!("Generated {} fuzzy rules", rules.len());

    // 5. Run batch prediction on new data
    let x_test = df![
        "temperature" => &[20.0, 80.0],
        "humidity" => &[30.0, 75.0],
    ]?;

    let (predictions, binary_predictions) = 
        model.predict(&x_test, None, Some(0.5), 0.0, TNorm::Product)?;

    println!("Continuous predictions: {:?}", predictions);
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

![CLI Demo](assets/demo.gif)

## License

* MIT license [LICENSE]

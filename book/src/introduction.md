# WMFuzzy: Interactive Wang-Mendel Fuzzy Logic in Rust

The library `wm_fuzzy` is a high performance, written entirely in Rust. It is designed for
extracting fuzzy rules from data and performing fuzzy inference using the Wang Mendel (WM)
algorithm.

Built with data pipelines in mind, `wm_fuzzy` provides seamless integration with the `Polars`
environment, a compact JSON configuration API, and strict domain invariant enforcement.

## Key Features

- **Dual Initialization APIs**:
    - **Compact JSON Configuration**: Define partition strategies, domain knots, and linguistic labels
      in a single declarative JSON schema (`WMModel::from_json`).
    - **Fluent Builder Pattern**: Construct models programmatically with -- for example -- add_linear_strategy,
      add_custom_strategy, and explicit domain limit checking.
- **Strict Domain Validation**: Automatically enforces strictly sorted grid knots, minimum knot counts
  ($N \geq 2$), and geometry-aware granularity inference across membership function kinds (`MFKind`).
- **Polars First-Class Integration**: Directly compute fuzzy rules from Polars DataFrame inputs and
  perform batch vector predictions across test splits.
- **Robust Inference & Fallback**: 
    - Exact fuzzy firing strength evaluation via T-Norms (`Product` and `Minimum`).
    - Minimum Euclidean distance fallback mechanism when input vectors lie outside active fuzzy
      partition boundaries ($\mu = 0$), i.e, when no rule in activated.
- **Zero-Cost Modular Architecture**: Gated Polars dependencies via `default = ["polars"]`, allowing headless deployment in memory-constrained environments.

## The Workflow

WMFuzzy bridges raw tabular datasets and interpretable rule-based inference through a structured,
four-phase pipeline:

flowchart LR
    A["<b>1. Input Data</b><br/>Polars DataFrame"] --> B["<b>2. Model Setup</b><br/>JSON or Builder API"]
    B --> C["<b>3. Rule Extraction</b><br/>Wang-Mendel Algorithm"]
    C --> D["<b>4. Defuzzification</b><br/>Inference Output (ȳ)"]

Pipeline Steps

1. **Partition Setup & Grid Discretization**

Continuous numerical domains are partitioned into
fuzzy sets using either linearly spaced boundaries or empirical quantiles computed directly from
Polars DataFrames.

2. **Model Initialization & Invariant Checking**

Membership function geometry
(`MFKind::Triangular`, etc.) and linguistic partition counts are configured via a compact JSON schema
or the fluent Rust Builder API, enforcing domain invariants at build time.

3. **Wang-Mendel Rule Extraction**

The training dataset passes through fuzzification to form candidate IF-THEN rules. Each
candidate rule's Degree of Consistency ($DoC$) is computed to automatically resolve conflicts and
eliminate redundant antecedents.

4. **Inference & Defuzzification**

Incoming feature vectors trigger active fuzzy rules using specified T-Norm operators (`Product` or
`Minimum`). The system computes a crisp weighted average prediction ($\bar{y}$) or falls back to the
nearest rule via normalized Euclidean distance when firing strength is zero ($\mu = 0$).

## Where to go next

- **Quick Start**: Get a working pipeline running in less than 5 min.
- **The Wang-Mendel Algorithm**: Explore the mathematical foundation and rule extraction mechanics.
- **Builder API**: Learn more about how to work with the API and build your model.

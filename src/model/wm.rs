use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

use crate::{
    dataset::{TabularDataset, TargetVector},
    error::WMModelError,
    fuzzy::{
        ActivatedRule, Antecedents, FuzzyRule, TNorm, antecedent::calculate_firing_strength,
        granularity::Granularity,
    },
    membership::mb_function::{
        FuzzyStrategy, InnerStrategy, MFKind, MembershipFunction, MembershipOp, Partitionable,
    },
    metrics::{ClassificationMetrics, EvaluationMetrics, RegressionMetrics},
    model::config::ModelConfig,
    types::FuzzyValues,
};

/// Configures and executes fuzzy inference using the Wang-Mendel method.
///
/// # Example
///
/// ```rust
/// use wm_fuzzy::prelude::{WMModel, MFKind, TNorm};
/// use std::collections::HashMap;
///
/// let mut model = WMModel::new();
/// model.set_kind("triangular").unwrap();
/// model.add_custom_strategy("temp", vec![0.0, 50.0, 100.0]).unwrap();
/// model.build().unwrap();
///
/// let mut input = HashMap::new();
/// input.insert("temp".to_string(), 25.0);
///
/// let (y_pred, active_rules) = model.infer(&input, TNorm::Minimum);
/// assert!(y_pred.is_none()); // No rules generated yet
/// ```
#[derive(Debug, Default)]
pub struct WMModel {
    kind: Option<MFKind>,
    strategies: HashMap<String, FuzzyStrategy>,
    mf_values: Vec<HashMap<String, FuzzyValues>>,
    rules: Vec<FuzzyRule>,
    granularities: HashMap<String, Granularity>,

    // Configuration and mappings to the input attributes
    labels: HashMap<String, Vec<String>>,
    limits: HashMap<String, Vec<f64>>,
    pub(super) mf_configs: HashMap<String, HashMap<String, MembershipFunction>>,
    mf_centers: HashMap<String, HashMap<String, f64>>,

    // Configuration and mappings for the target attribute
    target_strategy: Option<FuzzyStrategy>,
    target_granularity: Option<Granularity>,
    target_labels: Vec<String>,
    pub(super) target_mfs: HashMap<String, MembershipFunction>,
    target_centers: HashMap<String, f64>,

    mf_configs_built: bool,
}

/// Build methods to setup the target
impl WMModel {
    /// Set strategy for the target value directly
    pub fn set_target_strategy(&mut self, strategy: FuzzyStrategy) -> &mut Self {
        self.target_strategy = Some(strategy);
        self.mf_configs_built = false;
        self
    }
    /// Add custom strategy for the target value
    pub fn add_target_custom_strategy<T: Into<Vec<f64>>>(
        &mut self,
        points: T,
    ) -> Result<&mut Self, WMModelError> {
        self.target_strategy = Some(FuzzyStrategy::custom(points)?);
        self.mf_configs_built = false;
        Ok(self)
    }

    /// Add linear strategy for the target value
    pub fn add_target_linear_strategy(
        &mut self,
        min: f64,
        max: f64,
    ) -> Result<&mut Self, WMModelError> {
        self.target_strategy = Some(FuzzyStrategy::linear(min, max)?);
        self.mf_configs_built = false;
        Ok(self)
    }

    /// Add the granularity for the target value
    pub fn add_target_granularity(
        &mut self,
        granularity: impl TryInto<Granularity, Error = WMModelError>,
    ) -> Result<&mut Self, WMModelError> {
        self.target_granularity = Some(granularity.try_into()?);
        self.mf_configs_built = false;
        Ok(self)
    }

    /// Set target granularity directly
    pub fn set_target_granularity(&mut self, granularity: Granularity) -> &mut Self {
        self.target_granularity = Some(granularity);
        self.mf_configs_built = false;
        self
    }

    /// Set custom labels for the fuzzy groups in the target value
    pub fn set_target_labels<S: Into<String>>(&mut self, labels: Vec<S>) -> &mut Self {
        self.target_labels = labels.into_iter().map(Into::into).collect();
        self.mf_configs_built = false;
        self
    }
}

/// Build methods to setup the input values
impl WMModel {
    /// Sets the membership function kind in place.
    pub fn set_kind(
        &mut self,
        kind: impl TryInto<MFKind, Error = WMModelError>,
    ) -> Result<&mut Self, WMModelError> {
        self.kind = Some(kind.try_into()?);
        self.mf_configs_built = false;
        Ok(self)
    }

    /// Registers custom linguistic labels for a specific feature.
    pub fn add_labels<S: Into<String>>(
        &mut self,
        feature: impl Into<String>,
        labels: Vec<S>,
    ) -> &mut Self {
        let converted_labels: Vec<String> = labels.into_iter().map(Into::into).collect();
        self.labels.insert(feature.into(), converted_labels);
        self.mf_configs_built = false;
        self
    }

    /// Registers a map of custom labels for all features at once.
    pub fn set_labels(&mut self, labels: HashMap<String, Vec<String>>) -> &mut Self {
        self.labels = labels;
        self.mf_configs_built = false;
        self
    }

    /// Sets a pre-constructed `FuzzyStrategy` globally in place.
    pub fn set_strategies(&mut self, strategies: HashMap<String, FuzzyStrategy>) -> &mut Self {
        self.strategies = strategies;
        self.mf_configs_built = false;
        self
    }

    /// Configures a linear partition strategy globally in place.
    pub fn add_linear_strategy(
        &mut self,
        feature: impl Into<String>,
        min: f64,
        max: f64,
    ) -> Result<&mut Self, WMModelError> {
        let feature_name = feature.into();
        self.strategies
            .insert(feature_name, FuzzyStrategy::linear(min, max)?);
        self.mf_configs_built = false;
        Ok(self)
    }

    /// Configures a custom partition strategy globally in place.
    pub fn add_custom_strategy<T: Into<Vec<f64>>>(
        &mut self,
        feature: impl Into<String>,
        points: T,
    ) -> Result<&mut Self, WMModelError> {
        let feature_name = feature.into();
        self.strategies
            .insert(feature_name, FuzzyStrategy::custom(points)?);

        self.mf_configs_built = false;
        Ok(self)
    }

    /// Sets the granularity map for all features at once in place.
    pub fn set_granularities(&mut self, granularities: HashMap<String, Granularity>) -> &mut Self {
        self.granularities = granularities;
        self.mf_configs_built = false;
        self
    }

    /// Adds a single feature's granularity in place.
    pub fn add_granularity(
        &mut self,
        feature: impl Into<String>,
        granularity: impl TryInto<Granularity, Error = WMModelError>,
    ) -> Result<&mut Self, WMModelError> {
        self.granularities
            .insert(feature.into(), granularity.try_into()?);
        self.mf_configs_built = false;
        Ok(self)
    }

    /// Sets feature limits in place.
    pub fn with_limits(&mut self, limits: HashMap<String, Vec<f64>>) -> &mut Self {
        self.limits = limits;
        self.mf_configs_built = false;
        self
    }

    /// Adds limits (grid knots) array for a specific feature after validating invariants against the configured MFKind.
    pub fn add_limit(
        &mut self,
        feature: impl Into<String>,
        limits: impl Into<Vec<f64>>,
    ) -> Result<&mut Self, WMModelError> {
        let feature_name = feature.into();
        let knots: Vec<f64> = limits.into();

        // Validate minimum grid knots invariant
        if knots.len() < 2 {
            return Err(WMModelError::InvalidStrategy(format!(
                "Limits array for feature '{}' must contain at least 2 points, got {}.",
                feature_name,
                knots.len()
            )));
        }

        // Validate strictly ascending order invariant
        if !knots.windows(2).all(|w| w[0] < w[1]) {
            return Err(WMModelError::InvalidStrategy(format!(
                "Limits array for feature '{}' must be strictly sorted in ascending order.",
                feature_name
            )));
        }

        // Validate against configured granularity if present; otherwise infer it dynamically from MFKind
        if let Some(&configured) = self.granularities.get(&feature_name) {
            if knots.len() != configured.count() {
                return Err(WMModelError::LimitsMismatch(
                    knots.len(),
                    feature_name,
                    configured.count(),
                ));
            }
        } else {
            let kind = self.kind.ok_or_else(|| {
            WMModelError::InvalidStrategy(
                "Membership function kind (MFKind) must be set via `set_kind` before adding limits.".into(),
            )
        })?;
            let inferred = kind.infer_granularity(knots.len())?;
            self.granularities.insert(feature_name.clone(), inferred);
        }

        self.limits.insert(feature_name, knots);
        self.mf_configs_built = false;
        Ok(self)
    }
}

/// Builder global methods
impl WMModel {
    /// Creates a new `WMModel` instance with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds and validates the model in-place, expanding partition strategies into grid knots and generating MFs.
    pub fn build(&mut self) -> Result<&mut Self, WMModelError> {
        if self.mf_configs_built {
            return Ok(self);
        }

        let kind = self.kind.ok_or_else(|| {
            WMModelError::InvalidStrategy(
                "Membership function kind (MFKind) was not specified.".into(),
            )
        })?;

        for (feature, strategy) in &self.strategies {
            if let InnerStrategy::Custom { knots } = strategy.inner()
                && !self.granularities.contains_key(feature)
            {
                let inferred = kind.infer_granularity(knots.len())?;
                self.granularities.insert(feature.clone(), inferred);
            }
        }

        self.compute_labels()?;

        // Setup the MF values for the input
        for (feature, strategy) in &self.strategies {
            let granularity = self.granularities.get(feature).copied().ok_or_else(|| {
                WMModelError::InvalidStrategy(format!(
                    "Feature '{}' has strategy configured but lacks granularity.",
                    feature
                ))
            })?;

            let partition =
                MembershipFunction::create_partition(kind, strategy.clone(), granularity);
            let labels = &self.labels[feature];

            let mut feature_mfs = HashMap::with_capacity(partition.len());
            let mut feature_centers = HashMap::with_capacity(partition.len());
            let mut feature_limits = Vec::with_capacity(partition.len());

            for (label, mf) in labels.iter().zip(partition) {
                let center = mf.center();
                feature_limits.push(center);
                feature_centers.insert(label.clone(), center);
                feature_mfs.insert(label.clone(), mf);
            }

            self.limits.insert(feature.clone(), feature_limits);
            self.mf_configs.insert(feature.clone(), feature_mfs);
            self.mf_centers.insert(feature.clone(), feature_centers);
        }

        // Setup the MF values for the target
        if let Some(target_strat) = &self.target_strategy {
            let target_gran = self.target_granularity.ok_or_else(|| {
                WMModelError::InvalidStrategy(
                    "Target Y strategy configured but lacks granularity.".into(),
                )
            })?;

            let target_partition =
                MembershipFunction::create_partition(kind, target_strat.clone(), target_gran);

            if self.target_labels.is_empty() {
                self.target_labels = target_gran.default_labels()?;
            }

            self.target_mfs.clear();
            self.target_centers.clear();

            for (label, mf) in self.target_labels.iter().zip(target_partition) {
                let center = mf.center();
                self.target_centers.insert(label.clone(), center);
                self.target_mfs.insert(label.clone(), mf);
            }
        }

        self.mf_configs_built = true;
        Ok(self)
    }

    /// Computes and validates feature labels. Uses custom labels if provided,
    /// otherwise falls back to the default labels derived from Granularity.
    fn compute_labels(&mut self) -> Result<(), WMModelError> {
        // Considering the input features
        for (feature, &granularity) in &self.granularities {
            let expected_count = granularity.count();

            if let Some(custom_labels) = self.labels.get(feature) {
                // Validate that the number of custom labels matches the expected granularity count
                if custom_labels.len() != expected_count {
                    return Err(WMModelError::InvalidStrategy(format!(
                        "Feature '{}' expects {} labels for granularity {:?}, but {} custom labels were provided: {:?}",
                        feature,
                        expected_count,
                        granularity,
                        custom_labels.len(),
                        custom_labels
                    )));
                }
            } else {
                // Fall back to default labels if no custom labels were specified for this feature
                self.labels
                    .insert(feature.clone(), granularity.default_labels()?);
            }
        }

        // Considering the target values
        if let Some(target_granularity) = self.target_granularity {
            let expected_count = target_granularity.count();

            if !self.target_labels.is_empty() {
                if self.target_labels.len() != expected_count {
                    return Err(WMModelError::InvalidStrategy(format!(
                        "Target variable 'y' expects {} labels for granularity {:?}, but {} custom labels were provided: {:?}",
                        expected_count,
                        target_granularity,
                        self.target_labels.len(),
                        self.target_labels
                    )));
                }
            } else {
                self.target_labels = target_granularity.default_labels()?;
            }
        }

        Ok(())
    }
}

// Prediction and inference
impl WMModel {
    /// Computes membership degrees ($\mu$) for all configured input features given a crisp sample row.
    fn get_row_memberships(
        &self,
        row: &HashMap<String, f64>,
    ) -> HashMap<String, HashMap<String, f64>> {
        let mut memberships = HashMap::with_capacity(self.mf_configs.len());

        for (feature, fuzzy_sets) in &self.mf_configs {
            if let Some(&value) = row.get(feature) {
                let mut group_mus = HashMap::with_capacity(fuzzy_sets.len());
                for (group_name, mf) in fuzzy_sets {
                    let mu = mf.eval(value);
                    group_mus.insert(group_name.clone(), mu);
                }
                memberships.insert(feature.clone(), group_mus);
            }
        }
        memberships
    }

    /// Performs batch prediction over a tabular dataset.
    ///
    /// Returns a tuple containing:
    /// - `Vec<f64>`: Continuous defuzzified predictions ($\bar{y}$).
    /// - `Vec<u32>`: Binary classification decisions based on the provided `threshold` (default `0.5`).
    pub fn predict<D: TabularDataset>(
        &mut self,
        x_test: &D,
        feature_names: Option<&[&str]>,
        threshold: Option<f64>,
        fallback: f64,
        t_norm: TNorm,
    ) -> Result<(Vec<f64>, Vec<u32>), WMModelError> {
        self.build()?;

        // Resolve feature names: use explicit argument or default to model keys
        let configured_features: Vec<String>;
        let feature_names: Vec<&str> = match feature_names {
            Some(names) => names.to_vec(),
            None => {
                configured_features = self.mf_configs.keys().cloned().collect();
                configured_features.iter().map(|s| s.as_str()).collect()
            }
        };

        let height = x_test.num_rows();
        let mut continuous_preds = Vec::with_capacity(height);
        let mut binary_preds = Vec::with_capacity(height);
        let threshold = threshold.unwrap_or(0.5);

        for row_idx in 0..height {
            let row_buffer = x_test.get_row_as_map(row_idx, &feature_names);
            let memberships = self.get_row_memberships(&row_buffer);

            let mut total_firing_strength = 0.0;
            let mut weighted_output_sum = 0.0;

            for rule in &self.rules {
                let firing_strength =
                    calculate_firing_strength(&rule.antecedents, &memberships, t_norm.clone());

                if firing_strength > 0.0 {
                    weighted_output_sum += firing_strength * rule.weighted_average;
                    total_firing_strength += firing_strength;
                }
            }

            // Defuzzify using Center of Gravity (weighted average) or fall back to nearest rule
            let y_pred = if total_firing_strength > 0.0 {
                weighted_output_sum / total_firing_strength
            } else if let (Some(nearest), _) = self.min_distance_rule(&row_buffer) {
                nearest.weighted_average
            } else {
                fallback
            };

            continuous_preds.push(y_pred);
            binary_preds.push(if y_pred >= threshold { 1 } else { 0 });
        }

        Ok((continuous_preds, binary_preds))
    }

    /// Finds the nearest rule in Euclidean input space when no rules fire for an input sample.
    pub fn min_distance_rule(
        &self,
        crisp_row: &HashMap<String, f64>,
    ) -> (Option<&FuzzyRule>, Option<f64>) {
        if self.rules.is_empty() {
            return (None, None);
        }

        let mut nearest_rule: Option<&FuzzyRule> = None;
        let mut min_dist = f64::INFINITY;

        for rule in &self.rules {
            if let Some(input_centers) = rule.antecedents.get_input_center(&self.mf_centers) {
                let mut sum_sq = 0.0;
                let mut matched_all = true;

                for (feature, value) in crisp_row {
                    if let Some(&c1) = input_centers.get(feature) {
                        sum_sq += (c1 - value).powi(2);
                    } else {
                        matched_all = false;
                        break;
                    }
                }

                if matched_all {
                    let dist = sum_sq.sqrt();
                    if dist < min_dist {
                        min_dist = dist;
                        nearest_rule = Some(rule);
                    }
                }
            }
        }

        (
            nearest_rule,
            if min_dist.is_infinite() {
                None
            } else {
                Some(min_dist)
            },
        )
    }

    /// Performs single-sample inference and returns activated rules sorted by firing strength.
    pub fn infer(
        &self,
        crisp_row: &HashMap<String, f64>,
        t_norm: TNorm,
    ) -> (Option<f64>, Vec<ActivatedRule<'_>>) {
        let memberships = self.get_row_memberships(crisp_row);
        let mut activated = Vec::new();
        let mut total_firing_strength = 0.0;
        let mut weighted_output_sum = 0.0;

        for rule in &self.rules {
            let firing_strength =
                calculate_firing_strength(&rule.antecedents, &memberships, t_norm.clone());

            if firing_strength > 0.0 {
                weighted_output_sum += firing_strength * rule.weighted_average;
                total_firing_strength += firing_strength;

                activated.push(ActivatedRule {
                    rule,
                    firing_strength,
                });
            }
        }

        // Sort activated rules in descending order of firing strength
        activated.sort_by(|a, b| {
            b.firing_strength
                .partial_cmp(&a.firing_strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut y_pred = None;

        if activated.is_empty() {
            println!("\n⚠️ No rules were activated for the given row:");
            println!("  {:?}", crisp_row);
        } else {
            println!("\n🔥 Activated Rules (Total: {}):", activated.len());
            for rule_act in &activated {
                println!("{}", rule_act.rule);
            }

            if total_firing_strength > 0.0 {
                let val = weighted_output_sum / total_firing_strength;
                y_pred = Some(val);
                println!("\n📊 [Defuzzification / Weighted Mean]:");
                println!("  ↳ y inferred: {:.4}", val);
                println!("  ↳ Combined firing strength: {:.4}", total_firing_strength);
            }
        }

        (y_pred, activated)
    }

    /// Resets the extracted rules and flags the model for rebuilding.
    pub fn clear_rules(&mut self) {
        self.rules.clear();
        self.mf_configs_built = false;
    }

    /// Returns the total number of extracted fuzzy rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Returns whether internal membership function structures have been built.
    pub fn is_configured(&self) -> bool {
        self.mf_configs_built
    }
}

// Rules generation
impl WMModel {
    /// Extracts data-driven unique candidate antecedents row-by-row from computed memberships.
    ///
    /// For each sample row, selects the dominant fuzzy label (maximum membership degree)
    /// per feature, deduplicating unique rule antecedents across the entire dataset.
    fn extract_antecedents(
        &self,
        row_memberships: &[HashMap<String, HashMap<String, f64>>],
        feature_names: &[&str],
    ) -> Vec<Antecedents> {
        let mut seen = HashSet::new();
        let mut candidate_antecedents = Vec::new();

        for row_mems in row_memberships {
            let mut antecedent_map = BTreeMap::new();

            for &feature in feature_names {
                if let Some(fuzzy_sets) = row_mems.get(feature)
                    && let Some((best_label, _)) = fuzzy_sets
                        .iter()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                {
                    antecedent_map.insert(feature.to_string(), best_label.clone());
                }
            }

            // Ensure all requested features were mapped successfully for this row
            if antecedent_map.len() == feature_names.len() {
                let antecedents = Antecedents {
                    data: antecedent_map,
                };
                if seen.insert(antecedents.clone()) {
                    candidate_antecedents.push(antecedents);
                }
            }
        }

        candidate_antecedents
    }

    /// Generates fuzzy rules using a unified data-driven Wang-Mendel extraction pipeline.
    ///
    /// Evaluates antecedent candidates ($A$) once, branching only at the consequent construction:
    /// - **Case 1 (Linguistic)**: Maps weighted average $\bar{y}$ to the dominant target fuzzy set $B^*$.
    /// - **Case 2 (Numeric)**: Employs continuous regression stats directly ($\bar{y}$ and $\sigma$).
    pub fn generate_rules<D: TabularDataset, Y: TargetVector>(
        &mut self,
        x_train: &D,
        y_train: &Y,
        feature_names: Option<&[&str]>,
        t_norm: TNorm,
        min_support: f64,
        min_confidence: f64,
    ) -> Result<Vec<FuzzyRule>, WMModelError> {
        self.build()?;

        let n_samples = x_train.num_rows();
        if n_samples == 0 || y_train.num_samples() == 0 {
            return Ok(Vec::new());
        }

        if n_samples != y_train.num_samples() {
            return Err(WMModelError::DatasetError(format!(
                "X and y length mismatch: {} vs {}",
                n_samples,
                y_train.num_samples()
            )));
        }

        // Determine feature names to process (user-specified or configured model keys)
        let default_names: Vec<String>;
        let feature_names: Vec<&str> = match feature_names {
            Some(names) => names.to_vec(),
            None => {
                default_names = self.mf_configs.keys().cloned().collect();
                default_names.iter().map(|s| s.as_str()).collect()
            }
        };

        // Extract continuous target values and target range
        let y_data = y_train.to_vec_f64()?;
        let (y_min, y_max) = y_data.iter().fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(min_v, max_v), &val| (min_v.min(val), max_v.max(val)),
        );
        let variance_range = (y_max - y_min).max(f64::EPSILON);

        // Step 1: Compute input membership degrees row-by-row
        let mut row_memberships = Vec::with_capacity(n_samples);
        for row_idx in 0..n_samples {
            let row_map = x_train.get_row_as_map(row_idx, &feature_names);
            row_memberships.push(self.get_row_memberships(&row_map));
        }

        // Step 2: Extract data-driven unique candidate antecedents (identical for both cases)
        let candidate_antecedents = self.extract_antecedents(&row_memberships, &feature_names);

        let mut generated_rules = Vec::new();

        // Step 3: Evaluate shared statistics and construct rules based on target configuration
        for antecedents in candidate_antecedents {
            let rule_strength: Vec<f64> = row_memberships
                .iter()
                .map(|mems| calculate_firing_strength(&antecedents, mems, t_norm.clone()))
                .collect();

            let sum_strength: f64 = rule_strength.iter().sum();
            if sum_strength <= 0.0 {
                continue;
            }

            let support = sum_strength / (n_samples as f64);
            if support < min_support {
                continue;
            }

            // Compute weighted average y_bar
            let weighted_sum: f64 = y_data
                .iter()
                .zip(&rule_strength)
                .map(|(&yi, &wi)| yi * wi)
                .sum();
            let weighted_avg = weighted_sum / sum_strength;

            // Compute weighted variance/dispersion
            let weighted_var_sum: f64 = y_data
                .iter()
                .zip(&rule_strength)
                .map(|(&yi, &wi)| (yi - weighted_avg).abs() * wi)
                .sum();
            let weighted_variance = weighted_var_sum / sum_strength;

            let base_doc = (1.0 - (weighted_variance / variance_range)).clamp(0.0, 1.0);

            if !self.target_mfs.is_empty() {
                // =========================================================================
                // CASE 1: Target Y has fuzzy partition groups (Linguistic Consequent)
                // =========================================================================
                let mut best_y_label = String::new();
                let mut max_y_mu = 0.0;

                // Fuzzify weighted_avg against target membership functions
                for (y_label, mf) in &self.target_mfs {
                    let mu = mf.eval(weighted_avg);
                    if mu > max_y_mu {
                        max_y_mu = mu;
                        best_y_label = y_label.clone();
                    }
                }

                if max_y_mu <= 0.0 {
                    continue;
                }

                let doc = base_doc * max_y_mu;
                if doc < min_confidence {
                    continue;
                }

                let target_center = *self.target_centers.get(&best_y_label).unwrap_or(&0.0);

                generated_rules.push(FuzzyRule::linguistic(
                    antecedents,
                    best_y_label,
                    target_center,
                    doc,
                    support,
                ));
            } else {
                // =========================================================================
                // CASE 2: Target Y is continuous/numeric without target partitions
                // =========================================================================
                let doc = base_doc;
                if doc < min_confidence {
                    continue;
                }

                generated_rules.push(FuzzyRule::numeric(
                    antecedents,
                    weighted_avg,
                    weighted_variance,
                    doc,
                    support,
                ));
            }
        }

        self.mf_values = row_memberships;
        self.rules = generated_rules.clone();
        Ok(generated_rules)
    }
}

// Persistence and evaluation
impl WMModel {
    /// Evaluates model performance on continuous regression targets.
    ///
    /// Computes Mean Squared Error (MSE), Root Mean Squared Error (RMSE),
    /// Mean Absolute Error (MAE), $R^2$ Score, and Mean Absolute Percentage Error (MAPE).
    pub fn evaluate_regression<D: TabularDataset, Y: TargetVector>(
        &mut self,
        x_test: &D,
        y_test: &Y,
        fallback: Option<f64>,
        output_path: Option<&Path>,
    ) -> Result<RegressionMetrics, WMModelError> {
        let (y_pred, _) =
            self.predict(x_test, None, None, fallback.unwrap_or(0.0), TNorm::Product)?;

        let y_true = y_test.to_vec_f64()?;
        let n = y_true.len() as f64;

        if n == 0.0 {
            return Ok(RegressionMetrics {
                mse: 0.0,
                rmse: 0.0,
                mae: 0.0,
                r2: 0.0,
                mape: 0.0,
            });
        }

        let y_true_mean = y_true.iter().sum::<f64>() / n;

        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;
        let mut mae_sum = 0.0;
        let mut mape_sum = 0.0;

        for (&yt, &yp) in y_true.iter().zip(y_pred.iter()) {
            let diff = yt - yp;
            ss_res += diff * diff;
            ss_tot += (yt - y_true_mean).powi(2);
            mae_sum += diff.abs();

            if yt.abs() > f64::EPSILON {
                mape_sum += (diff / yt).abs();
            }
        }

        let mse = ss_res / n;
        let rmse = mse.sqrt();
        let mae = mae_sum / n;
        let r2 = if ss_tot > f64::EPSILON {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };
        let mape = (mape_sum / n) * 100.0;

        let metrics = RegressionMetrics {
            mse,
            rmse,
            mae,
            r2,
            mape,
        };

        if let Some(path) = output_path {
            let file = File::create(path)?;
            serde_json::to_writer_pretty(file, &metrics)?;
            println!("✅ Regression metrics saved to: {:?}", path);
        }

        Ok(metrics)
    }

    /// Evaluates model performance on binary classification targets.
    pub fn evaluate_classification<D: TabularDataset, Y: TargetVector>(
        &mut self,
        x_test: &D,
        y_test: &Y,
        threshold: Option<f64>,
        fallback: Option<f64>,
        output_path: Option<&Path>,
    ) -> Result<ClassificationMetrics, WMModelError> {
        let threshold_val = threshold.unwrap_or(0.5);
        let (_, y_pred_bin) = self.predict(
            x_test,
            None,
            Some(threshold_val),
            fallback.unwrap_or(0.5),
            TNorm::Product,
        )?;

        let y_true = y_test.to_vec_f64()?;

        let mut tp = 0;
        let mut tn = 0;
        let mut fp = 0;
        let mut fn_val = 0;

        for (&yt_f, &yp) in y_true.iter().zip(y_pred_bin.iter()) {
            let yt = yt_f.round() as u32;
            match (yt, yp) {
                (1, 1) => tp += 1,
                (0, 0) => tn += 1,
                (0, 1) => fp += 1,
                (1, 0) => fn_val += 1,
                _ => {}
            }
        }

        let total = (tp + tn + fp + fn_val) as f64;
        let accuracy = if total > 0.0 {
            (tp + tn) as f64 / total
        } else {
            0.0
        };
        let precision = if (tp + fp) > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let recall = if (tp + fn_val) > 0 {
            tp as f64 / (tp + fn_val) as f64
        } else {
            0.0
        };
        let f1_score = if (precision + recall) > 0.0 {
            2.0 * (precision * recall) / (precision + recall)
        } else {
            0.0
        };

        let metrics = ClassificationMetrics {
            accuracy,
            precision,
            recall,
            f1_score,
            confusion_matrix: [[tn, fp], [fn_val, tp]],
        };

        if let Some(path) = output_path {
            let file = File::create(path)?;
            serde_json::to_writer_pretty(file, &metrics)?;
            println!("✅ Classification metrics saved to: {:?}", path);
        }

        Ok(metrics)
    }

    /// Smart evaluation entry point.
    ///
    /// Automatically performs classification evaluation if a `threshold` is provided,
    /// or regression evaluation if `threshold` is `None`.
    pub fn evaluate<D: TabularDataset, Y: TargetVector>(
        &mut self,
        x_test: &D,
        y_test: &Y,
        threshold: Option<f64>,
        fallback: Option<f64>,
        output_path: Option<&Path>,
    ) -> Result<EvaluationMetrics, WMModelError> {
        if let Some(th) = threshold {
            let metrics =
                self.evaluate_classification(x_test, y_test, Some(th), fallback, output_path)?;
            Ok(EvaluationMetrics::Classification(metrics))
        } else {
            let metrics = self.evaluate_regression(x_test, y_test, fallback, output_path)?;
            Ok(EvaluationMetrics::Regression(metrics))
        }
    }

    /// Serializes and saves trained rules to a JSON file.
    pub fn save_rules<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &self.rules).map_err(std::io::Error::other)?;
        Ok(())
    }

    /// Loads trained rules from a JSON file and resets MF configuration state.
    pub fn load_rules<P: AsRef<Path>>(&mut self, path: P) -> Result<(), std::io::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let rules: Vec<FuzzyRule> = serde_json::from_reader(reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        self.rules = rules;
        self.mf_configs_built = false;
        Ok(())
    }
}

// Getters and setters
impl WMModel {
    pub fn rules(&self) -> Result<&[FuzzyRule], WMModelError> {
        if self.rules.is_empty() {
            Err(WMModelError::NoRulesLoaded)
        } else {
            Ok(&self.rules)
        }
    }

    pub fn mf_values(&self) -> Result<&[HashMap<String, FuzzyValues>], WMModelError> {
        if self.mf_values.is_empty() {
            Err(WMModelError::NoMFValues)
        } else {
            Ok(&self.mf_values)
        }
    }

    pub fn target_mfs(&self) -> Result<&HashMap<String, MembershipFunction>, WMModelError> {
        if self.target_mfs.is_empty() {
            Err(WMModelError::TargetMFUnset)
        } else {
            Ok(&self.target_mfs)
        }
    }

    pub fn mf_configs(
        &self,
    ) -> Result<&HashMap<String, HashMap<String, MembershipFunction>>, WMModelError> {
        if self.mf_configs.is_empty() {
            Err(WMModelError::UnconfiguredError)
        } else {
            Ok(&self.mf_configs)
        }
    }

    /// Returns a reference to the configured feature limits map if available.
    pub fn limits(&self) -> Result<&HashMap<String, Vec<f64>>, WMModelError> {
        if self.limits.is_empty() {
            Err(WMModelError::InvalidStrategy(
                "No feature limits have been configured.".into(),
            ))
        } else {
            Ok(&self.limits)
        }
    }
}

impl WMModel {
    /// Instantiates an unbuilt `WMModel` initialized with settings from a `ModelConfig`.
    pub fn from_config(config: ModelConfig) -> Result<Self, WMModelError> {
        let mut model = WMModel::new();
        model.kind = Some(config.kind);

        let mut custom_labels = HashMap::new();

        // 1. Configure input features (X)
        for (feature, feat_cfg) in config.features {
            model.strategies.insert(feature.clone(), feat_cfg.strategy);

            if let Some(granularity) = feat_cfg.granularity {
                model.granularities.insert(feature.clone(), granularity);
            }

            if !feat_cfg.labels.is_empty() {
                custom_labels.insert(feature, feat_cfg.labels);
            }
        }

        if !custom_labels.is_empty() {
            model.set_labels(custom_labels);
        }

        // 2. Configure target variable (Y) if provided
        if let Some(target_cfg) = config.target {
            model.set_target_strategy(target_cfg.strategy);

            if let Some(granularity) = target_cfg.granularity {
                model.set_target_granularity(granularity);
            }

            if !target_cfg.labels.is_empty() {
                model.set_target_labels(target_cfg.labels);
            }
        }

        Ok(model)
    }

    /// Instantiates and builds a `WMModel` directly from a JSON configuration string.
    pub fn from_json(json_str: &str) -> Result<Self, WMModelError> {
        let config = ModelConfig::from_json(json_str)?;
        let mut model = Self::from_config(config)?;
        model.build()?;
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::membership::mb_function::MembershipOp;
    use crate::types::HashDataset;
    use std::collections::HashMap;

    /// Helper function to construct a `HashDataset` for tests.
    fn create_test_dataset(x1_values: &[f64]) -> HashDataset {
        x1_values
            .iter()
            .map(|&val| {
                let mut row = HashMap::new();
                row.insert("x1".to_string(), val);
                row
            })
            .collect()
    }

    #[test]
    fn test_builder_missing_kind_error() {
        let mut model = WMModel::new();
        model.add_granularity("x1", 3).unwrap();
        model.add_limit("x1", vec![0.0, 5.0, 10.0]).unwrap();

        // Error expected because MFKind was not specified
        let result = model.build();
        assert!(matches!(result, Err(WMModelError::InvalidStrategy(_))));
    }

    #[test]
    fn test_builder_limits_granularity_mismatch() {
        let mut model = WMModel::new();
        model.set_kind("triangular").unwrap();
        model.add_granularity("x1", 3).unwrap(); // Expects 3 points
        let result = model.add_limit("x1", vec![0.0, 10.0]); // Provides only 2 points

        assert!(matches!(result, Err(WMModelError::LimitsMismatch(2, _, 3))));
    }

    #[test]
    fn test_successful_build_with_linear_strategy() {
        let mut model = WMModel::new();
        model
            .set_kind("triangular")
            .unwrap()
            .add_granularity("x1", 3)
            .unwrap()
            .add_linear_strategy("x1", 0.0, 10.0)
            .unwrap();

        assert!(model.build().is_ok());
        assert!(model.is_configured());
        assert_eq!(
            model.limits().unwrap().get("x1").unwrap(),
            &[0.0, 5.0, 10.0]
        );
    }

    #[test]
    fn test_target_configuration_builder_methods() {
        let mut model = WMModel::new();
        model
            .set_kind("triangular")
            .unwrap()
            .add_granularity("x1", 3)
            .unwrap()
            .add_linear_strategy("x1", 0.0, 10.0)
            .unwrap()
            .add_target_granularity(3)
            .unwrap()
            .add_target_linear_strategy(0.0, 100.0)
            .unwrap()
            .set_target_labels(vec![
                "low".to_string(),
                "medium".to_string(),
                "high".to_string(),
            ]);

        assert!(model.build().is_ok());
        assert_eq!(model.target_mfs.len(), 3);
        assert!(model.target_mfs.contains_key("low"));
        assert!(model.target_mfs.contains_key("medium"));
        assert!(model.target_mfs.contains_key("high"));

        // Verify target center calculation using MembershipOp trait
        let medium_center = model.target_mfs.get("medium").unwrap().center();
        assert!((medium_center - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_generate_rules_case1_linguistic_target() {
        let mut model = WMModel::new();
        model
            .set_kind("triangular")
            .unwrap()
            .add_granularity("x1", 3)
            .unwrap()
            .add_linear_strategy("x1", 0.0, 10.0)
            .unwrap()
            .add_target_granularity(3)
            .unwrap()
            .add_target_linear_strategy(0.0, 100.0)
            .unwrap();

        let x_train: HashDataset = create_test_dataset(&[0.0, 5.0, 10.0]);
        let y_train: Vec<f64> = vec![0.0, 50.0, 100.0];

        let rules = model
            .generate_rules(&x_train, &y_train, None, TNorm::Product, 0.0, 0.0)
            .unwrap();

        assert!(!rules.is_empty());
        assert_eq!(model.rule_count(), rules.len());
        assert!(!model.target_mfs.is_empty());
    }

    #[test]
    fn test_generate_rules_case2_numeric_target() {
        let mut model = WMModel::new();
        model
            .set_kind("triangular")
            .unwrap()
            .add_granularity("x1", 3)
            .unwrap()
            .add_linear_strategy("x1", 0.0, 10.0)
            .unwrap();

        let x_train: HashDataset = create_test_dataset(&[0.0, 5.0, 10.0]);
        let y_train: Vec<f64> = vec![1.2, 5.4, 9.8];

        let rules = model
            .generate_rules(&x_train, &y_train, None, TNorm::Product, 0.0, 0.0)
            .unwrap();

        assert!(!rules.is_empty());
        assert!(model.target_mfs.is_empty()); // Case 2 has no target partitions
    }

    #[test]
    fn test_predict_and_infer() {
        let mut model = WMModel::new();
        model
            .set_kind("triangular")
            .unwrap()
            .add_granularity("x1", 3)
            .unwrap()
            .add_linear_strategy("x1", 0.0, 10.0)
            .unwrap();

        let x_train: HashDataset = create_test_dataset(&[0.0, 5.0, 10.0]);
        let y_train: Vec<f64> = vec![0.0, 0.5, 1.0];

        model
            .generate_rules(&x_train, &y_train, None, TNorm::Product, 0.0, 0.0)
            .unwrap();

        // Test batch prediction
        let (continuous_preds, binary_preds) = model
            .predict(&x_train, None, Some(0.5), 0.0, TNorm::Product)
            .unwrap();

        assert_eq!(continuous_preds.len(), 3);
        assert_eq!(binary_preds.len(), 3);

        // Test single sample inference
        let mut sample_row = HashMap::new();
        sample_row.insert("x1".to_string(), 5.0);
        let (y_pred, activated) = model.infer(&sample_row, TNorm::Product);

        assert!(y_pred.is_some());
        assert!(!activated.is_empty());
    }

    #[test]
    fn test_evaluate_regression_and_classification() {
        let mut model = WMModel::new();
        model
            .set_kind("triangular")
            .unwrap()
            .add_granularity("x1", 3)
            .unwrap()
            .add_linear_strategy("x1", 0.0, 10.0)
            .unwrap();

        let x_test: HashDataset = create_test_dataset(&[0.0, 5.0, 10.0]);
        let y_test: Vec<f64> = vec![0.0, 0.5, 1.0];

        model
            .generate_rules(&x_test, &y_test, None, TNorm::Product, 0.0, 0.0)
            .unwrap();

        // Regression evaluation (threshold = None)
        let reg_eval = model.evaluate(&x_test, &y_test, None, None, None).unwrap();
        if let EvaluationMetrics::Regression(metrics) = reg_eval {
            assert!(metrics.mse >= 0.0);
            assert!(metrics.mae >= 0.0);
        } else {
            panic!("Expected Regression metrics variant");
        }

        // Classification evaluation (threshold = Some(0.5))
        let clf_eval = model
            .evaluate(&x_test, &y_test, Some(0.5), None, None)
            .unwrap();
        if let EvaluationMetrics::Classification(metrics) = clf_eval {
            assert!(metrics.accuracy >= 0.0 && metrics.accuracy <= 1.0);
        } else {
            panic!("Expected Classification metrics variant");
        }
    }
}

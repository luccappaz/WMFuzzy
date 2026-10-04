use std::{
    collections::{HashMap, HashSet},
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
        InnerStrategy, MFKind, MembershipFunction, MembershipOp, Partitionable,
    },
    metrics::EvaluationMetrics,
    model::config::ModelConfig,
    types::FuzzyValues,
};

use crate::membership::mb_function::PartitionStrategy as FuzzyStrategy;

/// Core Wang-Mendel Fuzzy Model structure.
#[derive(Debug, Default)]
pub struct WMModel {
    kind: Option<MFKind>,
    strategies: HashMap<String, FuzzyStrategy>,
    mf_values: Vec<HashMap<String, FuzzyValues>>,
    rules: Vec<FuzzyRule>,
    granularities: HashMap<String, Granularity>,

    // Cached configurations and auxiliary feature mappings
    labels: HashMap<String, Vec<String>>,
    limits: HashMap<String, Vec<f64>>,
    pub(super) mf_configs: HashMap<String, HashMap<String, MembershipFunction>>,
    mf_centers: HashMap<String, HashMap<String, f64>>,
    mf_configs_built: bool,
}

/// Methods for the builder
impl WMModel {
    /// Creates a new `WMModel` instance with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

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

    /// Sets a pre-constructed `PartitionStrategy` globally in place.
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

        self.mf_configs_built = true;
        Ok(self)
    }

    /// Computes and validates feature labels. Uses custom labels if provided,
    /// otherwise falls back to the default labels derived from Granularity.
    fn compute_labels(&mut self) -> Result<(), WMModelError> {
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
                    .insert(feature.clone(), granularity.default_labels());
            }
        }

        Ok(())
    }
}

// Prediction and inference
impl WMModel {
    fn get_row_memberships(
        &self,
        row: &HashMap<String, f64>,
    ) -> HashMap<String, HashMap<String, f64>> {
        let mut memberships = HashMap::with_capacity(self.mf_configs.len());

        for (feature, fuzzy_sets) in &self.mf_configs {
            if let Some(&value) = row.get(feature) {
                let mut group_mus = HashMap::with_capacity(fuzzy_sets.len());
                for (group_name, mf) in fuzzy_sets {
                    // Chamada totalmente polimórfica sem saber os parâmetros internos da função
                    let mu = mf.eval(value);
                    group_mus.insert(group_name.clone(), mu);
                }
                memberships.insert(feature.clone(), group_mus);
            }
        }
        memberships
    }

    pub fn predict<D: TabularDataset>(
        &mut self,
        x_test: &D,
        feature_names: Option<&[&str]>,
        threshold: Option<f64>,
        fallback: f64,
        t_norm: TNorm,
    ) -> Result<(Vec<f64>, Vec<u32>), WMModelError> {
        self.build()?;

        // Determina os nomes das features: usa os passados via argumento ou usa as chaves do modelo
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
            // Obtém a linha como HashMap desacoplado via TabularDataset
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

            let y_pred = if total_firing_strength > 0.0 {
                weighted_output_sum / total_firing_strength
            } else {
                if let (Some(nearest), _) = self.min_distance_rule(&row_buffer) {
                    nearest.weighted_average
                } else {
                    fallback
                }
            };

            continuous_preds.push(y_pred);
            binary_preds.push(if y_pred >= threshold { 1 } else { 0 });
        }

        Ok((continuous_preds, binary_preds))
    }

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
                let mut sum_sq: f64 = 0.0;
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

        // Order activated rules by decreasing firing strength
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
                println!("{}", rule_act.rule)
            }

            if total_firing_strength > 0.0 {
                y_pred = Some(weighted_output_sum / total_firing_strength);
                println!("\n📊 [Defuzzification / Weighted Mean]:");
                println!("  ↳ y inferred: {:.4}", y_pred.unwrap());
                println!("  ↳ Total strength combined: {:.4}", total_firing_strength);
            }
        }

        (y_pred, activated)
    }

    pub fn clear_rules(&mut self) {
        self.rules.clear();
        self.mf_configs_built = false;
    }

    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    pub fn is_configured(&self) -> bool {
        self.mf_configs_built
    }
}

// Rules generation
impl WMModel {
    /// Extracts data-driven candidate antecedents row-by-row from computed memberships.
    ///
    /// For each sample row, it selects the fuzzy set label with the maximum
    /// degree of membership for each feature, avoiding combinatorial explosion.
    fn extract_antecedents(
        &self,
        row_memberships: &[HashMap<String, HashMap<String, f64>>],
        feature_names: &[&str],
    ) -> Vec<Antecedents> {
        let mut seen = HashSet::new();
        let mut candidate_antecedents = Vec::new();

        for row_mems in row_memberships {
            let mut antecedent_data = HashMap::with_capacity(feature_names.len());

            for &feature in feature_names {
                if let Some(fuzzy_sets) = row_mems.get(feature) {
                    // Find the fuzzy set label with the maximum degree of membership for this feature
                    let best_label = fuzzy_sets
                        .iter()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(label, _mu)| label.clone());

                    if let Some(label) = best_label {
                        antecedent_data.insert(feature.to_string(), label);
                    }
                }
            }

            // Ensure all requested features were mapped successfully for this row
            if antecedent_data.len() == feature_names.len() {
                // Build a canonical key (sorted key-value tuples) for deduplication
                let mut key: Vec<(String, String)> = antecedent_data.into_iter().collect();
                key.sort_unstable_by(|a, b| a.0.cmp(&b.0));

                if seen.insert(key.clone()) {
                    candidate_antecedents.push(Antecedents {
                        data: key.into_iter().collect(),
                    });
                }
            }
        }

        candidate_antecedents
    }

    /// Generates fuzzy rules using a data-driven Wang-Mendel extraction pipeline.
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

        // Extract continuous target values
        let y_data = y_train.to_vec_f64()?;

        // Calculate maximum target variation range
        let (y_min, y_max) = y_data.iter().fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(min_v, max_v), &val| (min_v.min(val), max_v.max(val)),
        );
        let variance_range = (y_max - y_min).max(f64::EPSILON);

        // Step 1: Compute membership degrees row-by-row
        let mut row_memberships = Vec::with_capacity(n_samples);
        for row_idx in 0..n_samples {
            let row_map = x_train.get_row_as_map(row_idx, &feature_names);
            row_memberships.push(self.get_row_memberships(&row_map));
        }

        // Step 2: Extract data-driven unique candidate antecedents directly from row memberships
        let candidate_antecedents = self.extract_antecedents(&row_memberships, &feature_names);

        let mut generated_rules = Vec::new();

        // Step 3: Evaluate firing strength, support, weighted variance, and confidence
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

            let weighted_sum: f64 = y_data
                .iter()
                .zip(&rule_strength)
                .map(|(&yi, &wi)| yi * wi)
                .sum();
            let weighted_avg = weighted_sum / sum_strength;

            let weighted_var_sum: f64 = y_data
                .iter()
                .zip(&rule_strength)
                .map(|(&yi, &wi)| (yi - weighted_avg).abs() * wi)
                .sum();
            let weighted_variance = weighted_var_sum / sum_strength;

            let doc = (1.0 - (weighted_variance / variance_range)).clamp(0.0, 1.0);
            if doc < min_confidence {
                continue;
            }

            generated_rules.push(FuzzyRule {
                antecedents,
                support,
                weighted_average: weighted_avg,
                weighted_variance,
                doc,
            });
        }

        self.mf_values = row_memberships;
        self.rules = generated_rules.clone();
        Ok(generated_rules)
    }
}

// Persistence and evaluation
impl WMModel {
    /// Evaluates model performance on test data using binary classification metrics.
    pub fn evaluate<D: TabularDataset, Y: TargetVector>(
        &mut self,
        x_test: &D,
        y_test: &Y,
        fallback: Option<f64>,
        output_path: Option<&Path>,
    ) -> Result<EvaluationMetrics, WMModelError> {
        let (_, y_pred_bin) = self.predict(
            x_test,
            None,
            Some(0.5),
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

        let metrics = EvaluationMetrics {
            accuracy,
            precision,
            recall,
            f1_score,
            confusion_matrix: [[tn, fp], [fn_val, tp]],
        };

        if let Some(path) = output_path {
            let file = File::create(path)?;
            serde_json::to_writer_pretty(file, &metrics)?;
            println!("✅ Metrics saved in: {:?}", path);
        }

        Ok(metrics)
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
}

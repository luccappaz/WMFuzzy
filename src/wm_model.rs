use plotters::prelude::*;
use polars::{chunked_array::ops::ChunkAgg, *};
use std::{
    collections::HashMap,
    fs::{self, File},
    io::{BufReader, BufWriter, Write},
    path::Path,
};

use polars::frame::DataFrame;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{ActivatedRule, Antecedents, EvaluationMetrics, FuzzyRule, Granularity, TNorm};

/// Calculate the rules fire strength
pub fn calculate_firing_strength(
    antecedents: &Antecedents,
    // Map storing the membership values for a row in dataset
    memberships: &HashMap<String, HashMap<String, f64>>,
    t_norm: TNorm,
) -> f64 {
    match t_norm {
        TNorm::Product => {
            let mut strength = 1.0;
            for (feature, group) in &antecedents.data {
                let mu = memberships
                    .get(feature)
                    .and_then(|groups| groups.get(group))
                    .copied()
                    .unwrap_or(0.0);
                strength *= mu;
                if strength == 0.0 {
                    break;
                }
            }
            strength
        }
        TNorm::Min => {
            let mut min_mu = 1.0;
            for (feature, group) in &antecedents.data {
                let mu = memberships
                    .get(feature)
                    .and_then(|groups| groups.get(group))
                    .copied()
                    .unwrap_or(0.0);
                if mu < min_mu {
                    min_mu = mu;
                }
                if min_mu == 0.0 {
                    break;
                }
            }
            min_mu
        }
    }
}

/// Plot the membership functions for each feature
pub fn plot_all_membership_functions(
    limits: &HashMap<String, Vec<f64>>,
    output_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_path = Path::new(output_dir);
    fs::create_dir_all(output_path)?;

    let colors = [&RED, &BLUE, &GREEN, &MAGENTA, &CYAN, &YELLOW];

    for (feature_name, lims) in limits {
        if lims.len() < 2 {
            continue;
        }

        let file_path = output_path.join(format!("mf_{}.png", feature_name));
        let root = BitMapBackend::new(&file_path, (800, 500)).into_drawing_area();
        root.fill(&WHITE)?;

        let min_x = lims.first().copied().unwrap_or(0.0);
        let max_x = lims.last().copied().unwrap_or(1.0);
        let span = if (max_x - min_x).abs() < f64::EPSILON {
            1.0
        } else {
            max_x - min_x
        };
        let padding = span * 0.05;

        let x_start = min_x - padding;
        let x_end = max_x + padding;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                format!("Membership Functions - {}", feature_name),
                ("sans-serif", 20).into_font(),
            )
            .margin(15)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(x_start..x_end, 0.0..1.05)?;

        chart
            .configure_mesh()
            .x_desc(feature_name)
            .y_desc("Membership (μ)")
            .draw()?;

        let num_mfs = lims.len();
        let steps = 300;

        for i in 0..num_mfs {
            let color = colors[i % colors.len()];
            let label = format!("MF_{}", i + 1);

            // Extract control points [a, b, c] for current membership function
            let a = if i == 0 { lims[0] } else { lims[i - 1] };
            let b = lims[i];
            let c = if i == num_mfs - 1 {
                lims[num_mfs - 1]
            } else {
                lims[i + 1]
            };

            let points: Vec<(f64, f64)> = (0..=steps)
                .map(|s| {
                    let x = x_start + (s as f64 / steps as f64) * (x_end - x_start);
                    let y = mb_function(x, [a, b, c]);
                    (x, y)
                })
                .collect();

            chart
                .draw_series(LineSeries::new(points, color.stroke_width(2)))?
                .label(label)
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
        }

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;

        root.present()?;
        println!("📊 Plot saved: {}", file_path.display());
    }

    Ok(())
}

pub fn mb_function(x: f64, [a, b, c]: [f64; 3]) -> f64 {
    if (a == b && x <= b) || (b == c && x >= b) || x == b {
        return 1.0;
    }
    if a != b && x > a && x < b {
        return (x - a) / (b - a);
    }
    if b != c && x > b && x < c {
        return (c - x) / (c - b);
    }
    0.0
}

/// Errors produced by the `WMModel` pipeline.
#[derive(Error, Debug)]
pub enum WMModelError {
    #[error("No membership values computed, generate rules first.")]
    NoMFValues,
    #[error("No rules loaded! Generate rules or load them from a file.")]
    NoRulesLoaded,
    #[error(
        "Granularity missing or unconfigured for feature: {0}. Need to call set_granularity first"
    )]
    MissingGranularity(String),
    #[error("Limits count ({0}) for feature '{1}' does not match granularity count ({2}).")]
    LimitsMismatch(usize, String, usize),
    #[error("Column '{0}' not found in DataFrame.")]
    ColumnNotFound(String),
    #[error("Expected Float64 column for '{0}'.")]
    ExpectedFloat64(String),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Polars error: {0}")]
    Polars(#[from] polars::error::PolarsError),
}

/// Main Wang-Mendel model engine with lazy initialization.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WMModel {
    granularity: HashMap<String, Granularity>,
    labels: HashMap<String, Vec<String>>,
    limits: HashMap<String, Vec<f64>>,
    mf_centers: HashMap<String, HashMap<String, f64>>,
    rules: Vec<FuzzyRule>,
    /// Store the membership values by row
    mf_values: Vec<HashMap<String, HashMap<String, f64>>>,
    /// Cache to store the triangular configs of the membership functions
    mf_configs: HashMap<String, HashMap<String, [f64; 3]>>,
    /// Track whether membership configs have been computed
    #[serde(skip)]
    mf_configs_built: bool,
}

type FuzzyValues = HashMap<String, f64>;

impl WMModel {
    pub fn build(
        &mut self,
        granularity: HashMap<String, Granularity>,
        limits: HashMap<String, Vec<f64>>,
    ) -> Result<(), WMModelError> {
        self.granularity = granularity;
        self.limits = limits;
        self.mf_configs_built = false; // Invalidate cached config
        self.compute_labels();
        self.build_mf_config()
    }

    pub fn get_rules(self) -> Result<Vec<FuzzyRule>, WMModelError> {
        if self.rules.is_empty() {
            Err(WMModelError::NoRulesLoaded)
        } else {
            Ok(self.rules)
        }
    }
    /// Rows at dim 0 and columns at 1
    pub fn get_mf_values(self) -> Result<Vec<HashMap<String, FuzzyValues>>, WMModelError> {
        if self.mf_values.is_empty() {
            Err(WMModelError::NoMFValues)
        } else {
            Ok(self.mf_values)
        }
    }

    fn compute_labels(&mut self) {
        for (feature, &gran) in &self.granularity {
            self.labels
                .entry(feature.clone())
                .or_insert_with(|| gran.default_labels());
        }
    }

    /// Auxiliary function
    fn get_abc(i: usize, granularity: Granularity, limits: &[f64]) -> [f64; 3] {
        let n = granularity.count();
        let prev = if i == 0 { limits[0] } else { limits[i - 1] };
        let curr = limits[i];
        let next = if i == n - 1 { limits[i] } else { limits[i + 1] };
        [prev, curr, next]
    }

    fn build_mf_config(&mut self) -> Result<(), WMModelError> {
        // Skip if already built to avoid redundant computation
        if self.mf_configs_built {
            return Ok(());
        }

        self.compute_labels();

        for (feature, limits) in &self.limits {
            let granularity = self
                .granularity
                .get(feature)
                .copied()
                .ok_or_else(|| WMModelError::MissingGranularity(feature.clone()))?;

            let gran_count = granularity.count();
            if limits.len() != gran_count {
                return Err(WMModelError::LimitsMismatch(
                    limits.len(),
                    feature.clone(),
                    gran_count,
                ));
            }

            let labels = &self.labels[feature];

            let mut feature_mfs = HashMap::with_capacity(gran_count);
            let mut feature_centers = HashMap::with_capacity(gran_count);

            for (i, label) in labels.iter().enumerate().take(gran_count) {
                let group_name = label.clone();
                let abc = Self::get_abc(i, granularity, limits);
                feature_centers.insert(group_name.clone(), abc[1]);
                feature_mfs.insert(group_name, abc);
            }

            self.mf_configs.insert(feature.clone(), feature_mfs);
            self.mf_centers.insert(feature.clone(), feature_centers);
        }

        self.mf_configs_built = true;
        Ok(())
    }

    fn get_row_memberships(
        &self,
        row: &HashMap<String, f64>,
    ) -> HashMap<String, HashMap<String, f64>> {
        let mut memberships = HashMap::with_capacity(self.mf_configs.len());

        for (feature, fuzzy_sets) in &self.mf_configs {
            if let Some(&value) = row.get(feature) {
                let mut group_mus = HashMap::with_capacity(fuzzy_sets.len());
                for (group_name, &abc) in fuzzy_sets {
                    let mu = mb_function(value, abc);
                    group_mus.insert(group_name.clone(), mu);
                }
                memberships.insert(feature.clone(), group_mus);
            }
        }
        memberships
    }

    fn extract_antecedents(&self, feature_names: &[String]) -> Vec<Antecedents> {
        let mut combinations: Vec<HashMap<String, String>> = vec![HashMap::new()];

        for feature in feature_names {
            let labels = match self.labels.get(feature) {
                Some(l) if !l.is_empty() => l,
                _ => continue,
            };

            let mut next_combinations = Vec::with_capacity(combinations.len() * labels.len());
            for current_map in &combinations {
                for label in labels {
                    let mut updated_map = current_map.clone();
                    updated_map.insert(feature.clone(), label.clone());
                    next_combinations.push(updated_map);
                }
            }
            combinations = next_combinations;
        }

        combinations
            .into_iter()
            .map(|data| Antecedents { data })
            .collect()
    }

    pub fn generate_rules(
        &mut self,
        x_train: &DataFrame,
        y_train: &series::Series,
        feature_names: Option<&[&str]>,
        t_norm: TNorm,
        min_support: f64,
        min_confidence: f64,
    ) -> Result<Vec<FuzzyRule>, WMModelError> {
        self.build_mf_config()?;

        let n_samples = x_train.height();
        if n_samples == 0 {
            return Ok(Vec::new());
        }

        let default_names: Vec<&str>;
        let feature_names: &[&str] = match feature_names {
            Some(names) => names,
            None => {
                default_names = x_train
                    .get_column_names()
                    .iter()
                    .map(|s| s.as_str())
                    .collect();
                &default_names
            }
        };

        let x_columns: Vec<(&str, &datatypes::Float64Chunked)> = feature_names
            .iter()
            .map(|&name| {
                let ca = x_train.column(name)?.f64()?;
                Ok((name, ca))
            })
            .collect::<Result<_, WMModelError>>()?;

        let y_ca = y_train.f64()?;
        let y_min = y_ca.min().unwrap_or(f64::NEG_INFINITY);
        let y_max = y_ca.max().unwrap_or(f64::INFINITY);
        let variance_range = (y_max - y_min).max(f64::EPSILON);

        let y_data: Vec<f64> = y_ca.iter().map(|opt| opt.unwrap_or(0.0)).collect();

        let mut row_buffer = HashMap::with_capacity(x_columns.len());
        let mut row_memberships = Vec::with_capacity(n_samples);

        for row_idx in 0..n_samples {
            row_buffer.clear();
            for &(feat_name, ca) in &x_columns {
                let val = ca.get(row_idx).unwrap_or(0.0);
                row_buffer.insert(feat_name.to_string(), val);
            }
            row_memberships.push(self.get_row_memberships(&row_buffer));
        }

        let feature_names_vec: Vec<String> = feature_names.iter().map(|s| s.to_string()).collect();
        let candidate_antecedents = self.extract_antecedents(&feature_names_vec);

        let mut generated_rules = Vec::new();

        for antecedents in candidate_antecedents {
            let rule_strength: Vec<f64> = row_memberships
                .iter()
                .map(|mems| calculate_firing_strength(&antecedents, mems, t_norm))
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
            let input_center = rule.antecedents.get_input_center(&self.mf_centers);

            let mut sum_sq = 0.0;
            for (feature, value) in crisp_row.iter() {
                if let Some(c1) = input_center.get(feature) {
                    sum_sq += (c1 - value).powi(2);
                } else {
                    break;
                }
            }
            let dist = sum_sq.sqrt();
            if dist < min_dist {
                min_dist = dist;
                nearest_rule = Some(rule);
            }
        }
        (nearest_rule, Some(min_dist))
    }

    pub fn predict(
        &mut self,
        x_test: &DataFrame,
        feature_names: Option<&[&str]>,
        threshold: Option<f64>,
        fallback: f64,
        t_norm: TNorm,
    ) -> Result<(Vec<f64>, Vec<u32>), WMModelError> {
        self.build_mf_config()?;

        let feature_names = match feature_names {
            Some(value) => value,
            None => {
                // Calculate the feature names if not given
                &x_test
                    .get_column_names()
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<&str>>()
            }
        };

        let columns: Vec<(&str, &datatypes::Float64Chunked)> = feature_names
            .iter()
            .map(|&feat| {
                let ca = x_test.column(feat)?.f64()?;
                Ok((feat, ca))
            })
            .collect::<Result<Vec<_>, WMModelError>>()?;

        let height = x_test.height();
        let mut continuous_preds = Vec::with_capacity(height);
        let mut binary_preds = Vec::with_capacity(height);
        let threshold = threshold.unwrap_or(0.5);

        let mut row_buffer = HashMap::with_capacity(columns.len());

        for row_idx in 0..height {
            // Clear after every prediction
            row_buffer.clear();
            for (feat, ca) in &columns {
                let val = ca.get(row_idx).unwrap_or(0.0);
                row_buffer.insert(feat.to_string(), val);
            }

            let memberships = self.get_row_memberships(&row_buffer);
            let mut total_firing_strength = 0.0;
            let mut weighted_output_sum = 0.0;

            for rule in &self.rules {
                let firing_strength =
                    calculate_firing_strength(&rule.antecedents, &memberships, t_norm);

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

    pub fn evaluate(
        &mut self,
        x_test: &DataFrame,
        y_test: &[u32],
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

        let mut tp = 0;
        let mut tn = 0;
        let mut fp = 0;
        let mut fn_val = 0;

        for (&yt, &yp) in y_test.iter().zip(y_pred_bin.iter()) {
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

    pub fn save_rules<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &self.rules).map_err(std::io::Error::other)?;
        Ok(())
    }

    pub fn load_rules<P: AsRef<Path>>(&mut self, path: P) -> Result<(), std::io::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let rules: Vec<FuzzyRule> = serde_json::from_reader(reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        self.rules = rules;
        // Reset cache since rules changed
        self.mf_configs_built = false;
        Ok(())
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
                calculate_firing_strength(&rule.antecedents, &memberships, t_norm);

            if firing_strength > 0.0 {
                weighted_output_sum += firing_strength * rule.weighted_average;
                total_firing_strength += firing_strength;

                activated.push(ActivatedRule {
                    rule,
                    firing_strength,
                });
            }
        }

        // Order the actiavted rules by decreasing firing strength
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

    pub fn prompt_infer(&mut self, t_norm: TNorm) -> Result<Vec<ActivatedRule<'_>>, WMModelError> {
        self.build_mf_config()?;

        let mut crisp_row = HashMap::new();

        println!("\n==================================================");
        println!("             INPUT INFERENCE                        ");
        println!("==================================================");

        for feature in self.mf_configs.keys() {
            loop {
                print!("\n👉 Enter the value for the attribute \"{}\": ", feature);
                std::io::stdout().flush().unwrap();

                let mut input = String::new();
                if std::io::stdin().read_line(&mut input).is_err() {
                    println!("❌ Error reading input. Try again.");
                    continue;
                }

                let input = input.trim();

                match input.parse::<f64>() {
                    Ok(val) => {
                        crisp_row.insert(feature.clone(), val);
                        break;
                    }
                    Err(_) => {
                        println!(
                            "❌ Invalid number \"{}\"! Please enter a valid float value (e.g., 12.5).",
                            input
                        );
                    }
                }
            }
        }

        let (_, activated_rules) = self.infer(&crisp_row, t_norm);

        Ok(activated_rules)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mb_function_triangular() {
        let abc = [0.0, 5.0, 10.0];

        assert_eq!(mb_function(5.0, abc), 1.0, "Peak value must be 1.0");
        assert_eq!(
            mb_function(2.5, abc),
            0.5,
            "Ascending slope midpoint must be 0.5"
        );
        assert_eq!(
            mb_function(7.5, abc),
            0.5,
            "Descending slope midpoint must be 0.5"
        );
        assert_eq!(mb_function(0.0, abc), 0.0, "Left boundary must be 0.0");
        assert_eq!(mb_function(10.0, abc), 0.0, "Right boundary must be 0.0");
        assert_eq!(
            mb_function(-1.0, abc),
            0.0,
            "Outside left support must be 0.0"
        );
        assert_eq!(
            mb_function(11.0, abc),
            0.0,
            "Outside right support must be 0.0"
        );
    }

    #[test]
    fn test_mb_function_shoulders() {
        // Left shoulder (a == b)
        let left_shoulder = [0.0, 0.0, 5.0];
        assert_eq!(
            mb_function(-2.0, left_shoulder),
            1.0,
            "Left of shoulder must be 1.0"
        );
        assert_eq!(mb_function(0.0, left_shoulder), 1.0);
        assert_eq!(mb_function(2.5, left_shoulder), 0.5);

        // Right shoulder (b == c)
        let right_shoulder = [5.0, 10.0, 10.0];
        assert_eq!(
            mb_function(12.0, right_shoulder),
            1.0,
            "Right of shoulder must be 1.0"
        );
        assert_eq!(mb_function(10.0, right_shoulder), 1.0);
        assert_eq!(mb_function(7.5, right_shoulder), 0.5);
    }
    #[test]
    fn test_calculate_firing_strength() {
        let mut data = HashMap::new();
        data.insert("temp".to_string(), "high".to_string());
        data.insert("humidity".to_string(), "low".to_string());
        let antecedents = Antecedents { data };

        let mut memberships = HashMap::new();
        let mut temp_mfs = HashMap::new();
        temp_mfs.insert("high".to_string(), 0.8);
        let mut hum_mfs = HashMap::new();
        hum_mfs.insert("low".to_string(), 0.5);

        memberships.insert("temp".to_string(), temp_mfs);
        memberships.insert("humidity".to_string(), hum_mfs);

        // T-Norm Product: 0.8 * 0.5 = 0.4
        let prod = calculate_firing_strength(&antecedents, &memberships, TNorm::Product);
        assert!((prod - 0.4).abs() < 1e-6);

        // T-Norm Min: min(0.8, 0.5) = 0.5
        let min_val = calculate_firing_strength(&antecedents, &memberships, TNorm::Min);
        assert!((min_val - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_calculate_firing_strength_missing_feature() {
        let mut data = HashMap::new();
        data.insert("temp".to_string(), "high".to_string());
        data.insert("unknown_feat".to_string(), "low".to_string());
        let antecedents = Antecedents { data };

        let memberships = HashMap::new(); // Empty map

        // Should return 0.0 via unwrap_or(0.0) without panicking
        let strength = calculate_firing_strength(&antecedents, &memberships, TNorm::Product);
        assert_eq!(strength, 0.0);
    }
}

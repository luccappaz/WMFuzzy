use crate::fuzzy::rule::{ActivatedRule, FuzzyRule};

pub trait Defuzzifier {
    fn defuzzify(
        &self,
        activated_rules: &[ActivatedRule],
        fallback_rule: Option<(&FuzzyRule, f64)>,
        default_fallback: f64,
    ) -> f64;
}

pub struct WeightedAverageDefuzzifier;

impl Defuzzifier for WeightedAverageDefuzzifier {
    fn defuzzify(
        &self,
        activated_rules: &[ActivatedRule],
        fallback_rule: Option<(&FuzzyRule, f64)>,
        default_fallback: f64,
    ) -> f64 {
        let mut total_strength = 0.0;
        let mut weighted_sum = 0.0;

        for act in activated_rules {
            weighted_sum += act.firing_strength * act.rule.weighted_average;
            total_strength += act.firing_strength;
        }

        if total_strength > 0.0 {
            weighted_sum / total_strength
        } else if let Some((rule, _)) = fallback_rule {
            rule.weighted_average
        } else {
            default_fallback
        }
    }
}

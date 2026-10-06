use serde::{Deserialize, Serialize};

use crate::{
    fuzzy::granularity::Granularity,
    membership::mb_function::{FuzzyStrategy, InnerStrategy, MFKind, MembershipOp, Partitionable},
};

/// Triangular Membership Function defined by parameters `(a, b, c)` where:
/// - `a`: lower bound (degree = 0.0)
/// - `b`: peak / center (degree = 1.0)
/// - `c`: upper bound (degree = 0.0)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TriangularMF {
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

impl TriangularMF {
    /// Creates a new `TriangularMF`.
    ///
    /// # Panics (Debug Mode)
    /// Asserts that parameters satisfy `a <= b <= c`.
    pub fn new(a: f64, b: f64, c: f64) -> Self {
        debug_assert!(
            a <= b && b <= c,
            "Triangular parameters must satisfy a <= b <= c"
        );
        Self { a, b, c }
    }

    /// Returns the `[a, b, c]` parameters array for this instance.
    #[inline]
    pub fn abc(&self) -> [f64; 3] {
        [self.a, self.b, self.c]
    }

    /// Computes raw triangular `[a, b, c]` parameters from a point sequence for a given index `i`.
    pub fn compute_abc(index: usize, points: &[f64]) -> [f64; 3] {
        let n = points.len();
        debug_assert!(n >= 2, "Points slice must contain at least 2 elements");

        let b = points[index];
        let a = if index == 0 {
            points[0]
        } else {
            points[index - 1]
        };
        let c = if index == n - 1 {
            points[n - 1]
        } else {
            points[index + 1]
        };

        [a, b, c]
    }

    /// Computes parameters `(a, b, c)` for the $i$-th term in a uniform linear partition.
    pub fn compute_params_linear(index: usize, min: f64, max: f64, count: usize) -> Self {
        assert!(count >= 2, "Granularity count must be at least 2");

        let step = (max - min) / (count - 1) as f64;
        let b = min + index as f64 * step;
        let a = if index == 0 { min } else { b - step };
        let c = if index == count - 1 { max } else { b + step };

        Self::new(a, b, c)
    }

    /// Computes parameters `(a, b, c)` from an array of custom key points.
    pub fn compute_params_custom(index: usize, points: &[f64]) -> Self {
        let [a, b, c] = Self::compute_abc(index, points);
        Self::new(a, b, c)
    }
}

impl MembershipOp for TriangularMF {
    fn eval(&self, x: f64) -> f64 {
        // Left shoulder support (a == b)
        if self.a == self.b && x <= self.b {
            return 1.0;
        }
        // Right shoulder support (b == c)
        if self.b == self.c && x >= self.b {
            return 1.0;
        }
        if x <= self.a || x >= self.c {
            return 0.0;
        }
        if x == self.b {
            return 1.0;
        }
        if x < self.b {
            (x - self.a) / (self.b - self.a)
        } else {
            (self.c - x) / (self.c - self.b)
        }
    }

    #[inline]
    fn center(&self) -> f64 {
        self.b
    }

    #[inline]
    fn support(&self) -> (f64, f64) {
        (self.a, self.c)
    }
}

impl Partitionable for TriangularMF {
    fn create_partition(
        _kind: MFKind,
        strategy: FuzzyStrategy,
        granularity: Granularity,
    ) -> Vec<Self> {
        match strategy.inner() {
            InnerStrategy::Linear { min, max } => {
                let n = granularity.count();
                (0..n)
                    .map(|i| Self::compute_params_linear(i, *min, *max, n))
                    .collect()
            }
            InnerStrategy::Custom { knots } => {
                let n = knots.len();
                assert!(n >= 2, "Custom partition requires at least 2 knots");
                (0..n)
                    .map(|i| Self::compute_params_custom(i, knots))
                    .collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuzzy::granularity::Granularity;

    const EPSILON: f64 = 1e-9;

    #[test]
    fn test_construction_and_getters() {
        let mf = TriangularMF::new(0.0, 5.0, 10.0);

        assert_eq!(mf.abc(), [0.0, 5.0, 10.0]);
        assert_eq!(mf.center(), 5.0);
        assert_eq!(mf.support(), (0.0, 10.0));
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Triangular parameters must satisfy a <= b <= c")]
    fn test_invalid_bounds_panic() {
        TriangularMF::new(5.0, 2.0, 10.0);
    }

    #[test]
    fn test_eval_standard_triangle() {
        // Standard triangle centered at 5.0 in interval [2.0, 8.0]
        let mf = TriangularMF::new(2.0, 5.0, 8.0);

        // Outside support
        assert_eq!(mf.eval(1.0), 0.0);
        assert_eq!(mf.eval(2.0), 0.0);
        assert_eq!(mf.eval(8.0), 0.0);
        assert_eq!(mf.eval(9.0), 0.0);

        // Peak
        assert_eq!(mf.eval(5.0), 1.0);

        // Slopes
        assert!((mf.eval(3.5) - 0.5).abs() < EPSILON); // (3.5 - 2.0) / (5.0 - 2.0) = 1.5 / 3.0 = 0.5
        assert!((mf.eval(6.5) - 0.5).abs() < EPSILON); // (8.0 - 6.5) / (8.0 - 5.0) = 1.5 / 3.0 = 0.5
    }

    #[test]
    fn test_eval_left_shoulder() {
        // Left shoulder: peak at x <= 0.0, dropping linearly to 0.0 at x = 5.0
        let mf = TriangularMF::new(0.0, 0.0, 5.0);

        // Should return 1.0 for all x <= 0.0
        assert_eq!(mf.eval(-10.0), 1.0);
        assert_eq!(mf.eval(-0.1), 1.0);
        assert_eq!(mf.eval(0.0), 1.0);

        // Linear drop off in (0.0, 5.0)
        assert!((mf.eval(2.5) - 0.5).abs() < EPSILON);

        // Outside right boundary
        assert_eq!(mf.eval(5.0), 0.0);
        assert_eq!(mf.eval(6.0), 0.0);
    }

    #[test]
    fn test_eval_right_shoulder() {
        // Right shoulder: linear rise from 5.0 to peak at x >= 10.0
        let mf = TriangularMF::new(5.0, 10.0, 10.0);

        // Outside left boundary
        assert_eq!(mf.eval(4.0), 0.0);
        assert_eq!(mf.eval(5.0), 0.0);

        // Linear rise in (5.0, 10.0)
        assert!((mf.eval(7.5) - 0.5).abs() < EPSILON);

        // Should return 1.0 for all x >= 10.0
        assert_eq!(mf.eval(10.0), 1.0);
        assert_eq!(mf.eval(10.1), 1.0);
        assert_eq!(mf.eval(20.0), 1.0);
    }

    #[test]
    fn test_compute_params_linear() {
        // Domain [0.0, 10.0] split into 3 partitions (step = 5.0)
        let mf0 = TriangularMF::compute_params_linear(0, 0.0, 10.0, 3);
        let mf1 = TriangularMF::compute_params_linear(1, 0.0, 10.0, 3);
        let mf2 = TriangularMF::compute_params_linear(2, 0.0, 10.0, 3);

        assert_eq!(mf0.abc(), [0.0, 0.0, 5.0]); // Left shoulder
        assert_eq!(mf1.abc(), [0.0, 5.0, 10.0]); // Center triangle
        assert_eq!(mf2.abc(), [5.0, 10.0, 10.0]); // Right shoulder
    }

    #[test]
    fn test_compute_params_custom() {
        // Non-uniform grid knots
        let knots = vec![0.0, 2.0, 10.0];

        let mf0 = TriangularMF::compute_params_custom(0, &knots);
        let mf1 = TriangularMF::compute_params_custom(1, &knots);
        let mf2 = TriangularMF::compute_params_custom(2, &knots);

        assert_eq!(mf0.abc(), [0.0, 0.0, 2.0]);
        assert_eq!(mf1.abc(), [0.0, 2.0, 10.0]);
        assert_eq!(mf2.abc(), [2.0, 10.0, 10.0]);
    }

    #[test]
    fn test_create_partition_linear_strategy() {
        let strategy = FuzzyStrategy::linear(0.0, 100.0).unwrap();
        let granularity = Granularity::THREE;

        let partition = TriangularMF::create_partition(MFKind::Triangular, strategy, granularity);

        assert_eq!(partition.len(), 3);
        assert_eq!(partition[0].abc(), [0.0, 0.0, 50.0]);
        assert_eq!(partition[1].abc(), [0.0, 50.0, 100.0]);
        assert_eq!(partition[2].abc(), [50.0, 100.0, 100.0]);
    }

    #[test]
    fn test_serde_roundtrip() {
        let mf = TriangularMF::new(1.0, 2.5, 4.0);
        let serialized = serde_json::to_string(&mf).unwrap();
        let deserialized: TriangularMF = serde_json::from_str(&serialized).unwrap();

        assert_eq!(mf, deserialized);
    }
}

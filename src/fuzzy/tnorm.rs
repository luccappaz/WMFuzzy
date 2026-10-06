use core::f64;

use serde::{Deserialize, Serialize};

// Calling variable
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TNorm {
    Product,
    Minimum,
}

impl TNormOp for TNorm {
    fn identity(&self) -> f64 {
        match self {
            TNorm::Product => 1.0,
            TNorm::Minimum => 1.0,
        }
    }
    fn combine(&self, a: f64, b: f64) -> f64 {
        match self {
            TNorm::Product => a * b,
            TNorm::Minimum => a.min(b),
        }
    }
}

pub trait TNormOp {
    fn combine(&self, a: f64, b: f64) -> f64;
    fn identity(&self) -> f64;

    // Compute the fire strength
    fn evaluate_all<I>(&self, mus: I) -> f64
    where
        I: IntoIterator<Item = f64>,
    {
        let mut acc = self.identity();
        for mu in mus {
            acc = self.combine(acc, mu);
            if acc == 0.0 {
                break;
            }
        }
        acc
    }
}

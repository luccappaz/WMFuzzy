// src/dataset/mod.rs
use crate::{error::WMModelError, types::HashDataset};
use std::collections::HashMap;

pub trait TabularDataset {
    fn num_rows(&self) -> usize;
    fn get_row_as_map(&self, row_idx: usize, features: &[&str]) -> HashMap<String, f64>;
    fn get_column_f64(&self, name: &str) -> Result<Vec<f64>, WMModelError>;
}

impl TabularDataset for HashDataset {
    fn num_rows(&self) -> usize {
        self.len()
    }

    fn get_row_as_map(&self, row_idx: usize, features: &[&str]) -> HashMap<String, f64> {
        let mut map = HashMap::with_capacity(features.len());
        if let Some(row) = self.get(row_idx) {
            for &feat in features {
                if let Some(&val) = row.get(feat) {
                    map.insert(feat.to_string(), val);
                }
            }
        }
        map
    }

    fn get_column_f64(&self, name: &str) -> Result<Vec<f64>, WMModelError> {
        Ok(self.iter().map(|r| *r.get(name).unwrap_or(&0.0)).collect())
    }
}

// TargetVector implementations
pub trait TargetVector {
    fn num_samples(&self) -> usize;
    fn to_vec_f64(&self) -> Result<Vec<f64>, WMModelError>;
}

impl TargetVector for Vec<f64> {
    fn num_samples(&self) -> usize {
        self.len()
    }

    fn to_vec_f64(&self) -> Result<Vec<f64>, WMModelError> {
        Ok(self.clone())
    }
}

impl TargetVector for &[f64] {
    fn num_samples(&self) -> usize {
        self.len()
    }

    fn to_vec_f64(&self) -> Result<Vec<f64>, WMModelError> {
        Ok(self.to_vec())
    }
}

impl TargetVector for &[u32] {
    fn num_samples(&self) -> usize {
        self.len()
    }

    fn to_vec_f64(&self) -> Result<Vec<f64>, WMModelError> {
        Ok(self.iter().map(|&v| v as f64).collect())
    }
}

impl TargetVector for Vec<u32> {
    fn num_samples(&self) -> usize {
        self.len()
    }

    fn to_vec_f64(&self) -> Result<Vec<f64>, WMModelError> {
        Ok(self.iter().map(|&v| v as f64).collect())
    }
}

#[cfg(feature = "polars")]
pub mod polars;

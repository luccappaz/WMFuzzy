use crate::error::WMModelError;

// src/dataset/polars.rs
use super::{TabularDataset, TargetVector};
use polars::prelude::*;
use std::collections::HashMap;

impl TabularDataset for DataFrame {
    fn num_rows(&self) -> usize {
        self.height()
    }

    fn get_row_as_map(&self, row_idx: usize, features: &[&str]) -> HashMap<String, f64> {
        let mut map = HashMap::with_capacity(features.len());
        for &feat in features {
            if let Ok(col) = self.column(feat)
                && let Ok(casted) = col.cast(&DataType::Float64)
                && let Ok(ca) = casted.f64()
                && let Some(val) = ca.get(row_idx)
            {
                map.insert(feat.to_string(), val);
            }
        }
        map
    }

    fn get_column_f64(&self, name: &str) -> Result<Vec<f64>, WMModelError> {
        let col = self
            .column(name)
            .map_err(|e| WMModelError::DatasetError(e.to_string()))?;

        let casted = col
            .cast(&DataType::Float64)
            .map_err(|e| WMModelError::DatasetError(e.to_string()))?;

        let ca = casted
            .f64()
            .map_err(|e| WMModelError::DatasetError(e.to_string()))?;

        Ok(ca.iter().map(|v| v.unwrap_or(0.0)).collect())
    }
}

impl TargetVector for polars::prelude::Series {
    fn num_samples(&self) -> usize {
        self.len()
    }

    fn to_vec_f64(&self) -> Result<Vec<f64>, WMModelError> {
        let casted = self
            .cast(&DataType::Float64)
            .map_err(|e| WMModelError::DatasetError(e.to_string()))?;

        let ca = casted
            .f64()
            .map_err(|e| WMModelError::DatasetError(e.to_string()))?;

        Ok(ca.iter().map(|v| v.unwrap_or(0.0)).collect())
    }
}

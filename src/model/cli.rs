use crate::error::WMModelError;
use crate::fuzzy::{ActivatedRule, TNorm};
use crate::model::wm::WMModel;
use std::collections::HashMap;
use std::io::{self, BufRead, Write};

/// Interactive CLI extension trait for [`WMModel`].
///
/// Extends `WMModel` with command-line user prompting capabilities for real-time,
/// single-sample fuzzy inference.
pub trait WMModelCliExt {
    /// Prompts the user via `stdin` and `stdout` for all input features configured in the model.
    ///
    /// Interactively collects crisp feature values, validates user inputs against float parsing errors,
    /// and invokes single-sample inference (`infer`).
    ///
    /// Works identically for both **Case 1** (Linguistic Target Partitioning) and **Case 2** (Continuous Numeric Target) models.
    ///
    /// # Arguments
    /// * `t_norm` - Triangular norm operator used to combine antecedent firing strengths (`TNorm::Product` or `TNorm::Minimum`).
    ///
    /// # Errors
    /// Returns [`WMModelError`] if the underlying model is not built or fails to resolve feature configurations.
    fn prompt_infer(&mut self, t_norm: TNorm) -> Result<Vec<ActivatedRule<'_>>, WMModelError>;
}

impl WMModelCliExt for WMModel {
    fn prompt_infer(&mut self, t_norm: TNorm) -> Result<Vec<ActivatedRule<'_>>, WMModelError> {
        let stdin = io::stdin();
        let mut stdout = io::stdout();
        prompt_infer_from_io(self, &mut stdin.lock(), &mut stdout, t_norm)
    }
}

/// Generic I/O helper function for interactive fuzzy model inference.
///
/// Accepts any input reader ([`BufRead`]) and output writer ([`Write`]), enabling testable
/// interactive prompting with custom streams, buffers, or standard console handles.
///
/// # Arguments
/// * `model` - Mutable reference to a built [`WMModel`].
/// * `reader` - Input stream handle (e.g., `stdin.lock()` or `Cursor<Vec<u8>>`).
/// * `writer` - Output stream handle (e.g., `stdout()` or `Vec<u8>`).
/// * `t_norm` - Triangular norm used for calculating rule activation levels.
///
/// # Returns
/// A vector of [`ActivatedRule`] references sorted in descending order of firing strength.
///
/// # Errors
/// Returns [`WMModelError::InvalidStrategy`] if model structures cannot be built.
pub fn prompt_infer_from_io<'a, R: BufRead, W: Write>(
    model: &'a mut WMModel,
    reader: &mut R,
    writer: &mut W,
    t_norm: TNorm,
) -> Result<Vec<ActivatedRule<'a>>, WMModelError> {
    model.build()?;

    let mut crisp_row = HashMap::new();

    writeln!(
        writer,
        "\n=================================================="
    )?;
    writeln!(writer, "                INPUT INFERENCE                   ")?;
    writeln!(writer, "==================================================")?;

    for feature in model.mf_configs.keys() {
        loop {
            write!(
                writer,
                "\n👉 Enter the value for the attribute \"{}\": ",
                feature
            )?;
            writer.flush()?;

            let mut input = String::new();
            if reader.read_line(&mut input).is_err() {
                writeln!(writer, "❌ Error reading input. Try again.")?;
                continue;
            }

            let trimmed = input.trim();
            match trimmed.parse::<f64>() {
                Ok(val) => {
                    crisp_row.insert(feature.clone(), val);
                    break;
                }
                Err(_) => {
                    writeln!(
                        writer,
                        "❌ Invalid number \"{}\"! Enter a valid float.",
                        trimmed
                    )?;
                }
            }
        }
    }

    let (_, activated_rules) = model.infer(&crisp_row, t_norm);
    Ok(activated_rules)
}

#[cfg(test)]
mod cli_tests {
    use super::*;
    use crate::types::HashDataset;
    use std::collections::HashMap;
    use std::io::Cursor;

    /// Helper: Sets up a trained model with Case 1 (Linguistic Target Partitioning).
    fn setup_case1_linguistic_model() -> WMModel {
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

        let x_train: HashDataset = vec![HashMap::from([("x1".to_string(), 5.0)])];
        let y_train = vec![50.0];

        model
            .generate_rules(&x_train, &y_train, None, TNorm::Product, 0.0, 0.0)
            .unwrap();

        model
    }

    /// Helper: Sets up a trained model with Case 2 (Continuous Numeric Target Regression).
    fn setup_case2_numeric_model() -> WMModel {
        let mut model = WMModel::new();
        model
            .set_kind("triangular")
            .unwrap()
            .add_granularity("x1", 3)
            .unwrap()
            .add_linear_strategy("x1", 0.0, 10.0)
            .unwrap();

        let x_train: HashDataset = vec![HashMap::from([("x1".to_string(), 5.0)])];
        let y_train = vec![5.2];

        model
            .generate_rules(&x_train, &y_train, None, TNorm::Product, 0.0, 0.0)
            .unwrap();

        model
    }

    #[test]
    fn test_prompt_infer_case1_linguistic_target() {
        let mut model = setup_case1_linguistic_model();

        let input_bytes = b"5.0\n";
        let mut reader = Cursor::new(input_bytes);
        let mut writer = Vec::new();

        let result = prompt_infer_from_io(&mut model, &mut reader, &mut writer, TNorm::Product);

        assert!(result.is_ok());
        let activated_rules = result.unwrap();
        assert!(!activated_rules.is_empty());
        assert!(model.target_mfs().is_ok());

        let output_text = String::from_utf8(writer).unwrap();
        assert!(output_text.contains("INPUT INFERENCE"));
        assert!(output_text.contains("Enter the value for the attribute \"x1\""));
    }

    #[test]
    fn test_prompt_infer_case2_numeric_target() {
        let mut model = setup_case2_numeric_model();

        let input_bytes = b"5.0\n";
        let mut reader = Cursor::new(input_bytes);
        let mut writer = Vec::new();

        let result = prompt_infer_from_io(&mut model, &mut reader, &mut writer, TNorm::Product);

        assert!(result.is_ok());
        let activated_rules = result.unwrap();
        assert!(!activated_rules.is_empty());
        assert!(model.target_mfs().is_err()); // Case 2 has no target MFs

        let output_text = String::from_utf8(writer).unwrap();
        assert!(output_text.contains("INPUT INFERENCE"));
        assert!(output_text.contains("Enter the value for the attribute \"x1\""));
    }

    #[test]
    fn test_prompt_infer_invalid_input_retry() {
        let mut model = setup_case1_linguistic_model();

        // First invalid string input ("abc"), followed by a valid float ("5.0")
        let input_bytes = b"abc\n5.0\n";
        let mut reader = Cursor::new(input_bytes);
        let mut writer = Vec::new();

        let result = prompt_infer_from_io(&mut model, &mut reader, &mut writer, TNorm::Product);

        assert!(result.is_ok());

        let output_text = String::from_utf8(writer).unwrap();
        assert!(output_text.contains("❌ Invalid number \"abc\"!"));
        assert!(output_text.contains("Enter the value for the attribute \"x1\""));
    }

    #[test]
    fn test_prompt_infer_multiple_features_sequence() {
        let mut model = WMModel::new();
        model
            .set_kind("triangular")
            .unwrap()
            .add_granularity("x1", 3)
            .unwrap()
            .add_linear_strategy("x1", 0.0, 10.0)
            .unwrap()
            .add_granularity("x2", 3)
            .unwrap()
            .add_linear_strategy("x2", 0.0, 10.0)
            .unwrap();

        let x_train: HashDataset = vec![HashMap::from([
            ("x1".to_string(), 5.0),
            ("x2".to_string(), 5.0),
        ])];
        let y_train = vec![1.0];

        model
            .generate_rules(&x_train, &y_train, None, TNorm::Product, 0.0, 0.0)
            .unwrap();

        let input_bytes = b"2.5\n7.5\n";
        let mut reader = Cursor::new(input_bytes);
        let mut writer = Vec::new();

        let result = prompt_infer_from_io(&mut model, &mut reader, &mut writer, TNorm::Product);

        assert!(result.is_ok());

        let output_text = String::from_utf8(writer).unwrap();
        assert!(output_text.contains("Enter the value for the attribute \"x1\""));
        assert!(output_text.contains("Enter the value for the attribute \"x2\""));
    }
}

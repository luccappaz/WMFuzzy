use crate::error::WMModelError;
use crate::fuzzy::{ActivatedRule, TNorm};
use crate::model::wm::WMModel;
use std::collections::HashMap;
use std::io::{self, BufRead, Write};

pub trait WMModelCliExt {
    fn prompt_infer(&mut self, t_norm: TNorm) -> Result<Vec<ActivatedRule<'_>>, WMModelError>;
}

impl WMModelCliExt for WMModel {
    fn prompt_infer(&mut self, t_norm: TNorm) -> Result<Vec<ActivatedRule<'_>>, WMModelError> {
        let stdin = io::stdin();
        let mut stdout = io::stdout();
        prompt_infer_from_io(self, &mut stdin.lock(), &mut stdout, t_norm)
    }
}

/// Função utilitária genérica que aceita qualquer leitor/escritor de I/O
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
    use crate::types::HashDataset;

    use super::*;
    use std::collections::HashMap;
    use std::io::Cursor;

    fn setup_trained_model() -> WMModel {
        let mut model = WMModel::new();
        model
            .set_kind("triangular")
            .unwrap()
            .add_granularity("x1", 3)
            .unwrap()
            .add_linear_strategy("x1", 0.0, 10.0)
            .unwrap();

        let x_train: HashDataset = vec![HashMap::from([("x1".to_string(), 5.0)])];
        let y_train = vec![1.0];

        model
            .generate_rules(&x_train, &y_train, None, TNorm::Product, 0.0, 0.0)
            .unwrap();

        model
    }

    #[test]
    fn test_prompt_infer_valid_input() {
        let mut model = setup_trained_model();

        let input_bytes = b"5.0\n";
        let mut reader = Cursor::new(input_bytes);
        let mut writer = Vec::new();

        let result = prompt_infer_from_io(&mut model, &mut reader, &mut writer, TNorm::Product);

        assert!(result.is_ok());
        let activated_rules = result.unwrap();

        assert!(!activated_rules.is_empty());

        let output_text = String::from_utf8(writer).unwrap();
        assert!(output_text.contains("INPUT INFERENCE"));
        assert!(output_text.contains("Enter the value for the attribute \"x1\""));
    }

    #[test]
    fn test_prompt_infer_invalid_input_retry() {
        let mut model = setup_trained_model();

        // Primeira entrada inválida, seguida de um valor float válido
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

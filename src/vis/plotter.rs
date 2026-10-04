use plotters::prelude::*;
use std::fs;
use std::path::Path;

use crate::membership::mb_function::MembershipOp;
use crate::model::wm::WMModel;

impl WMModel {
    /// Plots and saves membership function charts for all configured features.
    pub fn plot_membership_functions(
        &self,
        output_dir: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let output_path = Path::new(output_dir);
        fs::create_dir_all(output_path)?;

        let colors = [&RED, &BLUE, &GREEN, &MAGENTA, &CYAN, &YELLOW];

        let model_limits = self.limits()?;
        // Recorre sobre cada feature configurada no modelo
        for (feature_name, limits) in model_limits {
            if limits.len() < 2 {
                continue;
            }

            let feature_mfs = match self.mf_configs()?.get(feature_name) {
                Some(mfs) if !mfs.is_empty() => mfs,
                _ => continue,
            };

            let file_path = output_path.join(format!("mf_{}.png", feature_name));
            let root = BitMapBackend::new(&file_path, (800, 500)).into_drawing_area();
            root.fill(&WHITE)?;

            // Define os limites do eixo X com padding
            let min_x = limits.first().copied().unwrap_or(0.0);
            let max_x = limits.last().copied().unwrap_or(1.0);
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

            // Ordena os conjuntos nebulosos pelo centro (center) para manter a legenda coerente
            let mut sorted_mfs: Vec<(
                &String,
                &crate::membership::mb_function::MembershipFunction,
            )> = feature_mfs.iter().collect();
            sorted_mfs.sort_by(|a, b| {
                a.1.center()
                    .partial_cmp(&b.1.center())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let steps = 300;

            for (i, (label, mf)) in sorted_mfs.into_iter().enumerate() {
                let color = colors[i % colors.len()];

                // Gera os pontos amostrando x via método polimórfico mf.eval(x)
                let points: Vec<(f64, f64)> = (0..=steps)
                    .map(|s| {
                        let x = x_start + (s as f64 / steps as f64) * (x_end - x_start);
                        let y = mf.eval(x);
                        (x, y)
                    })
                    .collect();

                chart
                    .draw_series(LineSeries::new(points, color.stroke_width(2)))?
                    .label(label)
                    .legend(move |(x, y)| {
                        PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2))
                    });
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
}

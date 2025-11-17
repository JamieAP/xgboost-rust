use crate::error::{XGBoostError, XGBoostResult};
use crate::Booster;
use polars::prelude::*;

/// Extension trait for XGBoost Booster to support Polars DataFrames
pub trait BoosterPolarsExt {
    /// Predict using a Polars DataFrame as input
    ///
    /// This method efficiently converts the DataFrame to the format XGBoost expects
    /// and runs prediction. All numeric columns will be used as features.
    ///
    /// # Arguments
    /// * `df` - Input DataFrame with numeric features
    /// * `option_mask` - Prediction options (see `predict_option` module)
    /// * `training` - Whether this is for training (false for inference)
    ///
    /// # Returns
    /// A vector of prediction values
    ///
    /// # Example
    /// ```no_run
    /// # use xgboost_rust::{Booster, BoosterPolarsExt};
    /// # use polars::prelude::*;
    /// let booster = Booster::load("model.json").unwrap();
    ///
    /// let df = df! {
    ///     "feature1" => [1.0f32, 2.0, 3.0],
    ///     "feature2" => [4.0f32, 5.0, 6.0],
    /// }.unwrap();
    ///
    /// let predictions = booster.predict_dataframe(&df, 0, false).unwrap();
    /// ```
    fn predict_dataframe(
        &self,
        df: &DataFrame,
        option_mask: u32,
        training: bool,
    ) -> XGBoostResult<Vec<f32>>;

    /// Predict using specific columns from a Polars DataFrame
    ///
    /// # Arguments
    /// * `df` - Input DataFrame
    /// * `columns` - Column names to use as features (in order)
    /// * `option_mask` - Prediction options
    /// * `training` - Whether this is for training
    fn predict_dataframe_with_columns(
        &self,
        df: &DataFrame,
        columns: &[&str],
        option_mask: u32,
        training: bool,
    ) -> XGBoostResult<Vec<f32>>;
}

impl BoosterPolarsExt for Booster {
    fn predict_dataframe(
        &self,
        df: &DataFrame,
        option_mask: u32,
        training: bool,
    ) -> XGBoostResult<Vec<f32>> {
        let (data, num_rows, num_features) = dataframe_to_dense(df)?;
        self.predict(&data, num_rows, num_features, option_mask, training)
    }

    fn predict_dataframe_with_columns(
        &self,
        df: &DataFrame,
        columns: &[&str],
        option_mask: u32,
        training: bool,
    ) -> XGBoostResult<Vec<f32>> {
        let column_names: Vec<String> = columns.iter().map(|s| s.to_string()).collect();
        let selected = df.select(column_names).map_err(|e| XGBoostError {
            description: format!("Failed to select columns: {}", e),
        })?;

        let (data, num_rows, num_features) = dataframe_to_dense(&selected)?;
        self.predict(&data, num_rows, num_features, option_mask, training)
    }
}

/// Convert a Polars DataFrame to dense f32 data in row-major format
///
/// Optimized column-by-column conversion for better cache locality on source data.
fn dataframe_to_dense(df: &DataFrame) -> XGBoostResult<(Vec<f32>, usize, usize)> {
    let num_rows = df.height();
    let num_features = df.width();

    if num_rows == 0 || num_features == 0 {
        return Err(XGBoostError {
            description: "DataFrame has zero rows or columns".to_string(),
        });
    }

    // Pre-allocate with exact size
    let total_elements = num_rows * num_features;
    let mut data = vec![0.0f32; total_elements];

    // Process column by column - cast to Float32 for simplicity and speed
    for (col_idx, column) in df.get_columns().iter().enumerate() {
        let series = column.as_materialized_series();

        // Cast to Float32 - Polars handles all type conversions efficiently
        let f32_series = series.cast(&DataType::Float32).map_err(|e| XGBoostError {
            description: format!("Failed to cast column to f32: {}", e),
        })?;

        let ca = f32_series.f32().map_err(|e| XGBoostError {
            description: format!("Failed to get f32 array: {}", e),
        })?;

        for (row_idx, opt_val) in ca.iter().enumerate() {
            let val = opt_val.ok_or_else(|| XGBoostError {
                description: format!("Null value at row {}, col {}", row_idx, col_idx),
            })?;
            data[row_idx * num_features + col_idx] = val;
        }
    }

    Ok((data, num_rows, num_features))
}

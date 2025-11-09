use crate::error::{XGBoostError, XGBoostResult};
use crate::sys;
use std::ffi::CString;
use std::path::Path;
use std::ptr;

/// An XGBoost Booster for making predictions.
///
/// # Thread Safety
///
/// **Thread safety depends on XGBoost version:**
/// - **XGBoost ≥ 1.4**: Predictions are thread-safe for tree models (gbtree/dart).
///   However, prediction cache may still have edge cases.
/// - **XGBoost < 1.4**: NOT thread-safe for concurrent predictions.
///
/// This wrapper does NOT implement `Send` or `Sync` to be conservative.
/// For multi-threaded use cases, use one of these approaches:
///
/// 1. **Create one Booster per thread** (recommended, works with all versions):
///    ```ignore
///    let booster = Booster::load("model.json")?;
///    thread::spawn(move || {
///        booster.predict(...);  // Each thread owns its Booster
///    });
///    ```
///
/// 2. **Wrap in Arc<Mutex<Booster>>** for shared access:
///    ```ignore
///    use std::sync::{Arc, Mutex};
///    let booster = Arc::new(Mutex::new(Booster::load("model.json")?));
///    let booster_clone = booster.clone();
///    thread::spawn(move || {
///        let booster = booster_clone.lock().unwrap();
///        booster.predict(...);
///    });
///    ```
pub struct Booster {
    handle: sys::BoosterHandle,
}

// NOTE: We do NOT implement Send or Sync for Booster because:
// 1. Thread safety guarantees vary by XGBoost version (≥1.4 is safer)
// 2. C API documentation doesn't explicitly guarantee thread safety
// 3. Users should explicitly choose synchronization strategy (one-per-thread or Mutex)
//
// If you need to share a Booster across threads, wrap it in Arc<Mutex<Booster>>.

impl Booster {
    /// Load a model from a file
    ///
    /// # Arguments
    /// * `path` - Path to the model file (can be JSON, binary, or deprecated text format)
    ///
    /// # Example
    /// ```no_run
    /// use xgboost_rust::Booster;
    ///
    /// let booster = Booster::load("model.json").unwrap();
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> XGBoostResult<Self> {
        let path_str = path.as_ref().to_str()
            .ok_or_else(|| XGBoostError {
                description: "Path contains invalid UTF-8 characters".to_string(),
            })?;
        let path_c_str = CString::new(path_str)
            .map_err(|e| XGBoostError {
                description: format!("Path contains NUL byte: {}", e),
            })?;

        // Create a booster first
        let mut handle: sys::BoosterHandle = ptr::null_mut();
        XGBoostError::check_return_value(unsafe {
            sys::XGBoosterCreate(ptr::null(), 0, &mut handle)
        })?;

        // Load model into the booster
        XGBoostError::check_return_value(unsafe {
            sys::XGBoosterLoadModel(
                handle,
                path_c_str.as_ptr(),
            )
        })?;

        Ok(Booster { handle })
    }

    /// Load a model from a memory buffer
    ///
    /// # Arguments
    /// * `buffer` - Model content as bytes
    ///
    /// # Example
    /// ```no_run
    /// use xgboost_rust::Booster;
    /// use std::fs;
    ///
    /// let buffer = fs::read("model.json").unwrap();
    /// let booster = Booster::load_from_buffer(&buffer).unwrap();
    /// ```
    pub fn load_from_buffer(buffer: &[u8]) -> XGBoostResult<Self> {
        // Create a booster first
        let mut handle: sys::BoosterHandle = ptr::null_mut();
        XGBoostError::check_return_value(unsafe {
            sys::XGBoosterCreate(ptr::null(), 0, &mut handle)
        })?;

        // Load model from buffer into the booster
        XGBoostError::check_return_value(unsafe {
            sys::XGBoosterLoadModelFromBuffer(
                handle,
                buffer.as_ptr() as *const std::os::raw::c_void,
                buffer.len() as u64,
            )
        })?;

        Ok(Booster { handle })
    }

    /// Make predictions on data
    ///
    /// # Arguments
    /// * `data` - 2D array of features (row-major, num_rows x num_features)
    /// * `num_rows` - Number of rows in the data
    /// * `num_features` - Number of features per row
    /// * `option_mask` - Prediction options (see `predict_option` module)
    /// * `training` - Whether this is for training (false for inference)
    ///
    /// # Returns
    /// A vector of prediction values
    ///
    /// # Example
    /// ```no_run
    /// use xgboost_rust::Booster;
    ///
    /// let booster = Booster::load("model.json").unwrap();
    /// let data = vec![1.0, 2.0, 3.0, 4.0]; // 2 rows, 2 features
    /// let predictions = booster.predict(&data, 2, 2, 0, false).unwrap();
    /// ```
    pub fn predict(
        &self,
        data: &[f32],
        num_rows: usize,
        num_features: usize,
        option_mask: u32,
        training: bool,
    ) -> XGBoostResult<Vec<f32>> {
        // Create DMatrix from data
        let mut dmatrix_handle: sys::DMatrixHandle = ptr::null_mut();

        XGBoostError::check_return_value(unsafe {
            sys::XGDMatrixCreateFromMat(
                data.as_ptr(),
                num_rows as u64,
                num_features as u64,
                f32::NAN,
                &mut dmatrix_handle,
            )
        })?;

        // Make prediction
        let mut out_len: u64 = 0;
        let mut out_result: *const f32 = ptr::null();

        XGBoostError::check_return_value(unsafe {
            sys::XGBoosterPredict(
                self.handle,
                dmatrix_handle,
                option_mask as i32,
                0, // ntree_limit (0 means use all trees)
                training as i32,
                &mut out_len,
                &mut out_result,
            )
        })?;

        // Copy results to a Vec
        let results = unsafe {
            std::slice::from_raw_parts(out_result, out_len as usize).to_vec()
        };

        // Free DMatrix
        unsafe {
            sys::XGDMatrixFree(dmatrix_handle);
        }

        Ok(results)
    }

    /// Get the number of features the model expects
    ///
    /// # Returns
    /// The number of features
    pub fn num_features(&self) -> XGBoostResult<usize> {
        let mut out_num_features: u64 = 0;

        XGBoostError::check_return_value(unsafe {
            sys::XGBoosterGetNumFeature(
                self.handle,
                &mut out_num_features,
            )
        })?;

        Ok(out_num_features as usize)
    }

    /// Save the model to a file
    ///
    /// # Arguments
    /// * `path` - Path where to save the model
    ///
    /// # Example
    /// ```no_run
    /// use xgboost_rust::Booster;
    ///
    /// let booster = Booster::load("model.json").unwrap();
    /// booster.save("model_copy.json").unwrap();
    /// ```
    pub fn save<P: AsRef<Path>>(&self, path: P) -> XGBoostResult<()> {
        let path_str = path.as_ref().to_str()
            .ok_or_else(|| XGBoostError {
                description: "Path contains invalid UTF-8 characters".to_string(),
            })?;
        let path_c_str = CString::new(path_str)
            .map_err(|e| XGBoostError {
                description: format!("Path contains NUL byte: {}", e),
            })?;

        XGBoostError::check_return_value(unsafe {
            sys::XGBoosterSaveModel(
                self.handle,
                path_c_str.as_ptr(),
            )
        })
    }
}

impl Drop for Booster {
    fn drop(&mut self) {
        unsafe {
            sys::XGBoosterFree(self.handle);
        }
    }
}

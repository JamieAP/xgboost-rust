# xgboost-rust

Rust bindings for XGBoost, a gradient boosting library for machine learning.

## Features

- **Automatic Binary Download**: Downloads XGBoost binaries at build time from PyPI wheels
- **Cross-Platform**: Supports Linux (x86_64, aarch64), macOS (x86_64, arm64), and Windows (x86_64)
- **Version Control**: Specify XGBoost version via `XGBOOST_VERSION` environment variable
- **Easy to Use**: Simple, safe Rust API wrapping the XGBoost C API

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
xgboost-rust = "0.1.0"
```

## Requirements

- Rust 1.70 or later
- clang/LLVM (for bindgen)
- Internet connection (for downloading XGBoost binaries during build)

### Platform-Specific Requirements

**macOS**:
- `install_name_tool` (included with Xcode Command Line Tools)

**Linux**:
- Optional: `patchelf` (for setting SONAME, but not required)

## Usage

### Basic Example

```rust
use xgboost_rust::{Booster, XGBoostResult};

fn main() -> XGBoostResult<()> {
    // Load a pre-trained model
    let booster = Booster::load("model.json")?;

    // Prepare your data (row-major: num_rows x num_features)
    let data = vec![
        1.0, 2.0, 3.0,  // Row 1
        4.0, 5.0, 6.0,  // Row 2
    ];
    let num_rows = 2;
    let num_features = 3;

    // Make predictions
    let predictions = booster.predict(&data, num_rows, num_features, 0, false)?;
    println!("Predictions: {:?}", predictions);

    Ok(())
}
```

### Advanced Usage

See the [examples](examples/) directory for more examples including:
- Feature contributions (SHAP values)
- Loading models from buffers
- Different prediction options

## XGBoost Version

By default, XGBoost version 3.1.1 is used. To use a different version, set the `XGBOOST_VERSION` environment variable before building:

```bash
export XGBOOST_VERSION=3.0.0
cargo build
```

## How It Works

This crate downloads the appropriate XGBoost Python wheel from PyPI during the build process, extracts the compiled library, and links against it. This approach ensures:

1. No need to compile XGBoost from source
2. Consistent binaries across platforms
3. Easy version management

## Thread Safety

The `Booster` type is **not thread-safe**. For multi-threaded usage:

1. **Recommended**: Create one `Booster` per thread
2. **Alternative**: Wrap in `Arc<Mutex<Booster>>` for shared access

## Examples

Run the basic example:

```bash
cargo run --example basic_usage
```

Run the advanced example:

```bash
cargo run --example advanced_usage
```

## License

Apache-2.0

## Credits

Inspired by [catboost-rust](https://github.com/aryehlev/catboost-rust) and [lightgbm-rust](https://github.com/aryehlev/lightgbm-rust).

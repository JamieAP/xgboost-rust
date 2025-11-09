extern crate bindgen;

use std::env;
use std::path::{Path, PathBuf};
use std::fs;
use std::io;

fn get_xgboost_version() -> String {
    env::var("XGBOOST_VERSION").unwrap_or_else(|_| "3.1.1".to_string())
}

fn get_platform_info() -> (String, String) {
    let target = env::var("TARGET").unwrap();

    // Determine OS
    let os = if target.contains("apple-darwin") {
        "darwin"
    } else if target.contains("linux") {
        "linux"
    } else if target.contains("windows") {
        "windows"
    } else {
        panic!("Unsupported target: {}", target);
    };

    // Determine architecture
    let arch = if target.contains("x86_64") {
        "x86_64"
    } else if target.contains("aarch64") || target.contains("arm64") {
        "aarch64"
    } else if target.contains("i686") || target.contains("i586") {
        "i686"
    } else {
        panic!("Unsupported architecture for target: {}", target);
    };

    (os.to_string(), arch.to_string())
}

fn download_xgboost_headers(out_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let version = get_xgboost_version();

    // Create the include/xgboost directory
    let include_dir = out_dir.join("include/xgboost");
    fs::create_dir_all(&include_dir)?;

    // Download the c_api.h file
    let c_api_url = format!(
        "https://raw.githubusercontent.com/dmlc/xgboost/v{}/include/xgboost/c_api.h",
        version
    );

    println!("cargo:warning=Downloading c_api.h from: {}", c_api_url);

    let response = ureq::get(&c_api_url).call()?;
    let status = response.status();
    if status < 200 || status >= 300 {
        return Err(format!("Failed to download c_api.h: HTTP {}", status).into());
    }

    let c_api_path = include_dir.join("c_api.h");
    let mut file = fs::File::create(&c_api_path)?;
    io::copy(&mut response.into_reader(), &mut file)?;

    // Also download base.h which is referenced by c_api.h
    let base_url = format!(
        "https://raw.githubusercontent.com/dmlc/xgboost/v{}/include/xgboost/base.h",
        version
    );

    println!("cargo:warning=Downloading base.h from: {}", base_url);

    let response = ureq::get(&base_url).call()?;
    let status = response.status();
    if status < 200 || status >= 300 {
        return Err(format!("Failed to download base.h: HTTP {}", status).into());
    }

    let base_path = include_dir.join("base.h");
    let mut file = fs::File::create(&base_path)?;
    io::copy(&mut response.into_reader(), &mut file)?;

    Ok(())
}

fn download_and_extract_wheel(out_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let (os, arch) = get_platform_info();
    let version = get_xgboost_version();

    // Determine wheel filename based on platform
    let wheel_filename = match (os.as_str(), arch.as_str()) {
        ("linux", "x86_64") => format!("xgboost-{}-py3-none-manylinux_2_28_x86_64.whl", version),
        ("linux", "aarch64") => format!("xgboost-{}-py3-none-manylinux_2_28_aarch64.whl", version),
        ("darwin", "x86_64") => format!("xgboost-{}-py3-none-macosx_10_15_x86_64.whl", version),
        ("darwin", "aarch64") => format!("xgboost-{}-py3-none-macosx_12_0_arm64.whl", version),
        ("windows", "x86_64") => format!("xgboost-{}-py3-none-win_amd64.whl", version),
        _ => return Err(format!("Unsupported platform: {}-{}", os, arch).into()),
    };

    let download_url = format!(
        "https://files.pythonhosted.org/packages/py3/x/xgboost/{}",
        wheel_filename
    );

    println!("cargo:warning=Downloading XGBoost wheel from: {}", download_url);

    // Download the wheel
    let wheel_dir = out_dir.join("wheel");
    fs::create_dir_all(&wheel_dir)?;
    let wheel_path = wheel_dir.join(&wheel_filename);

    let response = ureq::get(&download_url).call()?;
    let status = response.status();
    if status < 200 || status >= 300 {
        return Err(format!("Failed to download wheel: HTTP {}", status).into());
    }

    let mut wheel_file = fs::File::create(&wheel_path)?;
    io::copy(&mut response.into_reader(), &mut wheel_file)?;
    drop(wheel_file);

    println!("cargo:warning=Extracting wheel: {}", wheel_path.display());

    // Extract the wheel (it's a ZIP file)
    let file = fs::File::open(&wheel_path)?;
    let mut archive = zip::ZipArchive::new(file)?;

    // Create libs directory
    let lib_dir = out_dir.join("libs");
    fs::create_dir_all(&lib_dir)?;

    // Determine library filename based on OS
    let lib_filename = match os.as_str() {
        "windows" => "xgboost.dll",
        "darwin" => "libxgboost.dylib",
        _ => "libxgboost.so",
    };

    // Search for the library file in the wheel
    let mut found = false;
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let file_path = file.name().to_string();

        // Look for the library file (usually in xgboost/lib/)
        if file_path.ends_with(lib_filename) {
            println!("cargo:warning=Found library at: {}", file_path);
            let dest_path = lib_dir.join(lib_filename);
            let mut dest = fs::File::create(&dest_path)?;
            io::copy(&mut file, &mut dest)?;
            found = true;
            break;
        }
    }

    if !found {
        return Err(format!("Library file {} not found in wheel", lib_filename).into());
    }

    println!("cargo:warning=Successfully extracted XGBoost library to: {}", lib_dir.display());

    Ok(())
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let xgb_include_root = out_dir.join("include");

    // Download the headers
    if let Err(e) = download_xgboost_headers(&out_dir) {
        eprintln!("Failed to download XGBoost headers: {}", e);
        panic!("Cannot proceed without headers");
    }

    // Download and extract the wheel
    if let Err(e) = download_and_extract_wheel(&out_dir) {
        eprintln!("Failed to download and extract wheel: {}", e);
        panic!("Cannot proceed without compiled library");
    }

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", xgb_include_root.display()))
        // Generate bindings for XGB and XGD functions (Booster and DMatrix)
        .allowlist_function("XGB.*")
        .allowlist_function("XGD.*")
        // Allowlist the main types we need
        .allowlist_type("BoosterHandle")
        .allowlist_type("DMatrixHandle")
        .allowlist_type("bst_ulong")
        .size_t_is_usize(true)
        .generate()
        .expect("Unable to generate bindings.");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Couldn't write bindings.");

    // Get platform info
    let (os, _arch) = get_platform_info();

    // Determine the library filename based on the OS
    let lib_filename = match os.as_str() {
        "windows" => "xgboost.dll",
        "darwin" => "libxgboost.dylib",
        _ => "libxgboost.so",
    };

    // Copy the library from OUT_DIR/libs to the final target directory
    let lib_source_path = out_dir.join("libs").join(lib_filename);

    // Find the final output directory (e.g., target/release)
    let target_dir = out_dir
        .ancestors()
        .find(|p| p.ends_with("target"))
        .unwrap()
        .join(env::var("PROFILE").unwrap());

    let lib_dest_path = target_dir.join(lib_filename);
    fs::copy(&lib_source_path, &lib_dest_path)
        .expect("Failed to copy library to target directory");

    // On macOS/Linux, change the install name/soname to use @loader_path/$ORIGIN
    if os == "darwin" {
        use std::process::Command;
        let _ = Command::new("install_name_tool")
            .arg("-id")
            .arg(format!("@loader_path/{}", lib_filename))
            .arg(&lib_source_path)
            .status();
        let _ = Command::new("install_name_tool")
            .arg("-id")
            .arg(format!("@loader_path/{}", lib_filename))
            .arg(&lib_dest_path)
            .status();
    } else if os == "linux" {
        use std::process::Command;
        // Use patchelf to set soname (if available)
        let _ = Command::new("patchelf")
            .arg("--set-soname")
            .arg(&lib_filename)
            .arg(&lib_source_path)
            .output();
        let _ = Command::new("patchelf")
            .arg("--set-soname")
            .arg(&lib_filename)
            .arg(&lib_dest_path)
            .output();
    }

    // Set the library search path for the build-time linker
    let lib_search_path = out_dir.join("libs");
    println!(
        "cargo:rustc-link-search=native={}",
        lib_search_path.display()
    );

    // Set the rpath for the run-time linker based on the OS
    match os.as_str() {
        "darwin" => {
            // For macOS, add multiple rpath entries for IDE compatibility
            println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
            println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../..");
            println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
            println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/../..");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_search_path.display());
            // Add the target directory to rpath as well
            if let Some(target_root) = out_dir.ancestors().find(|p| p.ends_with("target")) {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}/debug", target_root.display());
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}/release", target_root.display());
            }
        },
        "linux" => {
            // For Linux, use $ORIGIN
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../..");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_search_path.display());
            // Add the target directory to rpath as well
            if let Some(target_root) = out_dir.ancestors().find(|p| p.ends_with("target")) {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}/debug", target_root.display());
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}/release", target_root.display());
            }
        },
        _ => {} // No rpath needed for Windows
    }

    println!("cargo:rustc-link-lib=dylib=xgboost");
}

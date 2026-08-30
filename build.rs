//! Compiles the CUDA kernels and links the driver API.
//!
//! Only runs when the `cuda` feature is on, so a machine without a toolkit
//! still builds and tests everything else. `CLAUDE.md` scopes this project to
//! NVIDIA and sm_120, so there is no portability layer here on purpose.
//!
//! We target the **driver API** (`libcuda.so`), not the runtime API. That means
//! no `libcudart` to link or ship, and PTX loaded at run time rather than
//! kernels baked into the binary — which matters because sm_120 support is new
//! enough that being able to swap the PTX without relinking is useful.

use std::path::{Path, PathBuf};

/// Locate `nvcc` without depending on how the shell was started.
///
/// A build must not depend on an interactive `PATH`: cargo may be invoked from
/// an editor, a non-login shell, or a hook, and on this machine the toolkit
/// lives in `/usr/local/cuda` while `PATH` picks it up only from `.bashrc`.
/// `CUDA_PATH` and `CUDA_HOME` are the conventional overrides.
fn find_nvcc() -> PathBuf {
    for var in ["CUDA_PATH", "CUDA_HOME"] {
        println!("cargo:rerun-if-env-changed={var}");
        if let Ok(root) = std::env::var(var) {
            let candidate = Path::new(&root).join("bin/nvcc");
            if candidate.exists() {
                return candidate;
            }
        }
    }
    for candidate in ["/usr/local/cuda/bin/nvcc", "/opt/cuda/bin/nvcc"] {
        if Path::new(candidate).exists() {
            return PathBuf::from(candidate);
        }
    }
    // Fall back to PATH and let the error message do the explaining.
    PathBuf::from("nvcc")
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    let src = "kernels/kernels.cu";
    println!("cargo:rerun-if-changed={src}");

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR is always set by cargo");
    let ptx = format!("{out_dir}/kernels.ptx");

    // --fmad=false is load-bearing, not a tuning flag. nvcc contracts `a * b +
    // c` into an FMA by default, which rounds once instead of twice and would
    // make the kernel disagree with the CPU oracle in the last bits. Rust does
    // not contract, so switching it off here is what lets the first kernel be
    // compared for exact equality rather than a tolerance.
    let nvcc = find_nvcc();
    let status = std::process::Command::new(&nvcc)
        .args([
            "-ptx",
            "-arch=sm_120",
            "--fmad=false",
            "-O3",
            "-lineinfo",
            src,
            "-o",
            &ptx,
        ])
        .status();

    match status {
        Ok(s) if s.success() => {}
        Ok(s) => panic!(
            "nvcc failed with {s} compiling {src}.\n\
             If it rejects -arch=sm_120, the toolkit predates Blackwell support \
             (CUDA 12.8+ is required)."
        ),
        Err(e) => panic!(
            "could not run {} ({e}). The `cuda` feature needs the CUDA toolkit;              set CUDA_PATH if it lives somewhere other than /usr/local/cuda.",
            nvcc.display()
        ),
    }

    // libcuda.so ships with the driver, not the toolkit. On WSL2 it lives in a
    // WSL-specific directory that is not on the default search path; the stubs
    // directory is the fallback for link-time resolution.
    for dir in [
        "/usr/lib/wsl/lib",
        "/usr/local/cuda/lib64/stubs",
        "/usr/lib/x86_64-linux-gnu",
    ] {
        if Path::new(dir).exists() {
            println!("cargo:rustc-link-search=native={dir}");
        }
    }
    println!("cargo:rustc-link-lib=dylib=cuda");
}

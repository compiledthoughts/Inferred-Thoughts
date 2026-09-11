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
    // The whole directory, not just the entry point: kernels.cu includes the
    // per-family `.cuh` files, and cargo scans a directory for any change.
    println!("cargo:rerun-if-changed=kernels");

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR is always set by cargo");
    let ptx = format!("{out_dir}/kernels.ptx");

    // **The target, and why it can be `sm_120a`.** `sm_120` is forwards
    // compatible, so it leaves out instructions later architectures will not
    // have -- and the FP4 tensor-core `mma` is one of them. PTX ISA 9.4 says
    // `.e2m1` mma and the `.block_scale` qualifiers require `sm_120a`, and lists
    // `.kind::mxf4nvf4` only for `sm_120a` and `sm_121a`; CUDA 12.8's
    // `ptxas -arch=sm_120` rejects all of them, and accepts them for `sm_120a`.
    // llama.cpp compiles Blackwell as `120a` for the same reason
    // (`ggml-cuda/CMakeLists.txt`: "12X is forwards-compatible, 12Xa is not").
    //
    // `sm_120a` is the default. `INFERRED_SM_ARCH=sm_120` restores the
    // forwards-compatible PTX, without the NVFP4 kernels behind
    // `INFERRED_NVFP4_BLOCK_SCALE`. The int8 and fp16 tensor-core ceilings read
    // the same on both targets (BENCHMARKS-v2 12-09 (FP4 ceiling)).
    println!("cargo:rerun-if-env-changed=INFERRED_SM_ARCH");
    let arch = std::env::var("INFERRED_SM_ARCH").unwrap_or_else(|_| "sm_120a".to_string());
    if arch != "sm_120" && arch != "sm_120a" {
        panic!("INFERRED_SM_ARCH must be sm_120 or sm_120a, got {arch}");
    }
    let arch_flag = format!("-arch={arch}");
    // Tells the tests whether the NVFP4 kernels exist in this PTX, so a test of
    // one skips on an `sm_120` build instead of failing to find the kernel.
    println!("cargo::rustc-check-cfg=cfg(nvfp4_block_scale)");
    if arch == "sm_120a" {
        println!("cargo:rustc-cfg=nvfp4_block_scale");
    }

    // --fmad=false is load-bearing, not a tuning flag. nvcc contracts `a * b +
    // c` into an FMA by default, which rounds once instead of twice and would
    // make the kernel disagree with the CPU oracle in the last bits. Rust does
    // not contract, so switching it off here is what lets the first kernel be
    // compared for exact equality rather than a tolerance.
    let mut args: Vec<&str> = vec!["-ptx", &arch_flag, "--fmad=false", "-O3", "-lineinfo"];
    if arch == "sm_120a" {
        args.push("-DINFERRED_NVFP4_BLOCK_SCALE");
    }
    args.extend([src, "-o", &ptx]);

    let nvcc = find_nvcc();
    let status = std::process::Command::new(&nvcc).args(&args).status();

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

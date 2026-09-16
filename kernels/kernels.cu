// CUDA kernels, compiled to PTX by build.rs and loaded through the driver API.
//
// Compiled with --fmad=false. That is not a tuning choice: nvcc contracts
// `a * b + c` into an FMA by default, rounding once where the CPU rounds twice,
// and the first kernel here is meant to be compared against the CPU oracle for
// *exact* equality. Every kernel in this file must keep that property or say
// loudly that it does not.

#include <cuda_fp16.h>

// The kernels live in per-family include files, in dependency order: later
// files use helpers and defines from earlier ones. One compile unit, one PTX.
#include "core.cuh"
#include "q8_0.cuh"
#include "kquants.cuh"
#include "iq4_xs.cuh"
#include "moe.cuh"
#include "nvfp4.cuh"
#include "mma_int8.cuh"
#include "experiments.cuh"
#include "attention.cuh"
#include "attention_mma.cuh"
#include "attention_decode.cuh"
#include "gdn.cuh"
#include "qwen4exp.cuh"
#include "qsa.cuh"
#include "tier3.cuh"
#include "f32.cuh"
#include "diagnostics.cuh"

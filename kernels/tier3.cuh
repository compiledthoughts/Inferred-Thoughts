// tier3.cuh -- the SSD expert tier's device-side bookkeeping.
//
// Included by kernels.cu, which pulls every file into one compile unit and one
// PTX module in dependency order. Kernel names are the PTX symbols the backend
// looks up, so which file a kernel lives in changes nothing at run time.
//
// Launched only while the expert pool is oversubscribed; a model that fits never
// reaches it.
#pragma once

extern "C" {

// Apply a resolve's queued writes: pointer-table entries and residency flags.
//
// `patches` holds `n` triples of unsigned 64-bit values: a device address, the
// value to store there, and its width in bytes (8 for a table entry, 4 for a
// flag). One thread per patch. The host has already dropped duplicate addresses,
// keeping the last write, so no two threads write the same place.
//
// Replaces four host-to-device copies per fetched expert -- the victim's entry
// and flag, the fetched expert's entry and flag -- with one copy of the list and
// this launch per layer boundary. Measured before: ~1,270 of those copies a 125B
// token, 56.3 ms of a 315 ms token.
extern "C" __global__ void apply_patches(int n, const unsigned long long *__restrict__ patches) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned long long *p = patches + 3 * (size_t)i;
    if (p[2] == 8) {
        *(unsigned long long *)p[0] = p[1];
    } else {
        *(int *)p[0] = (int)p[1];
    }
}

}  // extern "C"

//! GGUF container parsing.
//!
//! Layout, quoted from the header comment of `ggml/include/gguf.h`:
//!
//! ```text
//! 1. File magic "GGUF" (4 bytes).
//! 2. File version (uint32_t).
//! 3. Number of ggml tensors in file (int64_t).
//! 4. Number of key-value-pairs in file (int64_t).
//! 5. For each KV pair: key (string), value type, [array elem type + count], value.
//! 6. For each tensor: name (string), n_dims (uint32_t), dims (int64_t each),
//!    type (ggml_type), offset into the data blob (uint64_t).
//! 7. The tensor data binary blob (optional, aligned).
//! ```
//!
//! Strings are a uint64 length followed by bytes with no NUL terminator.
//! Tensor offsets are relative to the start of the data blob, not the file.
//!
//! This module parses only. It records where each tensor's bytes live; it does
//! not dequantize them.

mod metadata;
mod reader;
mod types;

pub use metadata::{Array, Metadata, Value};
pub use types::{GgmlType, MAX_DIMS};

use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::error::{Error, Result};
use reader::Cursor;

pub const MAGIC: [u8; 4] = *b"GGUF";
pub const VERSION: u32 = 3;
/// `GGUF_DEFAULT_ALIGNMENT` from `ggml/include/gguf.h`, used when the file does
/// not carry a `general.alignment` key.
pub const DEFAULT_ALIGNMENT: u64 = 32;
pub const KEY_ALIGNMENT: &str = "general.alignment";

/// Where one tensor's bytes live and how to interpret them.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// Dimensions in ggml order, fastest-varying first: `dims[0]` is the row
    /// length. A 2-D weight `[n_in, n_out]` is stored with `dims[0] == n_in`.
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Offset from the start of the data blob.
    pub offset: u64,
    /// Offset from the start of the file. Precomputed so callers never have to
    /// remember to add the blob base.
    pub file_offset: u64,
    /// Size of this tensor's data in bytes.
    pub n_bytes: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }

    /// ggml prints shapes padded to 4 dimensions with 1s.
    pub fn padded_dims(&self) -> [u64; MAX_DIMS as usize] {
        let mut out = [1u64; MAX_DIMS as usize];
        for (i, d) in self.dims.iter().enumerate() {
            out[i] = *d;
        }
        out
    }
}

pub struct GgufFile {
    pub path: PathBuf,
    pub version: u32,
    pub alignment: u64,
    pub metadata: Metadata,
    pub tensors: Vec<TensorInfo>,
    /// File offset at which the tensor data blob begins.
    pub data_offset: u64,
    /// Wrapped so `Drop` can release the mapping *before* evicting the file
    /// from the page cache — pages a live mapping holds cannot be evicted, so
    /// the order is the whole point. See the `Drop` impl.
    mmap: std::mem::ManuallyDrop<Mmap>,
}

impl Drop for GgufFile {
    /// Hand the model file back to the system.
    ///
    /// **Because the page cache is the larger half of this program's host
    /// footprint, and nothing else was going to.** Loading the 35B reads
    /// 18.8 GB through the mapping; the pages stay cached after the process
    /// exits, and WSL does not return them to Windows. Observed repeatedly at
    /// 15-16 GB of cache with no process running, on a 32 GB machine.
    ///
    /// The expert cache already evicts each tensor as it places it, but that
    /// covers only pooled weights on the CUDA path with a model path supplied.
    /// It missed the non-expert weights, every model without experts, and every
    /// test — which construct `Cuda` directly and never set a path. Here it is
    /// unconditional: whoever opened the file, on whatever backend, gets the
    /// cache dropped when they are done with it.
    ///
    /// `madvise` is not a substitute: on a private file mapping it unmaps the
    /// pages from this process while the kernel keeps them, which is why an
    /// earlier attempt reported 757 MB of process against 16.7 GB of cache.
    ///
    /// Best-effort. A failure leaves the cache, which is the behaviour before
    /// this existed.
    fn drop(&mut self) {
        // SAFETY: `mmap` is dropped exactly once, here, and nothing reads it
        // afterwards — `self` is being destroyed.
        unsafe { std::mem::ManuallyDrop::drop(&mut self.mmap) };
        drop_page_cache(&self.path);
    }
}

/// Evict a whole file from the page cache.
///
/// Addresses the file rather than a descriptor, so a fresh `open` reaches the
/// same pages the mapping used. A no-op on Windows; see
/// [`crate::platform::release_range`] for why that is acceptable.
fn drop_page_cache(path: &Path) {
    crate::platform::release_file(path);
}

impl GgufFile {
    /// Where the mapping begins, so a byte range inside it can be turned back
    /// into a file offset.
    ///
    /// **Needed to bound host memory.** Placement reads the whole expert pool
    /// through this mapping, which fills the page cache with the model — 16 GiB
    /// on the 35B — and WSL does not hand that back. `posix_fadvise` can evict
    /// it as placement goes, but it addresses the *file*, so a pointer into the
    /// mapping has to be convertible to an offset. Nothing else needs this and
    /// nothing should read the mapping through it.
    pub fn map_base(&self) -> usize {
        self.mmap.as_ptr() as usize
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|source| Error::Io {
            path: path.display().to_string(),
            source,
        })?;
        // SAFETY: the mapping is read-only and the file is not modified by this
        // process. A concurrent external truncation would be UB; that is the
        // same contract llama.cpp accepts for mmap-backed model loading.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|source| Error::Io {
            path: path.display().to_string(),
            source,
        })?;

        Self::parse(path.to_path_buf(), mmap)
    }

    fn parse(path: PathBuf, mmap: Mmap) -> Result<Self> {
        let mut cur = Cursor::new(&mmap[..]);

        let magic: [u8; 4] = cur.take(4)?.try_into().expect("take(4) returned 4 bytes");
        if magic != MAGIC {
            return Err(Error::BadMagic { found: magic });
        }

        let version = cur.u32_()?;
        if version != VERSION {
            return Err(Error::UnsupportedVersion {
                found: version,
                supported: VERSION,
            });
        }

        // Both counts are int64 on disk. Negative means a corrupt header.
        let n_tensors = cur.i64_()?;
        let n_kv = cur.i64_()?;
        if n_tensors < 0 || n_kv < 0 {
            return Err(Error::Truncated {
                offset: cur.pos(),
                needed: 0,
                len: cur.len(),
            });
        }

        let metadata = Metadata::read(&mut cur, n_kv as u64)?;

        let alignment = match metadata.get(KEY_ALIGNMENT) {
            // Spec-sanctioned default, not an invented constant: gguf.h says
            // GGUF_DEFAULT_ALIGNMENT applies when the key is absent.
            None => DEFAULT_ALIGNMENT,
            Some(_) => {
                let a = metadata.get_u64(KEY_ALIGNMENT)?;
                if a == 0 || !a.is_power_of_two() {
                    return Err(Error::BadAlignment(a));
                }
                a
            }
        };

        let mut tensors = Vec::with_capacity(n_tensors as usize);
        let mut seen = std::collections::HashSet::new();

        for _ in 0..n_tensors {
            let name = cur.string("tensor name")?;
            if !seen.insert(name.clone()) {
                return Err(Error::DuplicateTensor { name });
            }

            let n_dims = cur.u32_()?;
            if n_dims > MAX_DIMS {
                return Err(Error::TooManyDims {
                    name,
                    n_dims,
                    max: MAX_DIMS,
                });
            }

            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(cur.i64_()? as u64);
            }

            let ty_code = cur.u32_()?;
            let ty = types::tensor_type_from_u32(&name, ty_code)?;
            let offset = cur.u64_()?;

            // ggml stores rows contiguously, so it is the row length that must
            // be block-aligned, not merely the total element count.
            let ne0 = dims.first().copied().unwrap_or(1);
            let block_size = ty.block_size();
            if ne0 % block_size != 0 {
                return Err(Error::NotBlockAligned {
                    name,
                    ne0,
                    block_size,
                    ty: ty.name(),
                });
            }

            if offset % alignment != 0 {
                return Err(Error::MisalignedTensor {
                    name,
                    offset,
                    alignment,
                });
            }

            let n_elements: u64 = dims.iter().product();
            let n_bytes = ty.n_bytes(n_elements);

            tensors.push(TensorInfo {
                name,
                dims,
                ty,
                offset,
                file_offset: 0, // filled in once the blob base is known
                n_bytes,
            });
        }

        // The data blob starts at the next alignment boundary after the header.
        let data_offset = align_up(cur.pos() as u64, alignment);

        let len = mmap.len() as u64;
        for t in &mut tensors {
            t.file_offset = data_offset + t.offset;
            let end = t.file_offset.checked_add(t.n_bytes).ok_or_else(|| {
                Error::TensorOutOfBounds {
                    name: t.name.clone(),
                    offset: t.file_offset,
                    size: t.n_bytes,
                    len,
                }
            })?;
            if end > len {
                return Err(Error::TensorOutOfBounds {
                    name: t.name.clone(),
                    offset: t.file_offset,
                    size: t.n_bytes,
                    len,
                });
            }
        }

        Ok(Self {
            path,
            version,
            alignment,
            metadata,
            tensors,
            data_offset,
            mmap: std::mem::ManuallyDrop::new(mmap),
        })
    }

    pub fn file_size(&self) -> u64 {
        self.mmap.len() as u64
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// The raw, still-quantized bytes of one tensor.
    ///
    /// Bounds were validated at parse time, so this cannot slice out of range.
    pub fn tensor_bytes(&self, info: &TensorInfo) -> &[u8] {
        let start = info.file_offset as usize;
        let end = start + info.n_bytes as usize;
        &self.mmap[start..end]
    }

    /// Total bytes of tensor data, i.e. the model's on-disk weight size.
    pub fn total_tensor_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.n_bytes).sum()
    }

    pub fn total_parameters(&self) -> u64 {
        self.tensors.iter().map(|t| t.n_elements()).sum()
    }
}

fn align_up(offset: u64, alignment: u64) -> u64 {
    offset.div_ceil(alignment) * alignment
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_up_rounds_to_boundary() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
    }

    #[test]
    fn padded_dims_matches_ggml_display() {
        let t = TensorInfo {
            name: "x".into(),
            dims: vec![4096, 11008],
            ty: GgmlType::Q8_0,
            offset: 0,
            file_offset: 0,
            n_bytes: 0,
        };
        assert_eq!(t.padded_dims(), [4096, 11008, 1, 1]);
        assert_eq!(t.n_elements(), 4096 * 11008);
    }
}

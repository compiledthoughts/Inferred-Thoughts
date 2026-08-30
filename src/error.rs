use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("io error on {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("not a GGUF file: expected magic \"GGUF\", found {found:?}")]
    BadMagic { found: [u8; 4] },

    #[error("unsupported GGUF version {found} (this parser implements version {supported})")]
    UnsupportedVersion { found: u32, supported: u32 },

    #[error("unexpected end of file: needed {needed} bytes at offset {offset}, file is {len} bytes")]
    Truncated {
        offset: usize,
        needed: usize,
        len: usize,
    },

    #[error("invalid UTF-8 in {context} at offset {offset}")]
    InvalidUtf8 { context: String, offset: usize },

    #[error("string length {len} at offset {offset} exceeds remaining file size")]
    StringTooLong { len: u64, offset: usize },

    #[error("unknown GGUF metadata value type {ty} for key \"{key}\"")]
    UnknownValueType { key: String, ty: u32 },

    // llama.cpp rejects arrays whose element type is itself ARRAY
    // (ggml/src/gguf.cpp, the `case GGUF_TYPE_ARRAY:` fallthrough to error).
    // GGUF arrays are exactly one level deep.
    #[error("key \"{key}\" is an array of arrays, which GGUF does not permit")]
    NestedArray { key: String },

    #[error("duplicate metadata key \"{key}\"")]
    DuplicateKey { key: String },

    #[error("missing required metadata key \"{0}\"")]
    MissingKey(String),

    #[error("metadata key \"{key}\" has type {actual}, expected {expected}")]
    TypeMismatch {
        key: String,
        expected: &'static str,
        actual: &'static str,
    },

    #[error("unknown ggml tensor type {ty} for tensor \"{name}\"")]
    UnknownTensorType { name: String, ty: u32 },

    #[error("tensor \"{name}\" has type {ty}, which was removed from ggml and cannot be read")]
    RemovedTensorType { name: String, ty: &'static str },

    #[error("tensor \"{name}\" has {n_dims} dimensions, maximum is {max}")]
    TooManyDims {
        name: String,
        n_dims: u32,
        max: u32,
    },

    #[error(
        "tensor \"{name}\": row length {ne0} is not a multiple of block size {block_size} for type {ty}"
    )]
    NotBlockAligned {
        name: String,
        ne0: u64,
        block_size: u64,
        ty: &'static str,
    },

    #[error("duplicate tensor name \"{name}\"")]
    DuplicateTensor { name: String },

    #[error("tensor \"{name}\" offset {offset} is not a multiple of alignment {alignment}")]
    MisalignedTensor {
        name: String,
        offset: u64,
        alignment: u64,
    },

    #[error(
        "tensor \"{name}\" data runs past end of file: offset {offset} + {size} bytes exceeds {len}"
    )]
    TensorOutOfBounds {
        name: String,
        offset: u64,
        size: u64,
        len: u64,
    },

    #[error("general.alignment is {0}, which is not a power of two")]
    BadAlignment(u64),

    #[error("dequantization of {ty} is not implemented yet")]
    UnsupportedQuantType { ty: &'static str },

    #[error(
        "dequantizing {n} elements of {ty} needs {expected} bytes of input, got {got}"
    )]
    QuantSizeMismatch {
        ty: &'static str,
        n: usize,
        expected: u64,
        got: usize,
    },

    #[error("cannot dequantize {n} elements of {ty}: not a multiple of block size {block_size}")]
    QuantNotBlockAligned {
        ty: &'static str,
        n: usize,
        block_size: u64,
    },

    #[error("output buffer holds {got} elements, need exactly {expected}")]
    OutputLenMismatch { expected: usize, got: usize },

    #[error("tokenizer.ggml.model is \"{model}\"; only \"gpt2\" (BPE) is implemented")]
    UnsupportedTokenizerModel { model: String },

    #[error(
        "tokenizer.ggml.pre is \"{pre}\"; this build implements only the pre-tokenizers \
         our target models use ({supported})"
    )]
    UnsupportedPreTokenizer {
        pre: String,
        supported: &'static str,
    },

    #[error("merge {index} is not a \"left right\" pair: {text:?}")]
    BadMerge { index: usize, text: String },

    #[error(
        "tokenizer.ggml.token_type has {n_types} entries but tokenizer.ggml.tokens has {n_tokens}"
    )]
    TokenTypeLengthMismatch { n_types: usize, n_tokens: usize },

    #[error("token id {id} is out of range for a vocabulary of {vocab_size}")]
    TokenOutOfRange { id: u32, vocab_size: usize },

    #[error("model is missing required tensor \"{name}\"")]
    MissingTensor { name: String },

    #[error("tensor \"{name}\" has shape {got:?}, expected {expected:?}")]
    TensorShapeMismatch {
        name: String,
        expected: Vec<u64>,
        got: Vec<u64>,
    },

    #[error(
        "architecture \"{arch}\" is not implemented; this build supports: {supported}"
    )]
    UnsupportedArchitecture {
        arch: String,
        supported: &'static str,
    },

    #[error(
        "inconsistent architecture: {what} ({detail})"
    )]
    InconsistentArchitecture {
        what: &'static str,
        detail: String,
    },

    #[error(
        "position {pos} exceeds the KV cache context of {n_ctx}; raise --ctx"
    )]
    ContextOverflow { pos: usize, n_ctx: usize },

    #[error("cannot apply a chat template to this model: {detail}")]
    UnsupportedChatTemplate { detail: String },

    #[error("cuda: {what} failed: {detail}")]
    Cuda {
        what: &'static str,
        detail: String,
    },
}

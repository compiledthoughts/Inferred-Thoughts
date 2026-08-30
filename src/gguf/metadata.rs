//! The GGUF key/value metadata store.
//!
//! Value type codes are from `enum gguf_type` in `ggml/include/gguf.h`.
//! Arrays are exactly one level deep: llama.cpp rejects an array whose element
//! type is `GGUF_TYPE_ARRAY` (`ggml/src/gguf.cpp`), so we do too.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::gguf::reader::Cursor;

/// A single metadata value. Arrays keep their element type rather than
/// degrading to `Vec<Value>`, because later stages read
/// `tokenizer.ggml.tokens` as 250k strings and want them contiguous.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Array),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Array {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
    String(Vec<String>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F64(Vec<f64>),
}

impl Array {
    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::I8(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::Bool(v) => v.len(),
            Self::String(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::F64(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The name ggml prints for the element type.
    pub fn elem_type_name(&self) -> &'static str {
        match self {
            Self::U8(_) => "u8",
            Self::I8(_) => "i8",
            Self::U16(_) => "u16",
            Self::I16(_) => "i16",
            Self::U32(_) => "u32",
            Self::I32(_) => "i32",
            Self::F32(_) => "f32",
            Self::Bool(_) => "bool",
            Self::String(_) => "str",
            Self::U64(_) => "u64",
            Self::I64(_) => "i64",
            Self::F64(_) => "f64",
        }
    }

    /// Render element `i` for display. Used by `inspect` when truncating.
    pub fn elem_to_string(&self, i: usize) -> String {
        match self {
            Self::U8(v) => v[i].to_string(),
            Self::I8(v) => v[i].to_string(),
            Self::U16(v) => v[i].to_string(),
            Self::I16(v) => v[i].to_string(),
            Self::U32(v) => v[i].to_string(),
            Self::I32(v) => v[i].to_string(),
            Self::F32(v) => v[i].to_string(),
            Self::Bool(v) => v[i].to_string(),
            Self::String(v) => format!("{:?}", v[i]),
            Self::U64(v) => v[i].to_string(),
            Self::I64(v) => v[i].to_string(),
            Self::F64(v) => v[i].to_string(),
        }
    }
}

impl Value {
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::U8(_) => "u8",
            Self::I8(_) => "i8",
            Self::U16(_) => "u16",
            Self::I16(_) => "i16",
            Self::U32(_) => "u32",
            Self::I32(_) => "i32",
            Self::F32(_) => "f32",
            Self::Bool(_) => "bool",
            Self::String(_) => "str",
            Self::U64(_) => "u64",
            Self::I64(_) => "i64",
            Self::F64(_) => "f64",
            Self::Array(_) => "arr",
        }
    }

    /// Read one value of the given type code. `is_array` and `n` come from the
    /// caller because GGUF encodes the array header before the element type.
    fn read(cur: &mut Cursor<'_>, key: &str, ty: u32, is_array: bool, n: u64) -> Result<Value> {
        if !is_array {
            return Ok(match ty {
                0 => Value::U8(cur.u8_()?),
                1 => Value::I8(cur.i8_()?),
                2 => Value::U16(cur.u16_()?),
                3 => Value::I16(cur.i16_()?),
                4 => Value::U32(cur.u32_()?),
                5 => Value::I32(cur.i32_()?),
                6 => Value::F32(cur.f32_()?),
                7 => Value::Bool(cur.bool_()?),
                8 => Value::String(cur.string(key)?),
                9 => {
                    return Err(Error::NestedArray {
                        key: key.to_string(),
                    });
                }
                10 => Value::U64(cur.u64_()?),
                11 => Value::I64(cur.i64_()?),
                12 => Value::F64(cur.f64_()?),
                _ => {
                    return Err(Error::UnknownValueType {
                        key: key.to_string(),
                        ty,
                    });
                }
            });
        }

        // Guard the allocation: refuse a count that cannot fit in the file even
        // at one byte per element, so a corrupt header cannot request a huge
        // reservation before the read fails.
        let remaining = (cur.len() - cur.pos()) as u64;
        if n > remaining {
            return Err(Error::Truncated {
                offset: cur.pos(),
                needed: n as usize,
                len: cur.len(),
            });
        }
        let n = n as usize;

        macro_rules! arr {
            ($variant:ident, $read:ident) => {{
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(cur.$read()?);
                }
                Array::$variant(v)
            }};
        }

        Ok(Value::Array(match ty {
            0 => arr!(U8, u8_),
            1 => arr!(I8, i8_),
            2 => arr!(U16, u16_),
            3 => arr!(I16, i16_),
            4 => arr!(U32, u32_),
            5 => arr!(I32, i32_),
            6 => arr!(F32, f32_),
            7 => arr!(Bool, bool_),
            8 => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(cur.string(key)?);
                }
                Array::String(v)
            }
            9 => {
                return Err(Error::NestedArray {
                    key: key.to_string(),
                });
            }
            10 => arr!(U64, u64_),
            11 => arr!(I64, i64_),
            12 => arr!(F64, f64_),
            _ => {
                return Err(Error::UnknownValueType {
                    key: key.to_string(),
                    ty,
                });
            }
        }))
    }
}

/// Insertion-ordered metadata. Order is preserved so `inspect` output can be
/// compared against `gguf_dump.py`, which lists keys in file order.
#[derive(Debug, Default, Clone)]
pub struct Metadata {
    entries: Vec<(String, Value)>,
    index: HashMap<String, usize>,
}

impl Metadata {
    pub(crate) fn read(cur: &mut Cursor<'_>, n_kv: u64) -> Result<Self> {
        let mut md = Metadata::default();
        for _ in 0..n_kv {
            let key = cur.string("metadata key")?;

            // Layout per the gguf.h header comment: value type, then if that
            // type is ARRAY(9), the element type and element count follow.
            let mut ty = cur.u32_()?;
            let mut is_array = false;
            let mut n = 0u64;
            if ty == 9 {
                is_array = true;
                ty = cur.u32_()?;
                n = cur.u64_()?;
            }

            let value = Value::read(cur, &key, ty, is_array, n)?;
            md.insert(key, value)?;
        }
        Ok(md)
    }

    fn insert(&mut self, key: String, value: Value) -> Result<()> {
        if self.index.contains_key(&key) {
            return Err(Error::DuplicateKey { key });
        }
        self.index.insert(key.clone(), self.entries.len());
        self.entries.push((key, value));
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.index.get(key).map(|&i| &self.entries[i].1)
    }

    fn require(&self, key: &str) -> Result<&Value> {
        self.get(key).ok_or_else(|| Error::MissingKey(key.to_string()))
    }

    fn mismatch(key: &str, expected: &'static str, found: &Value) -> Error {
        Error::TypeMismatch {
            key: key.to_string(),
            expected,
            actual: found.type_name(),
        }
    }
}

/// Typed accessors. Each returns a named error rather than a default, per the
/// project rule against inventing architecture constants.
///
/// The integer accessors widen: a key stored as U32 is readable as u64, because
/// GGUF writers are inconsistent about integer width for the same logical key
/// across model families. They do not narrow, and they do not cross the
/// signed/unsigned boundary silently.
impl Metadata {
    pub fn get_string(&self, key: &str) -> Result<&str> {
        match self.require(key)? {
            Value::String(s) => Ok(s),
            other => Err(Self::mismatch(key, "str", other)),
        }
    }

    pub fn get_bool(&self, key: &str) -> Result<bool> {
        match self.require(key)? {
            Value::Bool(b) => Ok(*b),
            other => Err(Self::mismatch(key, "bool", other)),
        }
    }

    pub fn get_f32(&self, key: &str) -> Result<f32> {
        match self.require(key)? {
            Value::F32(v) => Ok(*v),
            Value::F64(v) => Ok(*v as f32),
            other => Err(Self::mismatch(key, "f32", other)),
        }
    }

    pub fn get_u32(&self, key: &str) -> Result<u32> {
        let v = self.get_u64(key)?;
        u32::try_from(v).map_err(|_| Error::TypeMismatch {
            key: key.to_string(),
            expected: "u32",
            actual: "u64 (value too large)",
        })
    }

    pub fn get_u64(&self, key: &str) -> Result<u64> {
        match self.require(key)? {
            Value::U8(v) => Ok(u64::from(*v)),
            Value::U16(v) => Ok(u64::from(*v)),
            Value::U32(v) => Ok(u64::from(*v)),
            Value::U64(v) => Ok(*v),
            other => Err(Self::mismatch(key, "unsigned int", other)),
        }
    }

    pub fn get_i64(&self, key: &str) -> Result<i64> {
        match self.require(key)? {
            Value::I8(v) => Ok(i64::from(*v)),
            Value::I16(v) => Ok(i64::from(*v)),
            Value::I32(v) => Ok(i64::from(*v)),
            Value::I64(v) => Ok(*v),
            other => Err(Self::mismatch(key, "signed int", other)),
        }
    }

    pub fn get_i32(&self, key: &str) -> Result<i32> {
        let v = self.get_i64(key)?;
        i32::try_from(v).map_err(|_| Error::TypeMismatch {
            key: key.to_string(),
            expected: "i32",
            actual: "i64 (value out of range)",
        })
    }

    pub fn get_string_array(&self, key: &str) -> Result<&[String]> {
        match self.require(key)? {
            Value::Array(Array::String(v)) => Ok(v),
            other => Err(Self::mismatch(key, "arr[str]", other)),
        }
    }

    pub fn get_array(&self, key: &str) -> Result<&Array> {
        match self.require(key)? {
            Value::Array(a) => Ok(a),
            other => Err(Self::mismatch(key, "arr", other)),
        }
    }

    /// The architecture name, which selects the model implementation.
    pub fn architecture(&self) -> Result<&str> {
        self.get_string("general.architecture")
    }

    /// Prefix a per-architecture key, e.g. `arch_key("block_count")` ->
    /// `"qwen3.block_count"`.
    pub fn arch_key(&self, suffix: &str) -> Result<String> {
        Ok(format!("{}.{}", self.architecture()?, suffix))
    }

    /// Read a key namespaced by the file's own architecture.
    pub fn get_arch_u32(&self, suffix: &str) -> Result<u32> {
        self.get_u32(&self.arch_key(suffix)?)
    }

    pub fn get_arch_f32(&self, suffix: &str) -> Result<f32> {
        self.get_f32(&self.arch_key(suffix)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn md() -> Metadata {
        let mut m = Metadata::default();
        m.insert("general.architecture".into(), Value::String("qwen3".into()))
            .unwrap();
        m.insert("qwen3.block_count".into(), Value::U32(36)).unwrap();
        m.insert("small".into(), Value::U8(7)).unwrap();
        m
    }

    #[test]
    fn missing_key_names_the_key() {
        let e = md().get_u32("qwen3.nope").unwrap_err();
        assert!(e.to_string().contains("qwen3.nope"), "{e}");
    }

    #[test]
    fn wrong_type_reports_both_types() {
        let e = md().get_string("qwen3.block_count").unwrap_err();
        let s = e.to_string();
        assert!(s.contains("expected str") && s.contains("has type u32"), "{s}");
    }

    #[test]
    fn integer_accessors_widen_but_do_not_cross_signedness() {
        let m = md();
        assert_eq!(m.get_u64("small").unwrap(), 7);
        assert_eq!(m.get_u32("small").unwrap(), 7);
        assert!(m.get_i64("small").is_err());
    }

    #[test]
    fn arch_key_is_built_from_the_file_not_a_guess() {
        assert_eq!(md().arch_key("block_count").unwrap(), "qwen3.block_count");
        assert_eq!(md().get_arch_u32("block_count").unwrap(), 36);
    }

    #[test]
    fn duplicate_key_is_rejected() {
        let mut m = md();
        assert!(m.insert("small".into(), Value::U8(1)).is_err());
    }
}

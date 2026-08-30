//! Bounds-checked little-endian cursor over the mapped file.
//!
//! GGUF is little-endian on disk. llama.cpp ships
//! `gguf-py/gguf/scripts/gguf_convert_endian.py` to produce big-endian files
//! for big-endian hosts; those are out of scope here and would be caught by the
//! magic check reading as "FUGG".
//!
//! Every read is bounds-checked and returns [`Error::Truncated`] with the
//! offset rather than panicking, so a corrupt file reports where it went wrong.

use crate::error::{Error, Result};

pub struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

macro_rules! read_le {
    ($name:ident, $ty:ty) => {
        pub fn $name(&mut self) -> Result<$ty> {
            const N: usize = std::mem::size_of::<$ty>();
            let bytes = self.take(N)?;
            let arr: [u8; N] = bytes.try_into().expect("take() returned N bytes");
            Ok(<$ty>::from_le_bytes(arr))
        }
    };
}

impl<'a> Cursor<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Advance past `n` bytes, or report exactly how far short the file fell.
    pub fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or(Error::Truncated {
            offset: self.pos,
            needed: n,
            len: self.buf.len(),
        })?;
        if end > self.buf.len() {
            return Err(Error::Truncated {
                offset: self.pos,
                needed: n,
                len: self.buf.len(),
            });
        }
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    read_le!(u8_, u8);
    read_le!(i8_, i8);
    read_le!(u16_, u16);
    read_le!(i16_, i16);
    read_le!(u32_, u32);
    read_le!(i32_, i32);
    read_le!(u64_, u64);
    read_le!(i64_, i64);
    read_le!(f32_, f32);
    read_le!(f64_, f64);

    /// GGUF bools are stored as int8. llama.cpp treats any nonzero as true.
    pub fn bool_(&mut self) -> Result<bool> {
        Ok(self.i8_()? != 0)
    }

    /// A GGUF string: u64 length, then that many bytes, no NUL terminator.
    pub fn string(&mut self, context: &str) -> Result<String> {
        let at = self.pos;
        let len = self.u64_()?;
        // Reject an absurd length before it is used as an allocation size.
        if len > (self.buf.len() - self.pos) as u64 {
            return Err(Error::StringTooLong { len, offset: at });
        }
        let bytes = self.take(len as usize)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| Error::InvalidUtf8 {
            context: context.to_string(),
            offset: at,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_little_endian_scalars() {
        let buf = [0x01, 0x00, 0x00, 0x00, 0x02, 0x00];
        let mut c = Cursor::new(&buf);
        assert_eq!(c.u32_().unwrap(), 1);
        assert_eq!(c.u16_().unwrap(), 2);
        assert_eq!(c.pos(), 6);
    }

    #[test]
    fn reads_length_prefixed_string() {
        let mut buf = 5u64.to_le_bytes().to_vec();
        buf.extend_from_slice(b"hello");
        let mut c = Cursor::new(&buf);
        assert_eq!(c.string("test").unwrap(), "hello");
    }

    #[test]
    fn truncation_reports_offset_and_shortfall() {
        let buf = [0u8; 2];
        let mut c = Cursor::new(&buf);
        match c.u32_() {
            Err(Error::Truncated { offset, needed, len }) => {
                assert_eq!((offset, needed, len), (0, 4, 2));
            }
            other => panic!("expected Truncated, got {other:?}"),
        }
    }

    #[test]
    fn absurd_string_length_is_rejected_without_allocating() {
        let buf = u64::MAX.to_le_bytes();
        let mut c = Cursor::new(&buf);
        assert!(matches!(
            c.string("test"),
            Err(Error::StringTooLong { .. })
        ));
    }
}

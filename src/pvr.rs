//! `.pvr` container format support.  Not to be confused with the PVRTC texture codec, which is in
//! the `pvrtc` module.
use std::collections::HashMap;
use std::convert::TryInto;
use std::io::{self, Read, Seek, SeekFrom, Cursor};
use std::iter;
use std::ops::Range;
use std::str;
use std::i16;
use byteorder::{ReadBytesExt, ByteOrder, BE, LE};
use crate::astc;
use crate::image::Image;
use crate::pvrtc;
use crate::read_from::{ReadFrom, ReadExt};

#[derive(Clone, Debug, Default)]
struct Header {
    format: u64,
    size: (u32, u32),
    metadata_len: u32,
}


pub struct PvrFile<T> {
    file: T,
}

impl<T: Read + Seek> PvrFile<T> {
    pub fn new(file: T) -> PvrFile<T> {
        PvrFile { file }
    }

    pub fn unwrap(self) -> T {
        self.file
    }

    /// Read the magic number.  Returns `true` if the file is opposite endianness from the machine.
    fn read_magic(&mut self) -> io::Result<bool> {
        let magic = self.file.read_u32::<LE>()?;
        if magic == 0x03525650 {
            Ok(true)
        } else if magic == 0x50565203 {
            Ok(false)
        } else {
            panic!("bad magic number for PVRTC file");
        }
    }

    fn read_header<E: ByteOrder>(&mut self) -> io::Result<Header> {
        let flags = self.file.read_u32::<E>()?;
        assert!(flags == 0, "flags are not supported");
        let format = self.file.read_u64::<E>()?;
        let _colorspace = self.file.read_u32::<E>()?;
        let channel_type = self.file.read_u32::<E>()?;
        assert!(channel_type == 0, "only u8 normalized channels are supported");
        let size = (self.file.read_u32::<E>()?, self.file.read_u32::<E>()?);
        let _depth = self.file.read_u32::<E>()?;
        let num_surfaces = self.file.read_u32::<E>()?;
        assert!(num_surfaces == 1, "multiple surfaces are not supported");
        let num_faces = self.file.read_u32::<E>()?;
        assert!(num_faces == 1, "multiple faces are not supported");
        let _num_mipmaps = self.file.read_u32::<E>()?;
        // Mipmap count is ignored.  The largest mipmap comes first, and we stop reading after that.
        let metadata_len = self.file.read_u32::<E>()?;

        Ok(Header { format, size, metadata_len })
    }

    pub fn read_image(&mut self) -> io::Result<Image> {
        let flip_endian = self.read_magic()?;
        if flip_endian {
            self.read_image_content::<LE>()
        } else {
            self.read_image_content::<BE>()
        }
    }

    fn read_image_content<E: ByteOrder>(&mut self) -> io::Result<Image> {
        let header = self.read_header::<E>()?;
        self.file.seek(SeekFrom::Current(header.metadata_len as i64))?;

        match header.format {
            3 => {
                let (w, h) = header.size;
                let (bw, bh) = (w as usize / 4, h as usize / 4);
                let words = self.file.read_many::<u64>(bw * bh)?;
                Ok(pvrtc::decode(w, h, words))
            },
            34 => {
                let (w, h) = header.size;
                let (bw, bh) = (w as usize / 8, h as usize / 8);
                let words = self.file.read_many::<u128>(bw * bh)?;
                Ok(astc::decode(w, h, words))
            },
            _ => panic!("unsupported texture format {}", header.format),
        }
    }
}

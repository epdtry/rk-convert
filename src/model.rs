use std::io::{self, Read, Seek, SeekFrom, Cursor};
use std::iter;
use std::str;
use byteorder::{ReadBytesExt, LE};
use crate::read_from::{ReadFrom, ReadExt};


const SEC_SUBOBJ_RANGE: u32 = 1;
const SEC_MATERIAL: u32 = 2;
const SEC_VERTEX: u32 = 3;
const SEC_FACE: u32 = 4;
const SEC_BONE: u32 = 7;
const SEC_SUBOBJ_NAME: u32 = 16;
const SEC_VERTEX_WEIGHT: u32 = 17;


pub struct Model {
}


pub struct ModelFile<T> {
    file: T,
}

impl<T: Read + Seek> ModelFile<T> {
    pub fn new(file: T) -> ModelFile<T> {
        ModelFile { file }
    }

    pub fn unwrap(self) -> T {
        self.file
    }

    pub fn read_headers(&mut self) -> io::Result<Vec<SectionHeader>> {
        self.file.seek(SeekFrom::Start(0x50))?;
        let mut v = Vec::new();
        loop {
            let h = self.file.read_one::<SectionHeader>()?;
            if h.tag == 0 {
                break;
            }
            v.push(h);
        }
        Ok(v)
    }

    pub fn read_section<U: ReadFrom>(&mut self, header: &SectionHeader) -> io::Result<Vec<U>> {
        self.read_section_with(header, |r| r.read_one::<U>())
    }

    pub fn read_section_with<U, F: FnMut(&mut T) -> io::Result<U>>(
        &mut self,
        header: &SectionHeader,
        mut read_one: F,
    ) -> io::Result<Vec<U>> {
        self.file.seek(SeekFrom::Start(header.offset as u64))?;
        if header.count == 0 {
            return Ok(Vec::new());
        }

        if header.byte_length % header.count != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("section {}: bad layout: count {} does not divide byte length {}",
                    header.tag, header.count, header.byte_length),
            ));
        }
        let item_size = header.byte_length / header.count;

        let mut v = Vec::with_capacity(header.count as usize);

        // Read one item, and check that we read the right amount of data.
        v.push(read_one(&mut self.file)?);
        let amount_read = self.file.seek(SeekFrom::Current(0))? as u32 - header.offset;
        if amount_read != item_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("section {}: bad item size: expected {}, but got {}",
                    header.tag, amount_read, item_size),
            ));
        }

        // Read remaining items
        for _ in 1 .. header.count {
            v.push(read_one(&mut self.file)?);
        }

        let full_amount_read = self.file.seek(SeekFrom::Current(0))? as u32 - header.offset;
        assert!(full_amount_read == amount_read * item_size,
            "element reader read a variable amount of data");

        Ok(v)
    }
}



pub struct SectionHeader {
    pub tag: u32,
    pub offset: u32,
    pub count: u32,
    pub byte_length: u32,
}

impl ReadFrom for SectionHeader {
    fn read_from<R: Read + ?Sized>(r: &mut R) -> io::Result<Self> {
        Ok(SectionHeader {
            tag: r.read_one()?,
            offset: r.read_one()?,
            count: r.read_one()?,
            byte_length: r.read_one()?,
        })
    }
}


use std::io::{self, Read, Seek, SeekFrom, Cursor};
use std::iter;
use byteorder::{ReadBytesExt, LE};
use zstd;
use crate::xxtea;
use crate::read_from::{ReadFrom, ReadExt};

pub struct ArkFile<T> {
    file: T,
}

pub struct ArkEntry {
    pub filename: String,
    pub directory: String,
    pub offset: u32,
    pub content_size: u32,
    pub compressed_size: u32,
    pub encrypted_size: u32,
}

impl<T: Read + Seek> ArkFile<T> {
    pub fn new(file: T) -> ArkFile<T> {
        ArkFile { file }
    }

    pub fn unwrap(self) -> T {
        self.file
    }

    pub fn read_metadata(&mut self) -> io::Result<Vec<ArkEntry>> {
        self.file.seek(SeekFrom::Start(0))?;
        let file_count = self.file.read_u32::<LE>()? as usize;
        let metadata_offset = self.file.read_u32::<LE>()?;
        let version = self.file.read_u32::<LE>()?;
        if version < 1 || version > 3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported ARK version: {}", version),
            ));
        }

        let mut buf = Vec::new();
        self.file.seek(SeekFrom::Start(metadata_offset as u64))?;
        self.file.read_to_end(&mut buf)?;

        xxtea::decrypt(&mut buf, XXTEA_KEY);

        let mut reader: Box<dyn Read> = match version {
            1 => Box::new(Cursor::new(&buf[..])),
            2 | 3 => Box::new(zstd::stream::Decoder::new(Cursor::new(&buf[..]))?),
            _ => unreachable!(),
        };

        let mut entries = Vec::with_capacity(file_count);
        for _ in 0 .. file_count {
            entries.push(reader.read_one::<ArkEntry>()?);
        }
        Ok(entries)
    }

    pub fn read_file(&mut self, entry: &ArkEntry) -> io::Result<Vec<u8>> {
        self.file.seek(SeekFrom::Start(entry.offset as u64))?;

        let packed_size =
            if entry.encrypted_size != 0 { entry.encrypted_size }
            else { entry.compressed_size };
        let mut buf = iter::repeat(0).take(packed_size as usize).collect::<Vec<u8>>();
        self.file.read_exact(&mut buf)?;

        if entry.encrypted_size != 0 {
            xxtea::decrypt(&mut buf, XXTEA_KEY);
        }
        if entry.compressed_size != entry.content_size {
            buf.truncate(entry.compressed_size as usize);
            buf = zstd::decode_all(Cursor::new(&buf[..]))?;
        }

        buf.truncate(entry.content_size as usize);
        Ok(buf)
    }
}

impl ReadFrom for ArkEntry {
    fn read_from<R: Read + ?Sized>(r: &mut R) -> io::Result<ArkEntry> {
        let mut buf = [0; METADATA_SIZE];
        r.read_exact(&mut buf)?;

        let mut curs = Cursor::new(&buf as &[u8]);
        let filename = curs.read_fixed_str(&mut [0; 128])?;
        let directory = curs.read_fixed_str(&mut [0; 128])?;
        let offset: u32 = curs.read_one()?;
        let content_size: u32 = curs.read_one()?;
        let compressed_size: u32 = curs.read_one()?;
        let encrypted_size: u32 = curs.read_one()?;
        let _timestamp: u32 = curs.read_one()?;
        let _md5_sum: [u32; 4] = curs.read_one()?;
        let _flags: u32 = curs.read_u32::<LE>()?;
        assert!(curs.position() == METADATA_SIZE as u64);

        Ok(ArkEntry {
            filename, directory,
            offset, content_size, compressed_size, encrypted_size,
        })
    }
}



const XXTEA_KEY: [u32; 4] = [0x3d5b2a34, 0x923fff10, 0x00e346a4, 0x0c74902b];
const METADATA_SIZE: usize = 296;

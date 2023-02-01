use std::io::{self, Read, Seek, SeekFrom};
use crate::read_from::{ReadFrom, ReadExt};

#[derive(Clone, Debug, Default)]
pub struct Anim {
    pub bone_count: usize,
    pub frames: Vec<Frame>,
}

#[derive(Clone, Debug, Default)]
pub struct Frame {
    pub bones: Vec<BonePose>,
}

#[derive(Clone, Debug, Default)]
pub struct BonePose {
    pub pos: [f32; 3],
    /// `[w, i, j, k]` order, where `w` is the scalar.
    pub quat: [f32; 4],
}


pub struct AnimFile<T> {
    file: T,
}

impl<T: Read + Seek> AnimFile<T> {
    pub fn new(file: T) -> AnimFile<T> {
        AnimFile { file }
    }

    pub fn read_header(&mut self) -> io::Result<FileHeader> {
        self.file.seek(SeekFrom::Start(0x50))?;
        self.file.read_one()
    }

    pub fn read_bone_pose(&mut self, frame_type: u32) -> io::Result<BonePose> {
        assert_eq!(frame_type, 4, "only frame_type = 4 is supported");
        let (x, y, z): (i16, i16, i16) = self.file.read_one()?;
        let (a, b, c, d): (i16, i8, i8, i8) = self.file.read_one()?;
        Ok(BonePose {
            pos: [
                x as f32 / 32.,
                y as f32 / 32.,
                z as f32 / 32.,
            ],
            quat: [
                a as f32 / 32767.,
                b as f32 / 127.,
                c as f32 / 127.,
                d as f32 / 127.,
            ],
        })
    }

    pub fn read_frame(&mut self, header: &FileHeader) -> io::Result<Frame> {
        let mut v = Vec::with_capacity(header.bone_count as usize);
        for _ in 0 .. header.bone_count {
            v.push(self.read_bone_pose(header.frame_type)?);
        }
        Ok(Frame { bones: v })
    }

    pub fn read_all_frames(&mut self, header: &FileHeader) -> io::Result<Anim> {
        let mut v = Vec::with_capacity(header.frame_count as usize);
        for _ in 0 .. header.frame_count {
            v.push(self.read_frame(header)?);
        }
        Ok(Anim {
            bone_count: header.bone_count as usize,
            frames: v,
        })
    }

    pub fn read_anim(&mut self) -> io::Result<Anim> {
        let header = self.read_header()?;
        self.read_all_frames(&header)
    }
}


pub struct FileHeader {
    bone_count: u32,
    frame_count: u32,
    frame_type: u32,
}

impl ReadFrom for FileHeader {
    fn read_from<R: Read + ?Sized>(r: &mut R) -> io::Result<Self> {
        Ok(FileHeader {
            bone_count: r.read_one()?,
            frame_count: r.read_one()?,
            frame_type: r.read_one()?,
        })
    }
}

impl FileHeader {
    pub fn frame_size(&self) -> usize {
        match self.frame_type {
            1 => 22,
            2 => 14,
            3 => 19,
            4 => 11,
            _ => panic!("unknown frame type {}", self.frame_type),
        }
    }
}

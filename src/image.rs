use std::io::{self, Write};
use std::slice;
use png;

pub struct Image {
    pub size: (u32, u32),
    pub data: Box<[[u8; 4]]>,
}

impl Image {
    pub fn new(w: u32, h: u32) -> Image {
        let size = w as usize * h as usize;
        Image {
            size: (w, h),
            data: vec![[0; 4]; size].into_boxed_slice(),
        }
    }

    pub fn pixel_mut(&mut self, x: usize, y: usize) -> &mut [u8; 4] {
        assert!(0 <= x && x < self.size.0 as usize);
        assert!(0 <= y && y < self.size.1 as usize);
        let idx = y * self.size.0 as usize + x;
        &mut self.data[idx]
    }

    pub fn bytes(&self) -> &[u8] {
        unsafe {
            let byte_len = self.data.len() * 4;
            slice::from_raw_parts(self.data.as_ptr() as *const u8, byte_len)
        }
    }

    pub fn write_raw<W: Write>(&self, w: W) -> io::Result<()> {
        let mut w = w;
        w.write_all(self.bytes())
    }

    pub fn write_png<W: Write>(&self, w: W) -> io::Result<()> {
        let mut enc = png::Encoder::new(w, self.size.0, self.size.1);
        enc.set_color(png::ColorType::RGBA);
        enc.set_depth(png::BitDepth::Eight);
        let mut writer = enc.write_header()?;
        writer.write_image_data(self.bytes())?;
        Ok(())
    }
}

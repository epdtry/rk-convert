use std::mem;
use std::process::Command;
use libc::{c_int, size_t};
use crate::image::Image;

extern "C" {
    fn astcwrap_decode(
        input: *const u8,
        input_len: size_t,
        output: *mut u8,
        output_len: size_t,
        x_size: c_int,
        y_size: c_int,
    ) -> c_int;
}

pub fn decode(w: u32, h: u32, words: Vec<u128>) -> Image {
    unsafe {
        let mut img = Image::new(w, h);
        let input = words.as_ptr() as *const u8;
        let input_len = words.len() * mem::size_of::<u128>();
        let output_bytes = img.bytes_mut();
        let ok = astcwrap_decode(
            input,
            input_len,
            output_bytes.as_mut_ptr(),
            output_bytes.len(),
            w as c_int,
            h as c_int,
        );
        assert!(ok == 0, "error occurred during ASTC decoding");
        img
    }
}

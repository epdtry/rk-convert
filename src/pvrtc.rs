use std::collections::HashMap;
use std::convert::TryInto;
use std::io::{self, Read, Seek, SeekFrom, Cursor};
use std::iter;
use std::ops::Range;
use std::str;
use std::i16;
use byteorder::{ReadBytesExt, ByteOrder, BE, LE};
use crate::image::Image;


#[derive(Clone, Copy, Default)]
struct Block {
    color_a: [u8; 4],
    color_b: [u8; 4],
    modulation: [ModValue; 16],
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
struct ModValue(u8);

impl ModValue {
    pub fn new(numerator: u8, punch: bool) -> ModValue {
        assert!(numerator <= 8);
        ModValue((numerator << 1) | (punch as u8))
    }

    pub fn punch(self) -> bool {
        (self.0 & 1) != 0
    }

    pub fn numerator(self) -> u8 {
        self.0 >> 1
    }
}


pub fn decode(w: u32, h: u32, words: Vec<u64>) -> Image {
    let mut image = Image::new(w, h);
    for (i, word) in words.into_iter().enumerate() {
        let block = decode_block(word);
        unimplemented!()
    }
    image
}

fn decode_block(word: u64) -> Block {
    let mut block = Block::default();

    block.color_b = read_color((word >> 48) as u16, false);
    block.color_a = read_color((word >> 32) as u16, true);
    let punch_mode = (word & 0x10000) != 0;
    for i in 0 .. 16 {
        block.modulation[i] = match (word >> (2 * i)) & 3 {
            0 => ModValue::new(0, false),
            1 if punch_mode => ModValue::new(3, false),
            1 if !punch_mode => ModValue::new(4, false),
            2 if punch_mode => ModValue::new(5, false),
            2 if !punch_mode => ModValue::new(4, true),
            3 => ModValue::new(8, false),
            _ => unreachable!(),
        };
    }

    block
}

fn unmorton1(x: u32) -> u32 {
    let x = x & 0x55555555;
    let x = (x | (x >> 1)) & 0x33333333;
    let x = (x | (x >> 2)) & 0x0f0f0f0f;
    let x = (x | (x >> 4)) & 0x00ff00ff;
    let x = (x | (x >> 8)) & 0x0000ffff;
    x
}

fn unmorton(x: u32) -> (usize, usize) {
    (unmorton1(x) as usize, unmorton1(x >> 1) as usize)
}

fn repeat_bits(x: u8, width: u8) -> u8 {
    let bits = x & ((1 << width) - 1);
    let mut acc = 0;
    let mut pos = 8 - width as i32;
    while pos >= 0 {
        acc |= bits << pos;
        pos -= width as i32;
    }
    acc
}

fn extract_rep(x: u16, pos: u8, width: u8) -> u8 {
    repeat_bits((x >> pos) as u8, width)
}

fn extract_zero(x: u16, pos: u8, width: u8) -> u8 {
    let y = (x >> pos) & ((1 << width) - 1);
    (y as u8) << (8 - width)
}

/// Parse `x` as an RGBA color value.  If `short` is set, then the lowest bit is ignored.
fn read_color(x: u16, short: bool) -> [u8; 4] {
    let opaque = (x >> 15) & 1 != 0;
    [
        if opaque { extract_rep(x, 10, 5) } else { extract_rep(x, 8, 4) },
        if opaque { extract_rep(x, 5, 5) } else { extract_rep(x, 4, 4) },
        if !short {
            if opaque { extract_rep(x, 0, 5) } else { extract_rep(x, 0, 4) }
        } else {
            if opaque { extract_rep(x, 1, 4) } else { extract_rep(x, 1, 3) }
        },
        if opaque { 0xff } else { extract_zero(x, 12, 3) },
    ]
}

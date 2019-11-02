use std::collections::{HashSet, HashMap};
use std::convert::TryInto;
use std::io::{self, Read, Seek, SeekFrom, Cursor};
use std::iter;
use std::ops::{BitAnd, BitOr, BitXor, Shl, Shr};
use std::ops::Range;
use std::str;
use std::i16;
use byteorder::{ReadBytesExt, ByteOrder, BE, LE};
use crate::image::Image;


#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum BlockMode {
    Normal(NormalBlockMode),
    VoidExtent,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct NormalBlockMode {
    width: u8,
    height: u8,
    range: u8,
    high_prec: bool,
    dual_plane: bool,
}

pub fn decode(w: u32, h: u32, words: Vec<u128>) -> Image {
    let mut modes = HashSet::new();
    let mut part_counts = HashSet::new();
    let mut cems_seen = HashSet::new();
    let mut ranges_seen = HashSet::new();
    let mut void_hdrs_seen = HashSet::new();

    let mut image = Image::new(w, h);
    let (bw, bh) = (w as usize / 8, h as usize / 8);
    for (i, w) in words.into_iter().enumerate() {
        let (bx, by) = (i % bw, i / bw);

        let bm = decode_block_mode(w.bits(0, 11) as u16);
        modes.insert(bm);
        let bm = match bm {
            BlockMode::Normal(x) => x,
            BlockMode::VoidExtent => {
                let hdr = w.bit(9);
                void_hdrs_seen.insert(hdr);
                let color = decode_void_extent_color(w.bits(64, 64) as u64, hdr);
                write_void_extent(&mut image, (bx, by), color);
                continue;
            },
        };
        ranges_seen.insert(bm.range);

        let parts = 1 + w.bits(11, 2) as u8;
        part_counts.insert(parts);
        let part_idx = if parts > 1 { w.bits(13, 10) as u16 } else { 0 };
        let cems = if parts == 1 {
            [w.bits(13, 4) as u8, 0, 0, 0]
        } else {
            // TODO: get high bits of partition CEMs (just below weights)
            //decode_partition_cems(parts, w.bits(23, 6) as u8, 0)
            [0; 4]
        };

        for i in 0 .. parts as usize {
            cems_seen.insert(cems[i]);
        }
    }

    eprintln!("found {} distinct block modes", modes.len());
    for m in modes {
        eprintln!("  {:?}", m);
    }
    eprintln!("found {} distinct partition counts", part_counts.len());
    for p in part_counts {
        eprintln!("  {}", p);
    }
    eprintln!("found {} distinct color endpoint modes", cems_seen.len());
    for cem in cems_seen {
        eprintln!("  {}", cem);
    }
    eprintln!("found {} distinct range modes", ranges_seen.len());
    for x in ranges_seen {
        eprintln!("  {}", x);
    }
    eprintln!("found {} distinct dynamic range modes in void extent blocks", void_hdrs_seen.len());
    for x in void_hdrs_seen {
        eprintln!("  {}", x);
    }
    image
}

fn decode_partition_cems(num_parts: u8, low: u8, high: u8) -> [u8; 4] {
    let mode = low.bits(0, 2);
    if mode == 0b00 {
        return [low.bits(2, 4); 4];
    }
    let base_class = mode - 1;
    let (ms, cs) = match num_parts {
        2 => (low.bits(2, 2), (high.bits(6, 2) << 2) | low.bits(4, 2)),
        3 => (low.bits(2, 3), (high.bits(3, 5) << 1) | low.bits(5, 1)),
        4 => (low.bits(2, 4), high),
        _ => unreachable!(),
    };

    let mut cems = [0; 4];
    for i in 0 .. num_parts as usize {
        let c = cs.bits(i, 1);
        let m = ms.bits(i * 2, 2);
        cems[i] = ((base_class + c) << 2) | m;
    }
    cems
}

fn decode_block_mode(w: u16) -> BlockMode {
    if w.bits(0, 9) == 0b111111100 {
        return BlockMode::VoidExtent;
    }

    if w.bits(0, 2) == 0b00 {
        let range = (w.bits(2, 2) << 1) | w.bits(4, 1);
        let high_prec = w.bit(9);
        let dual_plane = w.bit(10);
        let (width, height) = match w.bits(7, 2) {
            0b00 => (12, 2 + w.bits(5, 2)),
            0b01 => (2 + w.bits(5, 2), 12),
            0b10 => return BlockMode::Normal(NormalBlockMode {
                width: 6 + w.bits(5, 2) as u8,
                height: 6 + w.bits(9, 2) as u8,
                range: range as u8,
                high_prec: false,
                dual_plane: false,
            }),
            0b11 => match w.bits(5, 2) {
                0b00 => (6, 10),
                0b01 => (10, 6),
                _ => panic!("invalid block mode"),
            },
            _ => unreachable!(),
        };
        BlockMode::Normal(NormalBlockMode {
            width: width as u8,
            height: height as u8,
            range: range as u8,
            high_prec,
            dual_plane,
        })
    } else {
        let range = (w.bits(0, 2) << 1) | w.bits(4, 1);
        let high_prec = w.bit(9);
        let dual_plane = w.bit(10);
        let (width, height) = match w.bits(2, 2) {
            0b00 => (4 + w.bits(7, 2), 2 + w.bits(5, 2)),
            0b01 => (8 + w.bits(7, 2), 2 + w.bits(5, 2)),
            0b10 => (2 + w.bits(5, 2), 8 + w.bits(7, 2)),
            0b11 => match w.bits(8, 1) {
                0b0 => (2 + w.bits(5, 2), 6 + w.bits(7, 1)),
                0b1 => (2 + w.bits(7, 1), 2 + w.bits(5, 2)),
                _ => unreachable!(),
            },
            _ => unreachable!(),
        };
        BlockMode::Normal(NormalBlockMode {
            width: width as u8,
            height: height as u8,
            range: range as u8,
            high_prec,
            dual_plane,
        })
    }
}

const BAD_COLOR: [u8; 4] = [0xff, 0, 0xff, 0xff];

fn decode_void_extent_color(w: u64, hdr: bool) -> [u8; 4] {
    if hdr {
        return BAD_COLOR;
    }
    [
        w.bits(8, 8) as u8,
        w.bits(24, 8) as u8,
        w.bits(40, 8) as u8,
        w.bits(56, 8) as u8,
    ]
}

fn write_void_extent(image: &mut Image, pos: (usize, usize), color: [u8; 4]) {
    let (bx, by) = (pos.0 * 8, pos.1 * 8);
    for y in 0 .. 8 {
        for x in 0 .. 8 {
            *image.pixel_mut(bx + x, by + y) = color;
        }
    }
}

trait Bitwise:
    Copy +
    BitAnd<Self, Output=Self> +
    BitOr<Self, Output=Self> +
    BitXor<Self, Output=Self> +
    Shl<usize, Output=Self> +
    Shr<usize, Output=Self> {

    fn zero() -> Self;
    fn one() -> Self;

    fn mask(start: usize, len: usize) -> Self;
    fn bits(self, start: usize, len: usize) -> Self;
    fn bit(self, start: usize) -> bool;
}


macro_rules! impl_bitwise {
    ($($T:ty)*) => {
        $( impl Bitwise for $T {
            fn zero() -> Self { 0 }
            fn one() -> Self { 1 }

            fn mask(start: usize, len: usize) -> Self {
                ((1 << len) - 1) << start
            }

            fn bits(self, start: usize, len: usize) -> Self {
                (self >> start) & ((1 << len) - 1)
            }

            fn bit(self, start: usize) -> bool {
                self & (1 << start) != 0
            }
        } )*
    };
}

impl_bitwise!(u8 u16 u32 u64 u128);

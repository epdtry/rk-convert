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

impl NormalBlockMode {
    pub fn range_encoding(self) -> IntegerEncoding {
        let (quint, trit, bits) = match (self.high_prec, self.range) {
            (false, 0b010) => (false, false, 1),
            (false, 0b011) => (false, true, 0),
            (false, 0b100) => (false, false, 2),
            (false, 0b101) => (true, false, 0),
            (false, 0b110) => (false, true, 1),
            (false, 0b111) => (false, false, 1),

            (true, 0b010) => (true, false, 1),
            (true, 0b011) => (false, true, 2),
            (true, 0b100) => (false, false, 4),
            (true, 0b101) => (true, false, 2),
            (true, 0b110) => (false, true, 3),
            (true, 0b111) => (false, false, 5),

            _ => unreachable!(),
        };
        IntegerEncoding { quint, trit, bits }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct IntegerEncoding {
    quint: bool,
    trit: bool,
    bits: u8,
}

impl IntegerEncoding {
    pub fn bits_used(self, count: usize) -> usize {
        (if self.quint { (count * 7 + 2) / 3 } else { 0 }) +
        (if self.trit { (count * 8 + 4) / 5 } else { 0 }) +
        count * self.bits as usize
    }

    pub fn max_value(self) -> u32 {
        (if self.quint { 5 } else { 1 }) *
        (if self.trit { 3 } else { 1 }) *
        (1 << self.bits)
    }

    pub fn decode_sequence(self, n: usize, w: u128) -> Vec<u8> {
        match (self.trit, self.quint) {
            (false, false) => self.decode_sequence_bits(n, w),
            (true, false) => self.decode_sequence_trits(n, w),
            (false, true) => self.decode_sequence_quints(n, w),
            (true, true) => unreachable!(),
        }
    }

    fn decode_sequence_bits(self, n: usize, w: u128) -> Vec<u8> {
        assert!(self.max_value() <= 0xff);
        let mut out = Vec::with_capacity(n);

        for i in 0 .. n {
            out.push(w.bits(self.bits as usize * i, self.bits as usize) as u8);
        }

        assert!(out.len() == n);
        out
    }

    fn decode_sequence_trits(self, n: usize, w: u128) -> Vec<u8> {
        assert!(self.max_value() <= 0xff);
        let mut out = Vec::with_capacity(n);
        let block_size = self.bits as usize * 5 + 8;

        for i in 0 .. (n + 4) / 5 {
            let mut pos = 0;
            let mut take = |m: u8| {
                assert!(m <= 8);
                let x = w.bits(i * block_size + pos, m as usize);
                pos += m as usize;
                x as u8
            };

            let mut m = [0; 5];
            let mut t = [0; 5];
            let mut tbits = 0;

            m[0] = take(self.bits);
            tbits |= take(1) << 0;
            tbits |= take(1) << 1;
            m[1] = take(self.bits);
            tbits |= take(1) << 2;
            tbits |= take(1) << 3;
            m[2] = take(self.bits);
            tbits |= take(1) << 4;
            m[3] = take(self.bits);
            tbits |= take(1) << 5;
            tbits |= take(1) << 6;
            m[4] = take(self.bits);
            tbits |= take(1) << 7;

            let c = if tbits.bits(2, 3) == 0b111 {
                t[3] = 2;
                t[4] = 2;
                (tbits.bits(5, 3) << 2) | tbits.bits(0, 2)
            } else {
                if tbits.bits(5, 2) == 0b11 {
                    t[3] = tbits.bits(7, 1);
                    t[4] = 2;
                } else {
                    t[3] = tbits.bits(5, 2);
                    t[4] = tbits.bits(7, 1);
                }
                tbits.bits(0, 5)
            };

            if c.bits(0, 2) == 0b11 {
                t[0] = (c.bits(3, 1) << 1) | (c.bits(2, 1) & !c.bits(3, 1));
                t[1] = c.bits(4, 1);
                t[2] = 2;
            } else if c.bits(2, 2) == 0b11 {
                t[0] = c.bits(0, 2);
                t[1] = 2;
                t[2] = 2;
            } else {
                t[0] = (c.bits(1, 1) << 1) | (c.bits(0, 1) & !c.bits(1, 1));
                t[1] = c.bits(2, 2);
                t[2] = c.bits(4, 1);
            }

            for j in 0..5 {
                if i * 5 + j >= n {
                    break;
                }
                out.push((t[j] << self.bits) | m[j]);
            }
        }

        assert_eq!(out.len(), n);
        out
    }

    fn decode_sequence_quints(self, n: usize, w: u128) -> Vec<u8> {
        assert!(self.max_value() <= 0xff);
        let mut out = Vec::with_capacity(n);
        let block_size = self.bits as usize * 3 + 7;

        for i in 0 .. (n + 2) / 3 {
            let mut pos = 0;
            let mut take = |m: u8| {
                assert!(m <= 8);
                let x = w.bits(i * block_size + pos, m as usize);
                pos += m as usize;
                x as u8
            };

            let mut m = [0; 3];
            let mut q = [0; 3];
            let mut qbits = 0;

            m[0] = take(self.bits);
            qbits |= take(1) << 0;
            qbits |= take(1) << 1;
            qbits |= take(1) << 2;
            m[1] = take(self.bits);
            qbits |= take(1) << 3;
            qbits |= take(1) << 4;
            m[2] = take(self.bits);
            qbits |= take(1) << 5;
            qbits |= take(1) << 6;

            if qbits.bits(1, 2) == 0b11 && qbits.bits(5, 2) == 0b00 {
                q[0] = 4;
                q[1] = 4;
                q[2] = (qbits.bits(0, 1) << 2) |
                    ((qbits.bits(4, 1) & !qbits.bits(0, 1)) << 1) |
                    (qbits.bits(3, 1) & !qbits.bits(0, 1));
            } else {
                let c = if qbits.bits(1, 2) == 0b11 {
                    q[2] = 4;
                    (qbits.bits(3, 2) << 3) |
                        ((!qbits).bits(5, 2) << 1) |
                        (qbits.bits(0, 1))
                } else {
                    q[2] = qbits.bits(5, 2);
                    qbits.bits(0, 5)
                };

                if c.bits(0, 3) == 0b101 {
                    q[0] = c.bits(3, 2);
                    q[1] = 4;
                } else {
                    q[0] = c.bits(0, 3);
                    q[1] = c.bits(3, 2);
                }
            }

            for j in 0..3 {
                if i * 3 + j >= n {
                    break;
                }
                out.push((q[j] << self.bits) | m[j]);
            }
        }

        assert!(out.len() == n);
        out
    }

    pub fn unquantize(self, n: u8, max: u8) -> u8 {
        ((n as u32 * max as u32 + self.max_value() / 2) / self.max_value()) as u8
    }
}


pub fn decode(w: u32, h: u32, words: Vec<u128>) -> Image {
    let mut modes = HashSet::new();
    let mut cems_seen = HashMap::new();
    let mut ranges_seen = HashSet::new();
    let mut void_hdrs_seen = HashSet::new();

    let mut image = Image::new(w, h);
    let (bw, bh) = (w as usize / 8, h as usize / 8);
    for (i, w) in words.into_iter().enumerate() {
        let (bx, by) = (i % bw, i / bw);

        // Decode block mode

        let bm = decode_block_mode(w.bits(0, 11) as u16);
        modes.insert(bm);
        let bm = match bm {
            BlockMode::Normal(x) => x,
            BlockMode::VoidExtent => {
                // Void-extent blocks are much simpler and can be decoded immediately.
                let hdr = w.bit(9);
                void_hdrs_seen.insert(hdr);
                let color = decode_void_extent_color(w.bits(64, 64) as u64, hdr);
                write_void_extent(&mut image, (bx, by), color);
                continue;
            },
        };
        ranges_seen.insert(bm.range);

        // Extract raw subfields: partition count + index, weight bitstream, CEM bits, CCS mode,
        // color endpoint bits.

        let parts = 1 + w.bits(11, 2) as u8;
        let part_index = if parts == 1 {
            0
        } else {
            w.bits(13, 10) as u16
        };

        let num_weights =
            bm.width as usize * bm.height as usize * if bm.dual_plane { 2 } else { 1 };
        let num_weight_bits = bm.range_encoding().bits_used(num_weights);
        // Bitstream of weights.  It's stored in reverse order, starting from the top of `w`.
        let s = w.reverse_bits().bits(0, num_weight_bits);

        // Current position, for reading bits from just below the weights.  This is updated as we
        // consume the additional fields.
        let mut top = 128 - num_weight_bits;

        // Collect the bits of the CEM field.  The output `cem_bits` looks like one of these:
        //
        //  | 13 | 12 | 11 | 10 |  9 |  8 |  7 |  6 |  5 |  4 |  3 |  2 |  1 |  0 |
        //  |                                       |           raw CEM |   00    |
        //  |                             |   M1    |   M0    | C1 | C0 | 01..11  |
        //  |   M3    |   M2    |   M1    |   M0    | C3 | C2 | C1 | C0 | 01..11  |
        //
        // depending on number of partitions.  Note for 1 partition, we add the trailing 00 for
        // consistency with the multi-partition encoding.
        //
        // This may take some bits from `top` as a side effect.
        let cem_bits = {
            let cem_high = if parts == 1 {
                0
            } else {
                let size = 3 * parts as usize - 4;
                let shift = 3 * (4 - parts);
                top -= size;
                w.bits(top, size) as u16 >> shift
            };
            let cem_low = if parts == 1 {
                (w.bits(13, 4) as u16) << 2
            } else {
                w.bits(23, 6) as u16
            };
            (cem_high << 6) | cem_low
        };

        // Read the CCS mode, if dual_plane is enabled.
        let ccs_mode = if !bm.dual_plane {
            None
        } else {
            top -= 2;
            Some(w.bits(top, 2) as u8)
        };

        let bottom = if parts == 1 { 17 } else { 29 };
        let num_endpoint_bits = top - bottom;
        let endpoint_bits = w.bits(bottom, num_endpoint_bits);

        // Decode CEMs and color endpoints

        let cems = decode_partition_cems(parts, cem_bits);
        for i in 0 .. parts as usize {
            *cems_seen.entry(cems[i]).or_insert(0) += 1;
        }

        // For each partition, add up the number of raw values used to compute the endpoint pair
        // for that partition.
        let num_endpoint_values = (0 .. parts as usize)
            .map(|i| 2 * ((cems[i] >> 2) + 1) as usize)
            .sum();
        let endpoint_encoding = find_endpoint_encoding(num_endpoint_values, num_endpoint_bits);
        let endpoint_vs =
            endpoint_encoding.decode_sequence(num_endpoint_values, endpoint_bits)
                .into_iter()
                .map(|x| endpoint_encoding.unquantize(x, 255))
                .collect::<Vec<_>>();

        let (e0, e1) = decode_endpoint(cems[0], &endpoint_vs);
        write_void_extent(&mut image, (bx, by), e1);

        //if parts > 1 { continue; } // TODO

        //let endpoint_vs = decode_integer_sequence(

        //let cem = cems[0];
        //let num_values = 2 * (cem >> 2) as usize;

    }

    /*
    eprintln!("found {} distinct block modes", modes.len());
    for m in modes {
        eprintln!("  {:?}", m);
    }
    */
    eprintln!("found {} distinct color endpoint modes", cems_seen.len());
    for (cem, count) in cems_seen {
        eprintln!("  {} ({})", cem, count);
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

fn decode_partition_cems(parts: u8, w: u16) -> [u8; 4] {
    let mode = w.bits(0, 2) as u8;
    if mode == 0b00 {
        return [w.bits(2, 4) as u8; 4];
    }
    let base_class = mode - 1;

    let cs = w.bits(2, parts as usize);
    let ms = w.bits(2 + parts as usize, 2 * parts as usize);

    let mut cems = [0; 4];
    for i in 0 .. parts as usize {
        let c = cs.bits(i, 1) as u8;
        let m = ms.bits(i * 2, 2) as u8;
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


static VALID_ENDPOINT_ENCODINGS: [IntegerEncoding; 11] = [
    IntegerEncoding { quint: false, trit: true, bits: 1 },
    IntegerEncoding { quint: true, trit: false, bits: 1 },
    IntegerEncoding { quint: false, trit: true, bits: 2 },
    IntegerEncoding { quint: true, trit: false, bits: 2 },
    IntegerEncoding { quint: false, trit: true, bits: 3 },
    IntegerEncoding { quint: true, trit: false, bits: 3 },
    IntegerEncoding { quint: false, trit: true, bits: 4 },
    IntegerEncoding { quint: true, trit: false, bits: 4 },
    IntegerEncoding { quint: false, trit: true, bits: 5 },
    IntegerEncoding { quint: true, trit: false, bits: 5 },
    IntegerEncoding { quint: false, trit: true, bits: 6 },
];

fn find_endpoint_encoding(values: usize, bits: usize) -> IntegerEncoding {
    for enc in VALID_ENDPOINT_ENCODINGS.iter().rev() {
        if enc.bits_used(values) <= bits {
            return *enc;
        }
    }
    panic!("not enough bits ({}) to encode {} endpoint values?", bits, values)
}


fn decode_endpoint(cem: u8, v: &[u8]) -> ([u8; 4], [u8; 4]) {
    match cem {
        0 => ([v[0], v[0], v[0], 0xff], [v[1], v[1], v[1], 0xff]),
        6 => ([
                ((v[0] as u16 * v[3] as u16) >> 8) as u8,
                ((v[1] as u16 * v[3] as u16) >> 8) as u8,
                ((v[2] as u16 * v[3] as u16) >> 8) as u8,
                0xff,
            ],
            [v[0], v[1], v[2], 0xff],
        ),
        8 => {
            let s0 = v[0] as u16 + v[2] as u16 + v[4] as u16;
            let s1 = v[1] as u16 + v[3] as u16 + v[5] as u16;
            if s1 >= s0 {
                ([v[0], v[2], v[4], 0xff], [v[1], v[3], v[5], 0xff])
            } else {
                (
                    blue_contract(v[1], v[3], v[5], 0xff),
                    blue_contract(v[0], v[2], v[4], 0xff),
                )
            }
        },
        _ => ([0; 4], [0; 4]),
    }
}

fn blue_contract(r: u8, g: u8, b: u8, a: u8) -> [u8; 4] {
    [
        ((r as u16 + b as u16) >> 1) as u8,
        ((g as u16 + b as u16) >> 1) as u8,
        b,
        a,
    ]
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

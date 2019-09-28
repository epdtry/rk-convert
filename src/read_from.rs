use std::io::{self, Read};
use std::str;
use byteorder::{ReadBytesExt, LE};

pub trait ReadFrom: Sized {
    fn read_from<R: Read + ?Sized>(r: &mut R) -> io::Result<Self>;
}

macro_rules! read_byteorder {
    ($($ty:ty, $read_one:ident;)*) => {
        $(
            impl ReadFrom for $ty {
                fn read_from<R: Read + ?Sized>(r: &mut R) -> io::Result<Self> {
                    r.$read_one::<LE>()
                }
            }
        )*
    };
}

read_byteorder! {
    u16, read_u16;
    u32, read_u32;
    u64, read_u64;
    u128, read_u128;
    i16, read_i16;
    i32, read_i32;
    i64, read_i64;
    i128, read_i128;
    f32, read_f32;
    f64, read_f64;
}

impl ReadFrom for u8 {
    fn read_from<R: Read + ?Sized>(r: &mut R) -> io::Result<Self> {
        r.read_u8()
    }
}

impl ReadFrom for i8 {
    fn read_from<R: Read + ?Sized>(r: &mut R) -> io::Result<Self> {
        r.read_i8()
    }
}

macro_rules! read_array {
    ($($N:expr)*) => {
        $(
            impl<T: ReadFrom> ReadFrom for [T; $N] {
                fn read_from<R: Read + ?Sized>(r: &mut R) -> io::Result<Self> {
                    let mut v = Vec::with_capacity($N);
                    for _ in 0 .. $N {
                        v.push(T::read_from(r)?);
                    }
                    unsafe {
                        v.set_len(0);
                        Ok((v.as_ptr() as *const [T; $N]).read())
                    }
                }
            }
        )*
    };
}

read_array! {
    0 1 2 3 4 5 6 7 8 9
    10 11 12 13 14 15 16 17 18 19
    20 21 22 23 24 25 26 27 28 29
    30 31 32
}

macro_rules! read_tuple {
    ($($($A:ident)*;)*) => {
        $(
            impl<$($A: ReadFrom,)*> ReadFrom for ($($A,)*) {
                #[allow(non_snake_case)]
                fn read_from<R: Read + ?Sized>(r: &mut R) -> io::Result<Self> {
                    $( let $A: $A = $A::read_from(r)?; )*
                    Ok(($($A,)*))
                }
            }
        )*
    };
}

read_tuple! {
    ;
    A;
    A B;
    A B C;
    A B C D;
    A B C D E;
    A B C D E F;
    A B C D E F G;
    A B C D E F G H;
    A B C D E F G H I;
    A B C D E F G H I J;
}


pub trait ReadExt: Read {
    fn read_one<T: ReadFrom>(&mut self) -> io::Result<T> {
        T::read_from(self)
    }

    fn read_many<T: ReadFrom>(&mut self, n: usize) -> io::Result<Vec<T>> {
        let mut v = Vec::with_capacity(n);
        for _ in 0 .. n {
            v.push(self.read_one()?);
        }
        Ok(v)
    }

    /// Read a fixed-length string that is passed with zeros to the length of `buf`.
    fn read_fixed_str(&mut self, buf: &mut [u8]) -> io::Result<String> {
        self.read_exact(buf)?;
        let end = buf.iter().position(|&x| x == 0).unwrap_or(buf.len());
        let s = str::from_utf8(&buf[..end])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(s.to_owned())
    }
}

impl<R: Read + ?Sized> ReadExt for R {}

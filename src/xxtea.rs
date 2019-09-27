use std::num::Wrapping;
use std::slice;

pub fn decrypt(msg: &mut [u8], key: [u32; 4]) {
    assert!(msg.len() % 4 == 0, "xxtea requires message length to be divisible by 4");

    let data = unsafe {
        slice::from_raw_parts_mut(msg.as_ptr() as *mut Wrapping<u32>, msg.len() / 4)
    };

    const MAGIC: Wrapping<u32> = Wrapping(0x9e3779b9);

    // TODO: on a big-endian machine, we'd need to flip the order of each word before & after

    for i in (1 .. 7 + 52 / data.len()).rev() {
        for j in (0 .. data.len()).rev() {
            let ii = Wrapping(i as u32);
            let jj = Wrapping(j as u32);
            let x = data[(j + 1) % data.len()];
            let y = data[(data.len() + j - 1) % data.len()];

            let a1 = (x << 2) ^ (y >> 5);
            let a2 = (x >> 3) ^ (y << 4);
            let b1 = (ii * MAGIC) ^ x;
            let index = jj ^ ((ii * MAGIC) >> 2);
            let b2 = Wrapping(key[index.0 as usize & 3]) ^ y;

            data[j] -= (a1 + a2) ^ (b1 + b2);
        }
    }
}

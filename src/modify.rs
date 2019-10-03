use std::mem;
use crate::model::{Model, Object};

pub fn flip_axes(o: &mut Object) {
    for m in &mut o.models {
        for v in &mut m.verts {
            v.pos.swap(1, 2);
            v.pos[2] *= -1.;
        }
    }

    let flip_mat = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, -1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    for b in &mut o.bones {
        b.matrix = mat_mul(&flip_mat, &b.matrix);
    }
}

pub fn scale(o: &mut Object, c: f32) {
    for m in &mut o.models {
        for v in &mut m.verts {
            for x in &mut v.pos {
                *x *= c;
            }
        }
    }

    let scale_mat = [
        c, 0.0, 0.0, 0.0,
        0.0, c, 0.0, 0.0,
        0.0, 0.0, c, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    for b in &mut o.bones {
        // This order of multiplication results in scaling the bone origins as well as their actual
        // sizes.
        b.matrix = mat_mul(&scale_mat, &b.matrix);
    }
}

pub fn scale_bones(o: &mut Object, c: f32) {
    let scale_mat = [
        c, 0.0, 0.0, 0.0,
        0.0, c, 0.0, 0.0,
        0.0, 0.0, c, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    for b in &mut o.bones {
        // This order of multiplication affects only the sizes, not the origins.
        b.matrix = mat_mul(&b.matrix, &scale_mat);
    }
}

/// Computes `a * b`.  Matrices should be stored in column-major order.
fn mat_mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0; 16];
    for i in 0 .. 4 {
        for j in 0 .. 4 {
            let mut acc = 0.0;
            for k in 0 .. 4 {
                acc += a[k * 4 + j] * b[i * 4 + k];
            }
            out[i * 4 + j] = acc;
        }
    }
    out
}

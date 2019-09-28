use std::mem;
use crate::model::Model;

pub fn flip_axes(m: &mut Model) {
    for v in &mut m.verts {
        v.pos.swap(1, 2);
        v.pos[2] *= -1.;
    }
}

pub fn scale(m: &mut Model, c: f32) {
    for v in &mut m.verts {
        for x in &mut v.pos {
            *x *= c;
        }
    }
}

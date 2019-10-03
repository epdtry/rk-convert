use std::cmp;
use std::collections::{BTreeMap, HashMap};
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


pub fn prune_verts(o: &mut Object) {
    for m in &mut o.models {
        prune_model_verts(m);
    }
}

pub fn prune_model_verts(m: &mut Model) {
    let mut used = vec![false; m.verts.len()];
    for t in &m.tris {
        for &i in t {
            used[i] = true;
        }
    }

    let count = used.iter().map(|&b| if b { 1 } else { 0 }).sum();
    eprintln!("reducing model from {} to {} verts", used.len(), count);

    let mut verts = Vec::with_capacity(count);
    let mut idx_map = HashMap::with_capacity(count);
    for (i, b) in used.into_iter().enumerate() {
        if b {
            idx_map.insert(i, verts.len());
            verts.push(m.verts[i].clone());
        }
    }

    m.verts = verts;

    for t in &mut m.tris {
        for i in t {
            *i = idx_map[&*i];
        }
    }

}


pub fn split_connected_components(o: &mut Object) {
    let mut ccs = Vec::new();
    for m in &o.models {
        ccs.extend(get_connected_components(m));
    }
    o.models = ccs;
}

pub fn get_connected_components(m: &Model) -> Vec<Model> {
    eprintln!("splitting ccs on model with {} verts", m.verts.len());
    if m.verts.len() == 0 {
        return Vec::new();
    }

    // Assign a group number to each vertex.
    let mut group = (0 .. m.verts.len()).collect::<Vec<_>>();

    fn rep(group: &mut [usize], x: usize) -> usize {
        if group[x] == x {
            return x;
        }
        let parent = group[x];
        let r = rep(group, parent);
        if r != parent {
            group[x] = r;
        }
        r
    }

    fn join(group: &mut [usize], a: usize, b: usize) {
        let aa = rep(group, a);
        let bb = rep(group, b);
        let mm = cmp::min(aa, bb);
        group[aa] = mm;
        group[bb] = mm;
    }

    fn join3(group: &mut [usize], a: usize, b: usize, c: usize) {
        let aa = rep(group, a);
        let bb = rep(group, b);
        let cc = rep(group, c);
        let mm = cmp::min(cmp::min(aa, bb), cc);
        group[aa] = mm;
        group[bb] = mm;
        group[cc] = mm;
    }

    // Vertices of the same tri are assigned the same group.
    for t in &m.tris {
        join3(&mut group, t[0], t[1], t[2]);
    }

    // Vertices that have (nearly) the same position get assigned the same group.
    let mut bbox_min = m.verts[0].pos;
    let mut bbox_max = m.verts[0].pos;
    for v in &m.verts {
        for i in 0..3 {
            if v.pos[i] < bbox_min[i] {
                bbox_min[i] = v.pos[i];
            }
            if v.pos[i] > bbox_max[i] {
                bbox_max[i] = v.pos[i];
            }
        }
    }
    let approx_pos = |pos: [f32; 3]| {
        let mut a = [0_i8; 3];
        for i in 0..3 {
            a[i] = ((pos[i] - bbox_min[i]) / (bbox_max[i] - bbox_min[i])) as i8;
        }
        a
    };

    fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
        a.iter().zip(b.iter()).map(|(&aa, &bb)| (aa - bb) * (aa - bb)).sum()
    }

    let mut by_pos = HashMap::<_, Vec<_>>::new();
    for (i, v) in m.verts.iter().enumerate() {
        by_pos.entry(approx_pos(v.pos)).or_default().push(i);
    }

    for (i, v) in m.verts.iter().enumerate() {
        let [x, y, z] = approx_pos(v.pos);
        for xx in x - 1 .. x + 2 {
            for yy in y - 1 .. y + 2 {
                for zz in z - 1 .. z + 2 {
                    if let Some(js) = by_pos.get(&[xx, yy, zz]) {
                        for &j in js {
                            if i == j || rep(&mut group, i) == rep(&mut group, j) {
                                continue;
                            }
                            let d2 = dist2(v.pos, m.verts[j].pos);
                            if d2 < 1e-12 {
                                join(&mut group, i, j);
                            }
                        }
                    }
                }
            }
        }
    }


    let mut tri_lists = BTreeMap::<_, Vec<_>>::new();
    for t in &m.tris {
        let a = rep(&mut group, t[0]);
        tri_lists.entry(a).or_default().push(t.clone());
    }


    let mut models = Vec::with_capacity(tri_lists.len());
    for (i, (_, tris)) in tri_lists.into_iter().enumerate() {
        models.push(Model {
            name: format!("{}-cc{}", m.name, i),
            verts: m.verts.clone(),
            tris: tris,
        });
    }

    models
}

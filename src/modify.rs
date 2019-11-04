use std::cmp;
use std::collections::{BTreeMap, HashMap};
use std::mem;
use crate::model::{Model, Object, Vertex};

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
    let mut group = UnionFind::new(m.verts.len());

    // Vertices of the same tri are assigned the same group.
    for t in &m.tris {
        group.join3(t[0], t[1], t[2]);
    }

    // Vertices that have (nearly) the same position get assigned the same group.
    let nvs = NearbyVerts::index(&m.verts);

    for (i, v) in m.verts.iter().enumerate() {
        nvs.for_each_nearby_vertex(v.pos, 1e-12, |j, _| {
            if i == j || group.rep(i) == group.rep(j) {
                return;
            }
            group.join(i, j);
        });
    }


    let mut tri_lists = BTreeMap::<_, Vec<_>>::new();
    for t in &m.tris {
        let a = group.rep(t[0]);
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

pub fn merge_nearby_verts(o: &mut Object) {
    o.models = o.models.iter().map(merge_nearby_verts_model).collect();
}

pub fn merge_nearby_verts_model(m: &Model) -> Model {
    let nvs = NearbyVerts::index(&m.verts);
    let mut group = UnionFind::new(m.verts.len());

    for (i, v) in m.verts.iter().enumerate() {
        nvs.for_each_nearby_vertex(v.pos, 1e-12, |j, w| {
            if dist2([v.uv[0], v.uv[1], 0.0], [w.uv[0], w.uv[1], 0.0]) < 1e-12 {
                group.join(i, j);
            }
        });
    }

    let changed = (0 .. m.verts.len()).filter(|&i| group.rep(i) != i).count();
    eprintln!("removed {} verts by merging", changed);

    let mut new_tris = Vec::with_capacity(m.tris.len());
    for &tri in &m.tris {
        new_tris.push([
            group.rep(tri[0]),
            group.rep(tri[1]),
            group.rep(tri[2]),
        ]);
    }

    Model {
        name: m.name.clone(),
        verts: m.verts.clone(),
        tris: new_tris,
    }
}


struct NearbyVerts<'a> {
    by_pos: HashMap<[i8; 3], Vec<usize>>,
    vs: &'a [Vertex],
    bbox_min: [f32; 3],
    bbox_max: [f32; 3],
}

impl<'a> NearbyVerts<'a> {
    pub fn index(vs: &'a [Vertex]) -> NearbyVerts<'a> {
        let (bbox_min, bbox_max) = compute_bbox(vs);
        let mut nvs = NearbyVerts {
            by_pos: HashMap::new(),
            vs, bbox_min, bbox_max,
        };

        for (i, v) in nvs.vs.iter().enumerate() {
            let apos = nvs.approx_pos(v.pos);
            nvs.by_pos.entry(apos).or_default().push(i);
        }

        nvs
    }

    fn approx_pos(&self, pos: [f32; 3]) -> [i8; 3] {
        let mut a = [0_i8; 3];
        for i in 0..3 {
            a[i] = ((pos[i] - self.bbox_min[i]) * 100.0 / (self.bbox_max[i] - self.bbox_min[i])) as i8;
        }
        a
    }

    pub fn for_each_nearby_vertex(
        &self,
        pos: [f32; 3],
        max_dist2: f32,
        mut f: impl FnMut(usize, &'a Vertex),
    ) {
        let [x, y, z] = self.approx_pos(pos);
        for xx in x - 1 .. x + 2 {
            for yy in y - 1 .. y + 2 {
                for zz in z - 1 .. z + 2 {
                    if let Some(js) = self.by_pos.get(&[xx, yy, zz]) {
                        for &j in js {
                            let d2 = dist2(pos, self.vs[j].pos);
                            if d2 < max_dist2 {
                                f(j, &self.vs[j]);
                            }
                        }
                    }
                }
            }
        }
    }
}

fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    a.iter().zip(b.iter()).map(|(&aa, &bb)| (aa - bb) * (aa - bb)).sum()
}

fn compute_bbox(vs: &[Vertex]) -> ([f32; 3], [f32; 3]) {
    let mut bbox_min = vs[0].pos;
    let mut bbox_max = vs[0].pos;
    for v in vs {
        for i in 0..3 {
            if v.pos[i] < bbox_min[i] {
                bbox_min[i] = v.pos[i];
            }
            if v.pos[i] > bbox_max[i] {
                bbox_max[i] = v.pos[i];
            }
        }
    }
    (bbox_min, bbox_max)
}


struct UnionFind {
    group: Vec<usize>,
}

impl UnionFind {
    pub fn new(len: usize) -> UnionFind {
        UnionFind {
            group: (0 .. len).collect(),
        }
    }

    pub fn rep(&mut self, x: usize) -> usize {
        if self.group[x] == x {
            return x;
        }
        let parent = self.group[x];
        let r = self.rep(parent);
        if r != parent {
            self.group[x] = r;
        }
        r
    }

    pub fn join(&mut self, a: usize, b: usize) {
        let aa = self.rep(a);
        let bb = self.rep(b);
        let mm = cmp::min(aa, bb);
        self.group[aa] = mm;
        self.group[bb] = mm;
    }

    pub fn join3(&mut self, a: usize, b: usize, c: usize) {
        let aa = self.rep(a);
        let bb = self.rep(b);
        let cc = self.rep(c);
        let mm = cmp::min(cmp::min(aa, bb), cc);
        self.group[aa] = mm;
        self.group[bb] = mm;
        self.group[cc] = mm;
    }
}

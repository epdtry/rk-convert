use std::cmp;
use std::collections::{HashMap, HashSet};
use std::env;
use std::f32::consts::PI;
use std::ffi::OsStr;
use std::fs::{self, File};
use std::io;
use std::path::Path;
use gltf_json::{
    Root, Index, Node, Mesh, Scene, Accessor, Buffer, Skin, Animation, Image, Texture, Material,
};
use gltf_json::accessor::{self, ComponentType, GenericComponentType};
use gltf_json::animation;
use gltf_json::buffer::{self, View};
use gltf_json::extensions;
use gltf_json::image::MimeType;
use gltf_json::material::{
    self, PbrMetallicRoughness, PbrBaseColorFactor, StrengthFactor, AlphaMode, EmissiveFactor,
};
use gltf_json::mesh::{self, Primitive, Semantic};
use gltf_json::scene;
use gltf_json::texture;
use gltf_json::validation::Checked;
use nalgebra::{Vector3, Vector4, Matrix3, Matrix4, Rotation, Quaternion, UnitQuaternion};
use png;
use rk_convert::anim::{AnimFile, BonePose};
use rk_convert::anim_csv::{self, AnimRange};
use rk_convert::anim_xml::{self, AnimObjects, EyeMode};
use rk_convert::model::ModelFile;
use rk_convert::modify;
use rk_convert::pvr::PvrFile;


pub struct GltfBuilder {
    root: Root,
    bin: Vec<u8>,
    bin_buffer: Index<Buffer>,
}

impl Default for GltfBuilder {
    fn default() -> GltfBuilder {
        let mut root = Root::default();
        root.buffers.push(Buffer {
            byte_length: 0,
            uri: None,
            name: None,
            extensions: None,
            extras: Default::default(),
        });
        GltfBuilder {
            root,
            bin: Vec::new(),
            bin_buffer: Index::new(0),
        }
    }
}

impl GltfBuilder {
    pub fn push_node(&mut self, node: Node) -> Index<Node> {
        let i = Index::new(self.root.nodes.len() as u32);
        self.root.nodes.push(node);
        i
    }

    pub fn push_mesh(&mut self, mesh: Mesh) -> Index<Mesh> {
        let i = Index::new(self.root.meshes.len() as u32);
        self.root.meshes.push(mesh);
        i
    }

    pub fn push_accessor(&mut self, accessor: Accessor) -> Index<Accessor> {
        let i = Index::new(self.root.accessors.len() as u32);
        self.root.accessors.push(accessor);
        i
    }

    pub fn push_view(&mut self, view: View) -> Index<View> {
        let i = Index::new(self.root.buffer_views.len() as u32);
        self.root.buffer_views.push(view);
        i
    }

    pub fn push_scene(&mut self, scene: Scene) -> Index<Scene> {
        let i = Index::new(self.root.scenes.len() as u32);
        self.root.scenes.push(scene);
        i
    }

    pub fn push_skin(&mut self, skin: Skin) -> Index<Skin> {
        let i = Index::new(self.root.skins.len() as u32);
        self.root.skins.push(skin);
        i
    }

    pub fn push_animation(&mut self, animation: Animation) -> Index<Animation> {
        let i = Index::new(self.root.animations.len() as u32);
        self.root.animations.push(animation);
        i
    }

    pub fn push_image(&mut self, image: Image) -> Index<Image> {
        let i = Index::new(self.root.images.len() as u32);
        self.root.images.push(image);
        i
    }

    pub fn push_texture(&mut self, texture: Texture) -> Index<Texture> {
        let i = Index::new(self.root.textures.len() as u32);
        self.root.textures.push(texture);
        i
    }

    pub fn push_material(&mut self, material: Material) -> Index<Material> {
        let i = Index::new(self.root.materials.len() as u32);
        self.root.materials.push(material);
        i
    }


    pub fn node(&self, idx: Index<Node>) -> &Node {
        &self.root.nodes[idx.value()]
    }

    pub fn node_mut(&mut self, idx: Index<Node>) -> &mut Node {
        &mut self.root.nodes[idx.value()]
    }


    pub fn add_extension(&mut self, name: String, required: bool) {
        if required {
            self.root.extensions_required.push(name.clone());
        }
        self.root.extensions_used.push(name);
    }


    pub fn set_default_scene(&mut self, scene_idx: Index<Scene>) {
        self.root.scene = Some(scene_idx);
    }

    pub fn push_bin_view(
        &mut self,
        data: &[u8],
        target: Option<buffer::Target>,
    ) -> Index<View> {
        let offset = self.bin.len();
        self.bin.extend_from_slice(data);
        if data.len() % 4 != 0 {
            for i in (data.len() % 4) .. 4 {
                self.bin.push(0);
            }
        }

        self.push_view(View {
            buffer: self.bin_buffer,
            byte_length: data.len() as u32,
            byte_offset: Some(offset as u32),
            byte_stride: None,
            target: target.map(Checked::Valid),
            name: None,
            extensions: None,
            extras: Default::default(),
        })
    }

    pub fn push_prim_accessor<T: PrimType + std::fmt::Debug>(
        &mut self,
        data: &[T],
        buffer_target: Option<buffer::Target>,
        normalized: bool,
    ) -> Index<Accessor> {
        let byte_len = data.len() * T::SIZE;

        let offset = self.bin.len();
        self.bin.reserve(byte_len);
        let mut min: Option<T> = None;
        let mut max: Option<T> = None;
        for &x in data {
            x.push_bytes(&mut self.bin);
            min = Some(match min {
                Some(old) => old.componentwise_min(x),
                None => x,
            });
            max = Some(match max {
                Some(old) => old.componentwise_max(x),
                None => x,
            });
        }
        assert_eq!(self.bin.len() - offset, byte_len);
        while self.bin.len() % 4 != 0 {
            self.bin.push(0);
        }

        let view_idx = self.push_view(View {
            buffer: self.bin_buffer,
            byte_length: byte_len as u32,
            byte_offset: Some(offset as u32),
            byte_stride: None,
            target: buffer_target.map(Checked::Valid),
            name: None,
            extensions: None,
            extras: Default::default(),
        });
        self.push_accessor(Accessor {
            buffer_view: Some(view_idx),
            byte_offset: 0,
            count: data.len() as u32,
            component_type: Checked::Valid(T::COMPONENT_TYPE),
            type_: Checked::Valid(T::TYPE),
            min: min.map(|x| x.to_min_max_value()),
            max: max.map(|x| x.to_min_max_value()),
            normalized,
            sparse: None,
            name: None,
            extensions: None,
            extras: Default::default(),
        })
    }


    pub fn finish(mut self) -> Vec<u8> {
        self.root.buffers[self.bin_buffer.value()].byte_length = self.bin.len() as u32;

        let mut out = Vec::new();

        // File header
        out.extend_from_slice(b"glTF");
        out.extend_from_slice(&2_u32.to_le_bytes());
        let file_len_pos = out.len();
        out.extend_from_slice(&0_u32.to_le_bytes());

        // JSON chunk header
        let json_len_pos = out.len();
        out.extend_from_slice(&0_u32.to_le_bytes());
        out.extend_from_slice(b"JSON");
        let start = out.len();
        self.root.to_writer(&mut out).unwrap();
        while out.len() % 4 != 0 {
            out.push(b' ');
        }
        let end = out.len();
        let len = end - start;
        out[json_len_pos .. json_len_pos + 4].copy_from_slice(&(len as u32).to_le_bytes());

        // Binary chunk header
        let bin_len_pos = out.len();
        out.extend_from_slice(&0_u32.to_le_bytes());
        out.extend_from_slice(b"BIN\0");
        let start = out.len();
        out.extend_from_slice(&self.bin);
        while out.len() % 4 != 0 {
            out.push(b' ');
        }
        let end = out.len();
        let len = end - start;
        out[bin_len_pos .. bin_len_pos + 4].copy_from_slice(&(len as u32).to_le_bytes());

        let len = out.len();
        out[file_len_pos .. file_len_pos + 4].copy_from_slice(&(len as u32).to_le_bytes());

        out
    }
}


pub trait PrimType: Copy {
    const COMPONENT_TYPE: GenericComponentType;
    const TYPE: accessor::Type;
    const SIZE: usize;
    fn push_bytes(self, v: &mut Vec<u8>);
    fn componentwise_min(self, other: Self) -> Self;
    fn componentwise_max(self, other: Self) -> Self;
    fn to_min_max_value(self) -> gltf_json::Value;
}

impl PrimType for f32 {
    const COMPONENT_TYPE: GenericComponentType = GenericComponentType(ComponentType::F32);
    const TYPE: accessor::Type = accessor::Type::Scalar;
    const SIZE: usize = 4;
    fn push_bytes(self, v: &mut Vec<u8>) {
        v.extend_from_slice(&self.to_le_bytes());
    }
    fn componentwise_min(self, other: Self) -> Self {
        if other < self { other } else { self }
    }
    fn componentwise_max(self, other: Self) -> Self {
        if other > self { other } else { self }
    }
    fn to_min_max_value(self) -> gltf_json::Value { (&[self] as &[_]).into() }
}

fn iter_column_major(m: Matrix4<f32>) -> impl Iterator<Item = f32> {
    (0..4).flat_map(move |j| {
        (0..4).map(move |i| {
            m[(i, j)]
        })
    })
}

fn to_column_major(m: Matrix4<f32>) -> [f32; 16] {
    let mut out = [0.; 16];
    for (x, y) in iter_column_major(m).zip(out.iter_mut()) {
        *y = x;
    }
    out
}

impl PrimType for Matrix4<f32> {
    const COMPONENT_TYPE: GenericComponentType = GenericComponentType(ComponentType::F32);
    const TYPE: accessor::Type = accessor::Type::Mat4;
    const SIZE: usize = 4 * 16;
    fn push_bytes(self, v: &mut Vec<u8>) {
        for x in iter_column_major(self) {
            x.push_bytes(v);
        }
    }
    fn componentwise_min(self, other: Self) -> Self {
        self.zip_map(&other, |x, y| if y < x { y } else { x })
    }
    fn componentwise_max(self, other: Self) -> Self {
        self.zip_map(&other, |x, y| if y > x { y } else { x })
    }
    fn to_min_max_value(self) -> gltf_json::Value { (&to_column_major(self) as &[_]).into() }
}

impl<T: PrimType> PrimType for [T; 2] {
    const COMPONENT_TYPE: GenericComponentType = T::COMPONENT_TYPE;
    const TYPE: accessor::Type = accessor::Type::Vec2;
    const SIZE: usize = T::SIZE * 2;
    fn push_bytes(self, v: &mut Vec<u8>) {
        for &x in &self {
            x.push_bytes(v);
        }
    }
    fn componentwise_min(self, other: Self) -> Self {
        [
            self[0].componentwise_min(other[0]),
            self[1].componentwise_min(other[1]),
        ]
    }
    fn componentwise_max(self, other: Self) -> Self {
        [
            self[0].componentwise_max(other[0]),
            self[1].componentwise_max(other[1]),
        ]
    }
    fn to_min_max_value(self) -> gltf_json::Value {
        let mut v = Vec::new();
        for x in &self {
            let x_arr = x.to_min_max_value();
            for y in x_arr.as_array().unwrap().iter() {
                v.push(y.clone());
            }
        }
        v.into()
    }
}

impl<T: PrimType> PrimType for [T; 3] {
    const COMPONENT_TYPE: GenericComponentType = T::COMPONENT_TYPE;
    const TYPE: accessor::Type = accessor::Type::Vec3;
    const SIZE: usize = T::SIZE * 3;
    fn push_bytes(self, v: &mut Vec<u8>) {
        for &x in &self {
            x.push_bytes(v);
        }
    }
    fn componentwise_min(self, other: Self) -> Self {
        [
            self[0].componentwise_min(other[0]),
            self[1].componentwise_min(other[1]),
            self[2].componentwise_min(other[2]),
        ]
    }
    fn componentwise_max(self, other: Self) -> Self {
        [
            self[0].componentwise_max(other[0]),
            self[1].componentwise_max(other[1]),
            self[2].componentwise_max(other[2]),
        ]
    }
    fn to_min_max_value(self) -> gltf_json::Value {
        let mut v = Vec::new();
        for x in &self {
            let x_arr = x.to_min_max_value();
            for y in x_arr.as_array().unwrap().iter() {
                v.push(y.clone());
            }
        }
        v.into()
    }
}

impl<T: PrimType> PrimType for [T; 4] {
    const COMPONENT_TYPE: GenericComponentType = T::COMPONENT_TYPE;
    const TYPE: accessor::Type = accessor::Type::Vec4;
    const SIZE: usize = T::SIZE * 4;
    fn push_bytes(self, v: &mut Vec<u8>) {
        for &x in &self {
            x.push_bytes(v);
        }
    }
    fn componentwise_min(self, other: Self) -> Self {
        [
            self[0].componentwise_min(other[0]),
            self[1].componentwise_min(other[1]),
            self[2].componentwise_min(other[2]),
            self[3].componentwise_min(other[3]),
        ]
    }
    fn componentwise_max(self, other: Self) -> Self {
        [
            self[0].componentwise_max(other[0]),
            self[1].componentwise_max(other[1]),
            self[2].componentwise_max(other[2]),
            self[3].componentwise_max(other[3]),
        ]
    }
    fn to_min_max_value(self) -> gltf_json::Value {
        let mut v = Vec::new();
        for x in &self {
            let x_arr = x.to_min_max_value();
            for y in x_arr.as_array().unwrap().iter() {
                v.push(y.clone());
            }
        }
        v.into()
    }
}

impl PrimType for u16 {
    const COMPONENT_TYPE: GenericComponentType = GenericComponentType(ComponentType::U16);
    const TYPE: accessor::Type = accessor::Type::Scalar;
    const SIZE: usize = 2;
    fn push_bytes(self, v: &mut Vec<u8>) {
        v.extend_from_slice(&self.to_le_bytes());
    }
    fn componentwise_min(self, other: Self) -> Self { cmp::min(self, other) }
    fn componentwise_max(self, other: Self) -> Self { cmp::max(self, other) }
    fn to_min_max_value(self) -> gltf_json::Value { (&[self] as &[_]).into() }
}

impl PrimType for u32 {
    const COMPONENT_TYPE: GenericComponentType = GenericComponentType(ComponentType::U32);
    const TYPE: accessor::Type = accessor::Type::Scalar;
    const SIZE: usize = 4;
    fn push_bytes(self, v: &mut Vec<u8>) {
        v.extend_from_slice(&self.to_le_bytes());
    }
    fn componentwise_min(self, other: Self) -> Self { cmp::min(self, other) }
    fn componentwise_max(self, other: Self) -> Self { cmp::max(self, other) }
    fn to_min_max_value(self) -> gltf_json::Value { (&[self] as &[_]).into() }
}


fn decompose_bone_matrix(m: Matrix4<f32>) -> (Vector3<f32>, UnitQuaternion<f32>, Vector3<f32>) {
    let translate_vec4 = m * Vector4::new(0., 0., 0., 1.);
    let translate: Vector3<f32> = translate_vec4.remove_row(3);

    let mat3: Matrix3<f32> = m.remove_row(3).remove_column(3);

    let rotate = UnitQuaternion::from(Rotation::from_matrix_unchecked(mat3));

    let rotate_inv_mat = rotate.inverse().to_rotation_matrix();
    let scale_mat = mat3 * rotate_inv_mat;
    let scale = scale_mat.diagonal();

    (translate, rotate, scale)
}

fn compose_bone_matrix(t: Vector3<f32>, r: UnitQuaternion<f32>, s: Vector3<f32>) -> Matrix4<f32> {
    Matrix4::new_translation(&t) * Matrix4::from(r) * Matrix4::new_nonuniform_scaling(&s)
}

fn calc_pose_matrix(pose: &BonePose) -> Matrix4<f32> {
    let [a, b, c, d] = pose.quat;
    let [x, y, z] = pose.pos;
    compose_bone_matrix(
        Vector3::new(x, y, z),
        UnitQuaternion::from_quaternion(Quaternion::new(a, b, c, d)),
        Vector3::new(1., 1., 1.),
    )
}


fn main() -> io::Result<()> {
    // Load model

    let args = env::args_os().collect::<Vec<_>>();
    assert!(
        args.len() == 2 || args.len() == 3,
        "usage: {} <model.rk> [anim.csv|anim.anim]",
        args[0].to_string_lossy(),
    );

    let model_path = Path::new(&args[1]);
    let anim_path = args.get(2).map(|s| Path::new(s));

    eprintln!("load object from {}", model_path.display());
    let mut mf = ModelFile::new(File::open(model_path)?);
    let mut o = mf.read_object()?;

    eprintln!("bones:");
    for b in &o.bones {
        if let Some(i) = b.parent {
            eprintln!("  {}, parent = {}", b.name, o.bones[i].name);
        } else {
            eprintln!("  {}", b.name);
        }
    }

    // TODO: read anim.xml as well to get subobject visibility info
    let (mut anim, anim_ranges, anim_objs) = if let Some(anim_path) = anim_path {
        if anim_path.extension() == Some(OsStr::new("csv")) {
            let ranges = anim_csv::read_anim_csv(anim_path)?;
            let mut af = AnimFile::new(File::open(anim_path.with_extension("anim"))?);

            let xml_path = anim_path.with_extension("xml");
            let objs = if xml_path.exists() {
                eprintln!("read anim xml from {:?}", xml_path);
                anim_xml::read_anim_xml(xml_path)?
            } else {
                AnimObjects::default()
            };

            (Some(af.read_anim()?), ranges, objs)
        } else {
            let mut af = AnimFile::new(File::open(anim_path)?);
            let anim = af.read_anim()?;
            let ranges = vec![AnimRange {
                name: "all".into(),
                start: 0,
                end: anim.frames.len(),
                frame_rate: 15,
            }];
            (Some(anim), ranges, AnimObjects::default())
        }
    } else {
        (None, Vec::new(), AnimObjects::default())
    };

    let mut material_images = HashMap::new();
    for m in &o.models {
        eprintln!("model {} material = {}", m.name, m.material);
        if m.material.len() == 0 {
            continue;
        }
        if material_images.contains_key(&m.material) {
            continue;
        }
        let texture_path = model_path.with_file_name(format!("{}.pvr", m.material));
        eprintln!("  load {} from {}", m.material, texture_path.display());
        // TODO: read {name}.rkm and extract DiffuseTexture name
        let mut pf = PvrFile::new(File::open(texture_path)?);
        let img = pf.read_image()?;
        material_images.insert(m.material.clone(), img);
    }


    // Flip axes
    //
    // The .rk model format seems to use +X right, -Y up, -Z forward for its coordinate system.
    // glTF instead specifies -X right, +Y up, +Z forward.  We first mirror along the Y axis to
    // convert to the proper handedness, then rotate 180 around the Y axis to correct the other two
    // axes.

    fn flip_vector(v: Vector3<f32>) -> Vector3<f32> {
        Vector3::new(
            -v[0],
            -v[1],
            -v[2],
        )
    }
    fn flip_vector_array(arr: [f32; 3]) -> [f32; 3] {
        [-arr[0], -arr[1], -arr[2]]
    }
    fn flip_quaternion(q: UnitQuaternion<f32>) -> UnitQuaternion<f32> {
        // Flip Y axis
        let q = UnitQuaternion::from_quaternion(Quaternion::new(
            q.scalar(),
            -q.vector()[0],
            q.vector()[1],
            -q.vector()[2],
        ));
        // Rotate 180 degrees around Y axis
        let q = UnitQuaternion::from_euler_angles(0., PI, 0.) * q;
        q
    }
    fn flip_quaternion_array(arr: [f32; 4]) -> [f32; 4] {
        let [w, i, j, k] = arr;
        let q = UnitQuaternion::from_quaternion(Quaternion::new(w, i, j, k));
        let q = flip_quaternion(q);
        [
            q.scalar(),
            q.vector()[0],
            q.vector()[1],
            q.vector()[2],
        ]
    }

    for m in &mut o.models {
        for v in &mut m.verts {
            v.pos = flip_vector_array(v.pos);
        }
    }
    for b in &mut o.bones {
        let bone_mat = Matrix4::from_column_slice(&b.matrix);
        let (t, r, s) = decompose_bone_matrix(bone_mat);
        let t = flip_vector(t);
        let r = flip_quaternion(r);
        b.matrix = to_column_major(compose_bone_matrix(t, r, s));
    }
    if let Some(anim) = anim.as_mut() {
        for f in &mut anim.frames {
            for b in &mut f.bones {
                b.pos = flip_vector_array(b.pos);
                b.quat = flip_quaternion_array(b.quat);
            }
        }
    }


    // Build maps for processing object animation.

    // Maps "subobject" XML IDs to model index.
    let mut subobject_map: HashMap<String, usize> = HashMap::new();
    // Maps eye modes to their corresponding model index.
    let mut eye_map: HashMap<EyeMode, usize> = HashMap::new();
    // The set of models that are hidden by default.
    let mut default_hidden: HashSet<usize> = HashSet::new();

    let model_name_map = o.models.iter().enumerate().map(|(i, m)| (&m.name, i))
        .collect::<HashMap<_, _>>();

    if let Some(name) = model_path.file_stem() {
        let name = name.to_str()
            .unwrap_or_else(|| panic!("unsupported file name {:?}", name));
        let subobjs = anim_objs.subobjects.get(name).or_else(|| {
            // If the name ends with a `_lodN` prefix, try changing to `_lod0` through `_lod2`.
            if name.len() < 5 {
                return None;
            }
            let (a, b) = name.split_at(name.len() - 5);
            if !b.starts_with("_lod") {
                return None;
            }
            for i in 0 ..= 2 {
                let alt_name = format!("{}_lod{}", a, i);
                if let Some(subobjs) = anim_objs.subobjects.get(&alt_name) {
                    return Some(subobjs);
                }
            }
            None
        });

        if let Some(subobjs) = subobjs {
            for so in subobjs {
                if let Some(&idx) = model_name_map.get(&so.model_name) {
                    subobject_map.insert(so.xml_id.clone(), idx);
                    if !so.default_visible {
                        default_hidden.insert(idx);
                    }
                }
            }
        }
    }

    for (idx, m) in o.models.iter().enumerate() {
        if m.name.ends_with("_eyes_open") {
            eye_map.insert(EyeMode::Open, idx);
            // Leave visible by default
        } else if m.name.ends_with("_eyes_shut") {
            eye_map.insert(EyeMode::Closed, idx);
            default_hidden.insert(idx);
        } else if m.name.ends_with("_eyes_happy") {
            eye_map.insert(EyeMode::Happy, idx);
            default_hidden.insert(idx);
        } else if m.name.ends_with("_eyes_frown") {
            eye_map.insert(EyeMode::Frown, idx);
            default_hidden.insert(idx);
        }
    }

    // Indices of models with animated visibility.
    let visibility_animated_models = subobject_map.values().cloned()
        .chain(eye_map.values().cloned())
        .collect::<Vec<_>>();


    // Compute which bones need extra subbones for model visibility animations.

    // Pairs of visibility-animated model index and bone index, where the bone affects at least one
    // vertex of the model.
    let mut model_bone_pairs_vis = HashSet::<(usize, usize)>::new();
    for &i in &visibility_animated_models {
        let m = &o.models[i];
        for tri in &m.tris {
            for &j in &tri.verts {
                for bw in &m.verts[j].bone_weights {
                     if bw.weight != 0 {
                         model_bone_pairs_vis.insert((i, bw.bone));
                     }
                }
            }
        }
    }
    let mut model_bone_pairs_vis = model_bone_pairs_vis.into_iter().collect::<Vec<_>>();
    model_bone_pairs_vis.sort();
    let model_bone_pairs_vis = model_bone_pairs_vis;


    // Build GLTF

    let mut gltf = GltfBuilder::default();
    gltf.add_extension("KHR_materials_unlit".into(), true);

    // Materials

    let mut keys = material_images.keys().collect::<Vec<_>>();
    keys.sort();
    let mut material_idxs = HashMap::with_capacity(material_images.len());
    for name in keys {
        let img = material_images.get(name).unwrap();
        let png_bytes = img.to_png_vec();
        let view_idx = gltf.push_bin_view(&png_bytes, None);

        let image_idx = gltf.push_image(Image {
            buffer_view: Some(view_idx),
            mime_type: Some(MimeType("image/png".into())),
            uri: None,
            name: Some((*name).clone()),
            extensions: None,
            extras: Default::default(),
        });

        let texture_idx = gltf.push_texture(Texture {
            source: image_idx,
            sampler: None,
            name: Some((*name).clone()),
            extensions: None,
            extras: Default::default(),
        });

        let material_idx = gltf.push_material(Material {
            pbr_metallic_roughness: PbrMetallicRoughness {
                base_color_texture: Some(texture::Info {
                    index: texture_idx,
                    tex_coord: 0,
                    extensions: None,
                    extras: Default::default(),
                }),
                base_color_factor: PbrBaseColorFactor([1., 1., 1., 1.]),
                metallic_factor: StrengthFactor(0.),
                roughness_factor: StrengthFactor(1.),
                metallic_roughness_texture: None,
                extensions: None,
                extras: Default::default(),
            },
            alpha_cutoff: None,
            alpha_mode: Checked::Valid(AlphaMode::Blend),
            double_sided: false,
            normal_texture: None,
            occlusion_texture: None,
            emissive_texture: None,
            emissive_factor: EmissiveFactor([0., 0., 0.]),
            name: Some((*name).clone()),
            extensions: Some(extensions::material::Material {
                unlit: Some(extensions::material::Unlit {}),
                .. Default::default()
            }),
            extras: Default::default(),
        });

        material_idxs.insert((*name).to_owned(), material_idx);
    }

    // Bones

    let mut bone_mats = Vec::with_capacity(o.bones.len());
    let mut bone_mats_inv = Vec::with_capacity(o.bones.len());
    for b in &o.bones {
        let bone_mat = Matrix4::from_column_slice(&b.matrix);
        bone_mats.push(bone_mat);
        bone_mats_inv.push(bone_mat.try_inverse().unwrap());
    }

    let mut all_bone_nodes = Vec::with_capacity(o.bones.len());
    let mut inverse_bind_matrices_vec = Vec::with_capacity(o.bones.len());

    for (i, b) in o.bones.iter().enumerate() {
        let local_mat = match b.parent {
            None => bone_mats[i],
            Some(j) => bone_mats_inv[j] * bone_mats[i],
        };

        inverse_bind_matrices_vec.push(bone_mats_inv[i]);

        let (t, r, s) = decompose_bone_matrix(local_mat);

        let node_idx = gltf.push_node(Node {
            camera: None,
            children: None,
            matrix: None,
            mesh: None,
            rotation: Some(scene::UnitQuaternion([
                r.quaternion().vector()[0],
                r.quaternion().vector()[1],
                r.quaternion().vector()[2],
                r.quaternion().scalar(),
            ])),
            scale: Some(s.into()),
            translation: Some(t.into()),
            skin: None,
            weights: None,
            name: Some(b.name.clone()),
            extensions: None,
            extras: Default::default(),
        });
        all_bone_nodes.push(node_idx);
    }

    // `bone_nodes` contains only the nodes corresponding to bones in the original `Object`.
    // `all_bone_nodes` contains all bones, including additional ones created for model visibility
    // animations.
    let bone_nodes = all_bone_nodes.clone();

    // Set bone parents
    for (i, b) in o.bones.iter().enumerate() {
        if let Some(j) = b.parent {
            let bone_idx = bone_nodes[i];
            let parent_idx = bone_nodes[j];
            gltf.node_mut(parent_idx).children.get_or_insert_with(Vec::new).push(bone_idx);
        }
    }

    // Add visibility control bones.  We add these last so the real bones keep their original
    // indices in `bone_nodes` and `inverse_bind_matrices`.

    // Maps model index to node IDs of extra bones introduced to control visibility of the model.
    let mut model_vis_bones = HashMap::<usize, Vec<Index<Node>>>::new();
    // Maps old bone index to new bone index for certain models.  This is used to adjust vertex
    // weights so that visibility-animated models influenced by bone B will instead be influenced
    // by their visibility control bone whose parent is B.
    let mut model_bone_maps = HashMap::<usize, HashMap<usize, usize>>::new();

    for (mi, bi) in model_bone_pairs_vis {
        // The visibility control bone uses the same inverse bind matrix as its parent, so the
        // transform from world space to bone space is the same.  And by default the child bone
        // applies no transforms relative to its parent, so the transform from bone space back to
        // world space is also the same.
        let vis_bone_idx = gltf.push_node(Node {
            camera: None,
            children: None,
            matrix: None,
            mesh: None,
            rotation: None,
            scale: None,
            translation: None,
            skin: None,
            weights: None,
            name: Some(format!("{}__vis__{}", o.bones[bi].name, o.models[mi].name)),
            extensions: None,
            extras: Default::default(),
        });
        let bone_idx = bone_nodes[bi];
        gltf.node_mut(bone_idx).children.get_or_insert_with(Vec::new).push(vis_bone_idx);
        let vis_bi = all_bone_nodes.len();
        model_vis_bones.entry(mi).or_insert_with(Vec::new).push(vis_bone_idx);
        model_bone_maps.entry(mi).or_insert_with(HashMap::new).insert(bi, vis_bi);
        all_bone_nodes.push(vis_bone_idx);
        inverse_bind_matrices_vec.push(bone_mats_inv[bi]);
    }

    let bone_root_idx = gltf.push_node(Node {
        camera: None,
        children: Some(o.bones.iter().enumerate()
            .filter(|&(i, b)| b.parent.is_none())
            .map(|(i, _)| bone_nodes[i])
            .collect()),
        matrix: None,
        mesh: None,
        rotation: None,
        scale: None,
        translation: None,
        skin: None,
        weights: None,
        name: None,
        extensions: None,
        extras: Default::default(),
    });
    let inverse_bind_matrices_acc = gltf.push_prim_accessor(
        &inverse_bind_matrices_vec, None, false);
    let skin_idx = gltf.push_skin(Skin {
        joints: all_bone_nodes,
        skeleton: Some(bone_root_idx),
        inverse_bind_matrices: Some(inverse_bind_matrices_acc),
        name: None,
        extensions: None,
        extras: Default::default(),
    });

    // Meshes

    let mut model_nodes = Vec::with_capacity(o.models.len());
    for (mi, m) in o.models.iter().enumerate() {
        let mut attributes = HashMap::new();

        let pos_vec = m.tris.iter()
            .flat_map(|t| t.verts.iter())
            .map(|&i| m.verts[i].pos)
            .collect::<Vec<_>>();
        attributes.insert(Checked::Valid(Semantic::Positions),
            gltf.push_prim_accessor(&pos_vec, Some(buffer::Target::ArrayBuffer), false));

        let uv_vec = m.tris.iter()
            .flat_map(|t| t.uvs.iter().cloned())
            .map(|[u, v]| [u, 1. - v])
            .collect::<Vec<_>>();
        attributes.insert(Checked::Valid(Semantic::TexCoords(0)),
            gltf.push_prim_accessor(&uv_vec, Some(buffer::Target::ArrayBuffer), false));

        let model_bone_map = model_bone_maps.get(&mi);
        let map_bone = |bi| {
            model_bone_map.and_then(|m| m.get(&bi).cloned()).unwrap_or(bi)
        };

        // Joints and weights are specified in groups of 4.
        let joints_vec = m.tris.iter().flat_map(|t| t.verts.iter()).map(|&i| [
            map_bone(m.verts[i].bone_weights[0].bone) as u16,
            map_bone(m.verts[i].bone_weights[1].bone) as u16,
            map_bone(m.verts[i].bone_weights[2].bone) as u16,
            map_bone(m.verts[i].bone_weights[3].bone) as u16,
        ]).collect::<Vec<_>>();
        attributes.insert(Checked::Valid(Semantic::Joints(0)),
            gltf.push_prim_accessor(&joints_vec, Some(buffer::Target::ArrayBuffer), false));

        let weights_vec = m.tris.iter().flat_map(|t| t.verts.iter()).map(|&i| [
            m.verts[i].bone_weights[0].weight,
            m.verts[i].bone_weights[1].weight,
            m.verts[i].bone_weights[2].weight,
            m.verts[i].bone_weights[3].weight,
        ]).collect::<Vec<_>>();
        attributes.insert(Checked::Valid(Semantic::Weights(0)),
            gltf.push_prim_accessor(&weights_vec, Some(buffer::Target::ArrayBuffer), true));

        let prim = Primitive {
            attributes,
            indices: None,
            material: material_idxs.get(&m.material).cloned(),
            mode: Checked::Valid(mesh::Mode::Triangles),
            targets: None,
            extensions: None,
            extras: Default::default(),
        };

        let mesh_idx = gltf.push_mesh(Mesh {
            primitives: vec![prim],
            weights: None,
            name: None,
            extensions: None,
            extras: Default::default(),
        });

        let node_idx = gltf.push_node(Node {
            mesh: Some(mesh_idx),
            camera: None,
            children: None,
            matrix: None,
            rotation: None,
            scale: None,
            translation: None,
            skin: Some(skin_idx),
            weights: None,
            name: Some(m.name.clone()),
            extensions: None,
            extras: Default::default(),
        });

        model_nodes.push(node_idx);
    }

    let mut scene_nodes = model_nodes.clone();
    scene_nodes.push(bone_root_idx);
    let scene_idx = gltf.push_scene(Scene {
        nodes: scene_nodes,
        name: None,
        extensions: None,
        extras: Default::default(),
    });
    gltf.set_default_scene(scene_idx);

    // Animations

    if let Some(anim) = anim {
        for ar in &anim_ranges {
            let mut channels = Vec::with_capacity(o.bones.len());
            let mut samplers = Vec::with_capacity(o.bones.len());

            let frame_times = (0 .. ar.end - ar.start).map(|i| {
                i as f32 / ar.frame_rate as f32
            }).collect::<Vec<_>>();
            let frame_times_acc = gltf.push_prim_accessor(
                &frame_times, None, false);

            for (i, &bone_idx) in bone_nodes.iter().enumerate() {
                let mut pos_vec = Vec::with_capacity(ar.end - ar.start);
                let mut quat_vec = Vec::with_capacity(ar.end - ar.start);
                for f in ar.start .. ar.end {
                    let pose_mat = calc_pose_matrix(&anim.frames[f].bones[i]);

                    let local_pose_mat = match o.bones[i].parent {
                        None => pose_mat,
                        Some(j) => {
                            let parent_pose_mat = calc_pose_matrix(&anim.frames[f].bones[j]);
                            let parent_pose_mat_inv = parent_pose_mat.try_inverse().unwrap();
                            parent_pose_mat_inv * pose_mat
                        },
                    };

                    let (t, r, s) = decompose_bone_matrix(local_pose_mat);

                    pos_vec.push([
                        t[0],
                        t[1],
                        t[2],
                    ]);
                    quat_vec.push([
                        r.quaternion().vector()[0],
                        r.quaternion().vector()[1],
                        r.quaternion().vector()[2],
                        r.quaternion().scalar(),
                    ]);
                }

                // Rotation
                channels.push(animation::Channel {
                    sampler: Index::new(samplers.len() as u32),
                    target: animation::Target {
                        node: bone_idx,
                        path: Checked::Valid(animation::Property::Rotation),
                        extensions: None,
                        extras: Default::default(),
                    },
                    extensions: None,
                    extras: Default::default(),
                });
                samplers.push(animation::Sampler {
                    input: frame_times_acc,
                    output: gltf.push_prim_accessor(
                               &quat_vec, None, false),
                    interpolation: Checked::Valid(animation::Interpolation::Step),
                    extensions: None,
                    extras: Default::default(),
                });

                // Position
                channels.push(animation::Channel {
                    sampler: Index::new(samplers.len() as u32),
                    target: animation::Target {
                        node: bone_idx,
                        path: Checked::Valid(animation::Property::Translation),
                        extensions: None,
                        extras: Default::default(),
                    },
                    extensions: None,
                    extras: Default::default(),
                });
                samplers.push(animation::Sampler {
                    input: frame_times_acc,
                    output: gltf.push_prim_accessor(
                               &pos_vec, None, false),
                    interpolation: Checked::Valid(animation::Interpolation::Step),
                    extensions: None,
                    extras: Default::default(),
                });
            }

            // Object visibility

            let mut model_vis_anims = HashMap::<usize, Vec<(usize, bool)>>::new();
            let mut record = |frame, mi, show| {
                let vis = model_vis_anims.entry(mi).or_insert_with(Vec::new);
                if let Some(last) = vis.last_mut().filter(|& &mut (f, _)| f == frame) {
                    last.1 = show;
                } else {
                    vis.push((frame, show));
                }
            };
            for &i in &default_hidden {
                record(0, i, false);
            }
            let frame_objs = anim_objs.anims.get(&ar.name).map_or(&[] as &[_], |x| x);
            for f in frame_objs {
                if let Some(ref eye_state) = f.eye_state {
                    for (&mode, &mi) in &eye_map {
                        record(f.index, mi, mode == eye_state.mode);
                    }
                }

                for (model_name, &show) in &f.subobject_state {
                    if let Some(&mi) = subobject_map.get(model_name) {
                        record(f.index, mi, show);
                    }
                }
            }

            for (i, m) in o.models.iter().enumerate() {
                let vis_bones = match model_vis_bones.get(&i) {
                    Some(x) => x,
                    None => continue,
                };

                let vis_anims = match model_vis_anims.get(&i) {
                    Some(x) => x,
                    None => continue,
                };

                //eprintln!("visibility anims for {}, {} = {:?}", ar.name, m.name, vis_anims);

                let mut frame_times = Vec::with_capacity(ar.end - ar.start);
                let mut scale_vec = Vec::new();
                // If `vis_anims` doesn't specify the initial state, we explicitly show the model
                // on frame 0.  Otherwise, the state from the first keyframe will be used instead,
                // which is often incorrect (e.g. default is visible, but there's a keyframe on
                // frame 20 to hide it).
                if !vis_anims.get(0).map_or(false, |&(f, _)| f == 0) {
                    frame_times.push(0.);
                    scale_vec.push([1., 1., 1.]);
                }
                for &(frame, show) in vis_anims {
                    frame_times.push(frame as f32 / ar.frame_rate as f32);
                    if show {
                        scale_vec.push([1., 1., 1.]);
                    } else {
                        scale_vec.push([0., 0., 0.]);
                    }
                }

                let frame_times_acc = gltf.push_prim_accessor(
                    &frame_times, None, false);

                for &vis_bone_idx in vis_bones {
                    channels.push(animation::Channel {
                        sampler: Index::new(samplers.len() as u32),
                        target: animation::Target {
                            node: vis_bone_idx,
                            path: Checked::Valid(animation::Property::Scale),
                            extensions: None,
                            extras: Default::default(),
                        },
                        extensions: None,
                        extras: Default::default(),
                    });
                    samplers.push(animation::Sampler {
                        input: frame_times_acc,
                        output: gltf.push_prim_accessor(
                                   &scale_vec, None, false),
                        interpolation: Checked::Valid(animation::Interpolation::Step),
                        extensions: None,
                        extras: Default::default(),
                    });
                }
            }

            gltf.push_animation(Animation {
                channels,
                samplers,
                name: Some(ar.name.clone()),
                extensions: None,
                extras: Default::default(),
            });
        }
    }


    // Write output

    let gltf_bytes = gltf.finish();
    let file_name = model_path.with_extension("glb");
    fs::write(&file_name, gltf_bytes)?;
    println!("Output Filename: {}", file_name.display());



    Ok(())
}

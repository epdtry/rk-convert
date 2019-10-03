use std::collections::HashMap;
use std::convert::TryInto;
use std::io::{self, Read, Seek, SeekFrom, Cursor};
use std::iter;
use std::ops::Range;
use std::str;
use std::i16;
use byteorder::{ReadBytesExt, LE};
use crate::read_from::{ReadFrom, ReadExt};


const SEC_SUBOBJ_RANGE: u32 = 1;
const SEC_MATERIAL: u32 = 2;
const SEC_VERTEX: u32 = 3;
const SEC_FACE: u32 = 4;
const SEC_BONE: u32 = 7;
const SEC_SUBOBJ_NAME: u32 = 16;
const SEC_VERTEX_WEIGHT: u32 = 17;


#[derive(Clone, Debug, Default)]
pub struct Model {
    pub name: String,
    pub verts: Vec<Vertex>,
    pub tris: Vec<[usize; 3]>,
}

#[derive(Clone, Debug, Default)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub uv: [i16; 2],
    pub bone_weights: [BoneWeight; 4],
}

#[derive(Clone, Debug, Default)]
pub struct BoneWeight {
    pub bone: usize,
    pub weight: u16,
}

#[derive(Clone, Debug, Default)]
pub struct Bone {
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub name: String,
    pub matrix: [f32; 16],
}

#[derive(Clone, Debug, Default)]
pub struct Object {
    pub models: Vec<Model>,
    pub bones: Vec<Bone>,
    pub material: Option<String>,
}


pub struct ModelFile<T> {
    file: T,
}

impl<T: Read + Seek> ModelFile<T> {
    pub fn new(file: T) -> ModelFile<T> {
        ModelFile { file }
    }

    pub fn unwrap(self) -> T {
        self.file
    }

    pub fn read_headers(&mut self) -> io::Result<Vec<SectionHeader>> {
        self.file.seek(SeekFrom::Start(0x50))?;
        let mut v = Vec::new();
        loop {
            let h = self.file.read_one::<SectionHeader>()?;
            if h.tag == 0 {
                break;
            }
            v.push(h);
        }
        Ok(v)
    }

    pub fn read_section<U: ReadFrom>(&mut self, header: &SectionHeader) -> io::Result<Vec<U>> {
        self.read_section_with(header, |r| r.read_one::<U>())
    }

    pub fn read_section_with<U, F: FnMut(&mut T) -> io::Result<U>>(
        &mut self,
        header: &SectionHeader,
        mut read_one: F,
    ) -> io::Result<Vec<U>> {
        self.file.seek(SeekFrom::Start(header.offset as u64))?;
        if header.count == 0 {
            return Ok(Vec::new());
        }

        if header.byte_length % header.count != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("section {}: bad layout: count {} does not divide byte length {}",
                    header.tag, header.count, header.byte_length),
            ));
        }
        let item_size = header.byte_length / header.count;

        let mut v = Vec::with_capacity(header.count as usize);

        // Read one item, and check that we read the right amount of data.
        v.push(read_one(&mut self.file)?);
        let amount_read = self.file.seek(SeekFrom::Current(0))? as u32 - header.offset;
        if amount_read != item_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("section {}: bad item size: expected {}, but got {}",
                    header.tag, amount_read, item_size),
            ));
        }

        // Read remaining items
        for _ in 1 .. header.count {
            v.push(read_one(&mut self.file)?);
        }

        let full_amount_read = self.file.seek(SeekFrom::Current(0))? as u32 - header.offset;
        assert!(full_amount_read == item_size * header.count,
            "element reader read a variable amount of data");

        Ok(v)
    }

    pub fn read_tagged_section<U: ReadFrom>(
        &mut self,
        headers: &[SectionHeader],
        tag: u32,
    ) -> io::Result<Vec<U>> {
        if let Some(h) = headers.iter().find(|h| h.tag == tag) {
            self.read_section(h)
        } else {
            Ok(Vec::new())
        }
    }

    pub fn read_tagged_section_with<U, F: FnMut(&mut T) -> io::Result<U>>(
        &mut self,
        headers: &[SectionHeader],
        tag: u32,
        read_one: F,
    ) -> io::Result<Vec<U>> {
        if let Some(h) = headers.iter().find(|h| h.tag == tag) {
            self.read_section_with(h, read_one)
        } else {
            Ok(Vec::new())
        }
    }

    pub fn read_object(&mut self) -> io::Result<Object> {
        let mut m = Model::default();
        let mut o = Object::default();

        let mut headers = self.read_headers()?;

        let verts: Vec<([f32; 3], [i16; 2])> = self.read_tagged_section(&headers, SEC_VERTEX)?;
        let indices: Vec<u16> = self.read_tagged_section(&headers, SEC_FACE)?;
        let part_names: Vec<String> =
            self.read_tagged_section_with(&headers, SEC_SUBOBJ_NAME, |r| {
                r.read_fixed_str(&mut [0; 64])
            })?;
        let part_face_ranges: Vec<Range<usize>> =
            self.read_tagged_section_with(&headers, SEC_SUBOBJ_RANGE, |r| {
                let count = r.read_one::<u32>()? as usize;
                let offset = r.read_one::<u32>()? as usize / 3;
                let _unk = r.read_one::<[u32; 2]>()?;
                Ok(offset .. offset + count)
            })?;
        let raw_bones: Vec<RawBone> = self.read_tagged_section(&headers, SEC_BONE)?;
        let vert_weights: Vec<([u8; 4], [u16; 4])> =
            self.read_tagged_section(&headers, SEC_VERTEX_WEIGHT)?;
        let materials: Vec<([String; 6], [f32; 4])> =
            self.read_tagged_section_with(&headers, SEC_MATERIAL, |r| {
                let strs: Vec<_> = iter::from_fn(|| Some(r.read_fixed_str(&mut [0; 64])))
                    .take(6).collect::<io::Result<Vec<_>>>()?;
                let strs: &[_; 6] = (&strs as &[_]).try_into().unwrap();
                let vals = r.read_one::<[f32; 4]>()?;
                Ok((strs.clone(), vals))
            })?;

        // Parse basic model elements

        m.verts.reserve(verts.len());
        for (pos, uv) in verts {
            m.verts.push(Vertex { pos, uv, .. Vertex::default() });
        }

        assert!(indices.len() % 3 == 0, "expected a multiple of three indices");
        m.tris.reserve(indices.len() / 3);
        for xs in indices.chunks_exact(3) {
            m.tris.push([xs[0] as usize, xs[1] as usize, xs[2] as usize]);
        }

        // Parse bones and bone weights

        o.bones.resize_with(raw_bones.len(), Default::default);
        let bone_id_map = raw_bones.iter().enumerate()
            .map(|(i, rb)| (rb.id, i))
            .collect::<HashMap<_, _>>();
        assert!(bone_id_map.len() == raw_bones.len(),
            "duplicate IDs in bone list");
        for rb in raw_bones {
            let idx = bone_id_map[&rb.id];
            let parent_idx: Option<usize>;
            {
                let b = &mut o.bones[idx];
                b.parent = match rb.parent {
                    0xffffffff => None,
                    parent_id => Some(bone_id_map.get(&parent_id).cloned()
                        .unwrap_or_else(|| panic!("bad parent bone ID: {:#x}", parent_id))),
                };
                b.matrix = rb.matrix;
                b.name = rb.name;
                parent_idx = b.parent;
            }

            if let Some(parent_idx) = parent_idx {
                o.bones[parent_idx].children.push(idx);
            }
        }

        if vert_weights.len() > 0 {
            assert!(vert_weights.len() == m.verts.len());
            for (vert, (b, w)) in m.verts.iter_mut().zip(vert_weights.into_iter()) {
                for i in 0..4 {
                    let bone_id = b[i];
                    vert.bone_weights[i].bone = bone_id_map[&(bone_id as u32)];
                    vert.bone_weights[i].weight = w[i];
                }
            }
        }

        // Parse subobject info and build separate Models

        assert!(part_names.len() == part_face_ranges.len());
        for (part_name, face_range) in part_names.into_iter().zip(part_face_ranges.into_iter()) {
            let m2 = Model {
                name: part_name,
                verts: m.verts.clone(),
                tris: m.tris[face_range].to_owned(),
            };
            o.models.push(m2);
        }

        // Parse other Object-level info

        if materials.len() != 0 {
            assert!(materials.len() == 1, "expected only one set of materials");
            o.material = materials[0].0.iter().find(|s| s.len() > 0).cloned();
            if let Some(ref name) = o.material {
                assert!(materials[0].0.iter().all(|s| s.len() == 0 || s == name),
                    "expected all material names to match, but got {:?}", materials[0].0);
            }
        }

        Ok(o)
    }
}

pub struct SectionHeader {
    pub tag: u32,
    pub offset: u32,
    pub count: u32,
    pub byte_length: u32,
}

impl ReadFrom for SectionHeader {
    fn read_from<R: Read + ?Sized>(r: &mut R) -> io::Result<Self> {
        Ok(SectionHeader {
            tag: r.read_one()?,
            offset: r.read_one()?,
            count: r.read_one()?,
            byte_length: r.read_one()?,
        })
    }
}

pub struct RawBone {
    parent: u32,
    id: u32,
    num_children: u32,
    matrix: [f32; 16],
    name: String,
}

impl ReadFrom for RawBone {
    fn read_from<R: Read + ?Sized>(r: &mut R) -> io::Result<Self> {
        Ok(RawBone {
            parent: r.read_one()?,
            id: r.read_one()?,
            num_children: r.read_one()?,
            matrix: r.read_one()?,
            name: r.read_fixed_str(&mut [0; 64])?,
        })
    }
}

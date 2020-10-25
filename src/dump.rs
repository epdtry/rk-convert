use std::io::{self, Write};
use std::u16;
use byteorder::{WriteBytesExt, LE};
use crate::model::{Model, Object};

pub fn dump_object<W: Write>(w: &mut W, o: &Object) -> io::Result<()> {
    w.write_u32::<LE>(o.models.len() as u32)?;
    w.write_u32::<LE>(o.bones.len() as u32)?;

    for m in &o.models {
        dump_model(w, m)?;
    }

    for b in &o.bones {
        write_str(w, &b.name)?;
        w.write_u32::<LE>(b.parent.map_or(0xffff_ffff, |x| x as u32))?;
        for &x in &b.matrix {
            w.write_f32::<LE>(x)?;
        }
    }

    Ok(())
}

pub fn dump_model<W: Write>(w: &mut W, m: &Model) -> io::Result<()> {
    let mut weights = Vec::new();
    for (i, v) in m.verts.iter().enumerate() {
        for bw in v.bone_weights.iter() {
            if bw.weight != 0 {
                weights.push((i, bw.bone, bw.weight as f32 / u16::MAX as f32));
            }
        }
    }

    w.write_u32::<LE>(m.verts.len() as u32)?;
    w.write_u32::<LE>(m.tris.len() as u32)?;
    w.write_u32::<LE>(weights.len() as u32)?;

    write_str(w, &m.name)?;
    // TODO: material
    //write_str(w, &m.material)?;

    for v in &m.verts {
        for &x in &v.pos {
            w.write_f32::<LE>(x)?;
        }
    }

    for t in &m.tris {
        for &i in &t.verts {
            w.write_u32::<LE>(i as u32)?;
        }
        for &uv in &t.uvs {
            w.write_f32::<LE>(uv[0])?;
            w.write_f32::<LE>(uv[1])?;
        }
    }

    for (v, b, wt) in weights {
        w.write_u32::<LE>(v as u32)?;
        w.write_u32::<LE>(b as u32)?;
        w.write_f32::<LE>(wt)?;
    }

    Ok(())
}

pub fn write_str<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    w.write_u16::<LE>(s.len() as u16)?;
    w.write_all(s.as_bytes())?;
    Ok(())
}

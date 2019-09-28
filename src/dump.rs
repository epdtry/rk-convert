use std::io::{self, Write};
use byteorder::{WriteBytesExt, LE};
use crate::model::Model;

pub fn dump_model<W: Write>(w: &mut W, m: &Model) -> io::Result<()> {
    w.write_u32::<LE>(m.verts.len() as u32)?;
    w.write_u32::<LE>(m.tris.len() as u32)?;
    w.write_u32::<LE>(m.bones.len() as u32)?;

    for v in &m.verts {
        for &x in &v.pos {
            w.write_f32::<LE>(x)?;
        }
    }

    for t in &m.tris {
        for &i in t {
            w.write_u32::<LE>(i as u32)?;
        }
    }

    for b in &m.bones {
        write_str(w, &b.name)?;
        w.write_u32::<LE>(b.parent.map_or(0xffff_ffff, |x| x as u32))?;
        for &x in &b.matrix {
            w.write_f32::<LE>(x)?;
        }
    }

    Ok(())
}

pub fn write_str<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    w.write_u16::<LE>(s.len() as u16)?;
    w.write_all(s.as_bytes())?;
    Ok(())
}

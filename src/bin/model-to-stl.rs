use std::env;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use rkengine::model::ModelFile;
use rkengine::modify;

fn io_main() -> io::Result<()> {
    let args = env::args_os().collect::<Vec<_>>();
    assert!(args.len() == 2, "usage: {} <file.rk>", args[0].to_string_lossy());
    let mut mf = ModelFile::new(File::open(&args[1])?);
    let mut o = mf.read_object()?;
    modify::flip_axes(&mut o);
    modify::scale(&mut o, 1./3.);
    modify::prune_verts(&mut o);
    modify::split_connected_components(&mut o);
    modify::prune_verts(&mut o);

    let stem = Path::new(&args[1]).file_stem().unwrap().to_str().unwrap();
    for m in &o.models {
        let name = m.name.replace(|c: char| !c.is_ascii_alphanumeric(), "_");
        let file_name = format!("{}-{}.stl", stem, name);
        eprintln!("generating {}", file_name);
        let mut f = File::create(file_name)?;
        writeln!(f, "solid {}", name).unwrap();
        for tri in &m.tris {
            writeln!(f, "facet normal {:e} {:e} {:e}", 0.0, 0.0, 0.0)?;
            writeln!(f, " outer loop")?;
            for &i in &tri.verts {
                let [x,y,z] = m.verts[i].pos;
                writeln!(f, "  vertex {:e} {:e} {:e}", x, y, z)?;
            }
            writeln!(f, " endloop")?;
            writeln!(f, "endfacet")?;
        }
        writeln!(f, "endsolid {}", name)?;
    }

    Ok(())
}

fn main() {
    io_main().unwrap();
}

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
    let mut m = mf.read_model()?;
    modify::flip_axes(&mut m);
    modify::scale(&mut m, 1./3.);

    let stem = Path::new(&args[1]).file_stem().unwrap().to_str().unwrap();
    println!("solid {}", stem);
    for idxs in &m.tris {
        println!("facet normal {:e} {:e} {:e}", 0.0, 0.0, 0.0);
        println!(" outer loop");
        for &i in idxs {
            let [x,y,z] = m.verts[i].pos;
            println!("  vertex {:e} {:e} {:e}", x, y, z);
        }
        println!(" endloop");
        println!("endfacet");
    }
    println!("endsolid {}", stem);

    Ok(())
}

fn main() {
    io_main().unwrap();
}

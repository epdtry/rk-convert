use std::env;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use rkengine::model::ModelFile;

fn io_main() -> io::Result<()> {
    let args = env::args_os().collect::<Vec<_>>();
    assert!(args.len() == 2, "usage: {} <file.rk>", args[0].to_string_lossy());
    let mut mf = ModelFile::new(File::open(&args[1])?);
    let m = mf.read_model()?;
    println!("got {} verts, {} tris", m.verts.len(), m.tris.len());
    println!("material: {:?}", m.mat_name);
    Ok(())
}

fn main() {
    io_main().unwrap();
}

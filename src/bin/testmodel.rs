use std::env;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use rkengine::model::{ModelFile, Bone};

fn io_main() -> io::Result<()> {
    let args = env::args_os().collect::<Vec<_>>();
    assert!(args.len() == 2, "usage: {} <file.rk>", args[0].to_string_lossy());
    let mut mf = ModelFile::new(File::open(&args[1])?);
    let m = mf.read_model()?;
    println!("got {} verts, {} tris", m.verts.len(), m.tris.len());
    println!("material: {:?}", m.mat_name);

    println!("bone hierarchy:");
    fn print_bone(bs: &[Bone], i: usize, depth: usize) {
        println!("{1:0$}{2}", depth * 2, "", bs[i].name);
        for j in bs[i].children.iter().cloned() {
            print_bone(bs, j, depth + 1);
        }
    }
    for i in 0 .. m.bones.len() {
        if m.bones[i].parent.is_none() {
            print_bone(&m.bones, i, 1);
        }
    }

    Ok(())
}

fn main() {
    io_main().unwrap();
}

use std::env;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use rk_convert::model::ModelFile;
use rk_convert::dump;
use rk_convert::modify;

fn io_main() -> io::Result<()> {
    let args = env::args_os().collect::<Vec<_>>();
    assert!(args.len() == 3, "usage: {} <input.rk> <output.model>", args[0].to_string_lossy());

    let mut mf = ModelFile::new(File::open(&args[1])?);
    let mut o = mf.read_object()?;
    modify::flip_axes(&mut o);
    modify::scale(&mut o, 1./3.);
    modify::scale_bones(&mut o, 8.);
    modify::prune_verts(&mut o);
    eprintln!("splitting {} models", o.models.len());
    modify::split_connected_components(&mut o);
    eprintln!("got {} models", o.models.len());
    modify::prune_verts(&mut o);

    let mut of = File::create(Path::new(&args[2]))?;
    dump::dump_object(&mut of, &o)?;
    eprintln!("dumped {} models", o.models.len());

    Ok(())
}

fn main() {
    io_main().unwrap();
}

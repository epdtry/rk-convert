use std::env;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use rkengine::model::ModelFile;
use rkengine::dump;
use rkengine::modify;

fn io_main() -> io::Result<()> {
    let args = env::args_os().collect::<Vec<_>>();
    assert!(args.len() == 3, "usage: {} <input.rk> <output.model>", args[0].to_string_lossy());

    let mut mf = ModelFile::new(File::open(&args[1])?);
    let mut m = mf.read_model()?;
    modify::flip_axes(&mut m);
    modify::scale(&mut m, 1./3.);
    modify::scale_bones(&mut m, 8.);

    let mut of = File::create(Path::new(&args[2]))?;
    dump::dump_model(&mut of, &m)?;

    Ok(())
}

fn main() {
    io_main().unwrap();
}
